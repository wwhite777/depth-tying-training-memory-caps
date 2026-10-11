"""S3 data tests for src/lb/prepare_data.py outputs and the gated loader src/lb/data.py.

Run after prepare_data.py (from the repository root):
  CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_data.py -v
The tests read the outputs and, for the layout and normalization recomputations, the raw archives in data/raw/
(the HAR zip, the CIFAR-10 tarball and the 12-class Speech Commands test archive; never the 2.4 GB training archive).
"""
import io
import json
import os
import pathlib
import pickle
import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from scipy.io import wavfile  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lb import data, prereg  # noqa: E402

torch.set_num_threads(4)
TASKS = ("HAR", "KWS", "CIFAR10")
SPLITS = ("train", "val", "test")
MAN = json.loads(data.MANIFEST.read_text())
RAW = ROOT / "data" / "raw"
KWS_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "_unknown_", "_silence_"]
EXPECTED_SIZES_CIFAR = {"train": 45000, "val": 5000, "test": 10000}
HAR_N_VAL_VOLUNTEERS = 4
FRACTION_TOL_CLIPS = 1.0
NORM_TOL = 1e-5
_CACHE = {}


def get(task, split):
    if (task, split) not in _CACHE:
        _CACHE[(task, split)] = data.load_task(task, split, allow_test=True)
    return _CACHE[(task, split)]


def meta(task, split):
    return data.load_meta(task, split, allow_test=True)


def channels_last(task, X):
    """View X as rows of one raw channel each: HAR (N*128, 9), KWS (N*49, 10), CIFAR (N*64*16, 3)."""
    X = X.double()
    if task == "HAR":
        return X.reshape(-1, 9)
    if task == "KWS":
        return X.reshape(-1, 10)
    return X.reshape(-1, 3)


def har_raw():
    z = zipfile.ZipFile(RAW / "human+activity+recognition+using+smartphones.zip")
    if not any("Inertial Signals/" in n for n in z.namelist()):
        z = zipfile.ZipFile(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("UCI HAR Dataset.zip")][0])))
    names = [n for n in z.namelist() if "__MACOSX" not in n]
    chans = [f"{s}_{a}" for s in ("body_acc", "body_gyro", "total_acc") for a in "xyz"]

    def rd(suffix):
        return np.loadtxt(io.BytesIO(z.read([n for n in names if n.endswith(suffix)][0])))
    out = {}
    for part in ("train", "test"):
        out[part] = (np.stack([rd(f"{part}/Inertial Signals/{c}_{part}.txt") for c in chans], -1),
                     rd(f"{part}/subject_{part}.txt").astype(int), rd(f"{part}/y_{part}.txt").astype(int) - 1)
    return out


def cifar_raw():
    out = {}
    with tarfile.open(RAW / "cifar-10-python.tar.gz", "r:gz") as tf:
        for m in tf.getmembers():
            b = m.name.split("/")[-1]
            if b.startswith("data_batch_") or b == "test_batch":
                out[b] = pickle.load(tf.extractfile(m), encoding="bytes")
    tr = np.concatenate([out[f"data_batch_{i}"][b"data"] for i in range(1, 6)]).reshape(-1, 3, 32, 32)
    ytr = np.concatenate([out[f"data_batch_{i}"][b"labels"] for i in range(1, 6)])
    return tr, ytr, out["test_batch"][b"data"].reshape(-1, 3, 32, 32), np.asarray(out["test_batch"][b"labels"])


class ShapesAndCounts(unittest.TestCase):
    def test_shapes_dtypes_vs_task_cfg(self):
        for task in TASKS:
            cfg = prereg.task_cfg(task)
            for split in SPLITS:
                X, y = get(task, split)
                with self.subTest(task=task, split=split):
                    self.assertEqual(tuple(X.shape[1:]), (cfg["n_tokens"], cfg["token_dim"]))
                    self.assertEqual(X.dtype, torch.float32)
                    self.assertEqual(y.dtype, torch.int64)
                    self.assertEqual(tuple(y.shape), (X.shape[0],))
                    self.assertGreater(X.shape[0], 0)
                    self.assertTrue(bool(torch.isfinite(X).all()))
                    self.assertGreaterEqual(int(y.min()), 0)
                    self.assertLess(int(y.max()), cfg["classes"])

    def test_class_counts_equal_manifest(self):
        for task in TASKS:
            cfg = prereg.task_cfg(task)
            for split in SPLITS:
                _, y = get(task, split)
                got = np.bincount(y.numpy(), minlength=cfg["classes"]).tolist()
                with self.subTest(task=task, split=split):
                    self.assertEqual(got, MAN["tasks"][task]["splits"][split]["class_counts"])
                    self.assertEqual(len(y), MAN["tasks"][task]["splits"][split]["n"])
                    self.assertTrue(all(c > 0 for c in got), got)


class Disjointness(unittest.TestCase):
    def test_har_volunteers(self):
        vols = {s: set(meta("HAR", s)["volunteer"]) for s in SPLITS}
        ids = MAN["tasks"]["HAR"]["ids"]
        for s in SPLITS:
            self.assertEqual(sorted(vols[s]), ids[s])
            self.assertEqual(len(meta("HAR", s)["volunteer"]), len(get("HAR", s)[1]))
        self.assertFalse(vols["train"] & vols["val"])
        self.assertFalse(vols["train"] & vols["test"])
        self.assertFalse(vols["val"] & vols["test"])
        official_train = sorted(set(har_raw()["train"][1].tolist()))
        self.assertEqual(sorted(vols["train"] | vols["val"]), official_train)
        self.assertEqual(sorted(vols["val"]), official_train[-HAR_N_VAL_VOLUNTEERS:])
        print(f"\n  HAR volunteers: train {len(vols['train'])} {sorted(vols['train'])}; val {sorted(vols['val'])}; "
              f"test {len(vols['test'])} {sorted(vols['test'])}", file=sys.stderr)

    def test_kws_speakers_and_list_rule(self):
        clips = {s: meta("KWS", s)["clips"] for s in SPLITS}
        spk = {s: {c["speaker"] for c in clips[s] if c["speaker"] is not None} for s in SPLITS}
        ids = MAN["tasks"]["KWS"]["ids"]
        for s in SPLITS:
            self.assertEqual(sorted(spk[s]), ids[s])
            self.assertEqual(len(clips[s]), len(get("KWS", s)[1]))
            labels = get("KWS", s)[1].tolist()
            self.assertEqual([KWS_CLASSES.index(c["word"]) if c["word"] in KWS_CLASSES else 10 for c in clips[s]], labels)
        # every non-silence train/val clip carries a speaker
        for s in ("train", "val"):
            self.assertTrue(all(c["speaker"] is not None for c in clips[s] if c["word"] != "_silence_"))
        self.assertFalse(spk["train"] & spk["val"])
        self.assertFalse(spk["test"] & spk["train"])
        self.assertFalse(spk["test"] & spk["val"])
        n_unrec = sum(1 for c in clips["test"] if c["speaker"] is None)
        n_unrec_ns = sum(1 for c in clips["test"] if c["speaker"] is None and c["word"] != "_silence_")
        # official list rule
        d = ROOT / "data" / "KWS"
        vl = set((d / "validation_list.txt").read_text().split())
        tl = set((d / "testing_list.txt").read_text().split())
        tr_src = {c["source"] for c in clips["train"] if c["word"] != "_silence_"}
        va_src = {c["source"] for c in clips["val"] if c["word"] != "_silence_"}
        self.assertFalse(tr_src & (vl | tl))
        self.assertTrue(va_src <= vl)
        self.assertFalse(tr_src & va_src)
        print(f"\n  KWS speakers: train {len(spk['train'])}, val {len(spk['val'])}, test recoverable {len(spk['test'])}; "
              f"test clips without a recoverable speaker: {n_unrec} (non-silence {n_unrec_ns})", file=sys.stderr)

    def test_kws_unknown_silence_fractions(self):
        for s in ("train", "val"):
            y = get("KWS", s)[1].numpy()
            n = len(y)
            for k in (10, 11):
                with self.subTest(split=s, cls=KWS_CLASSES[k]):
                    self.assertLessEqual(abs(int((y == k).sum()) - 0.1 * n), FRACTION_TOL_CLIPS)

    def test_cifar_sizes_and_indices(self):
        idx = {s: meta("CIFAR10", s)["source_index"] for s in SPLITS}
        for s in SPLITS:
            self.assertEqual(len(get("CIFAR10", s)[1]), EXPECTED_SIZES_CIFAR[s])
            self.assertEqual(len(idx[s]), EXPECTED_SIZES_CIFAR[s])
            self.assertEqual(len(set(idx[s])), len(idx[s]))
        self.assertFalse(set(idx["train"]) & set(idx["val"]))
        self.assertEqual(idx["train"], list(range(45000)))
        self.assertEqual(idx["val"], list(range(45000, 50000)))
        self.assertEqual(idx["test"], list(range(10000)))


class Normalization(unittest.TestCase):
    def test_train_split_standardized_and_val_excluded(self):
        for task in TASKS:
            norm = MAN["tasks"][task]["normalization"]
            m, s = np.array(norm["mean"]), np.array(norm["std"])
            tr = channels_last(task, get(task, "train")[0]).numpy()
            va = channels_last(task, get(task, "val")[0]).numpy()
            with self.subTest(task=task):
                # normalized train has mean 0 and population std 1 per channel iff the stats are the train stats
                np.testing.assert_allclose(tr.mean(0), 0.0, atol=NORM_TOL)
                np.testing.assert_allclose(tr.std(0), 1.0, atol=NORM_TOL)
                # recomputed train statistics in raw units equal the manifest
                raw_tr = tr * s + m
                np.testing.assert_allclose(raw_tr.mean(0), m, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(raw_tr.std(0), s, rtol=1e-5)
                # statistics over train + val differ from the manifest (the check above can tell them apart)
                both = np.concatenate([raw_tr, va * s + m])
                self.assertGreater(np.abs(both.mean(0) - m).max() / s.max(), 10 * NORM_TOL)

    def test_har_stats_and_tokens_from_raw(self):
        raw = har_raw()
        sig, subj, lab = raw["train"]
        vals = sorted(set(subj.tolist()))[-HAR_N_VAL_VOLUNTEERS:]
        trm = ~np.isin(subj, vals)
        m, s = sig[trm].mean((0, 1)), sig[trm].std((0, 1))
        norm = MAN["tasks"]["HAR"]["normalization"]
        np.testing.assert_allclose(norm["mean"], m, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(norm["std"], s, rtol=1e-12)
        for split, (x, yy) in {"train": (sig[trm], lab[trm]), "val": (sig[~trm], lab[~trm]),
                               "test": (raw["test"][0], raw["test"][2])}.items():
            X, y = get("HAR", split)
            exp = ((x - m) / s).reshape(len(x), 16, 72)
            np.testing.assert_allclose(X.numpy(), exp, atol=1e-5, rtol=0)
            self.assertEqual(y.tolist(), yy.tolist())
        # token layout: token t, value j*9 + c = step 8t + j, channel c
        X = get("HAR", "test")[0].numpy()
        x = (raw["test"][0] - m) / s
        self.assertAlmostEqual(float(X[5, 3, 2 * 9 + 4]), float(x[5, 8 * 3 + 2, 4]), places=5)

    def test_cifar_stats_and_tokens_from_raw(self):
        tr, ytr, te, yte = cifar_raw()
        x45 = tr[:45000].astype(np.float64) / 255.0
        m, s = x45.mean((0, 2, 3)), x45.std((0, 2, 3))
        del x45
        norm = MAN["tasks"]["CIFAR10"]["normalization"]
        np.testing.assert_allclose(norm["mean"], m, rtol=1e-12)
        np.testing.assert_allclose(norm["std"], s, rtol=1e-12)
        for split, (imgs, labels) in {"train": (tr[:45000], ytr[:45000]), "val": (tr[45000:], ytr[45000:]),
                                      "test": (te, yte)}.items():
            X, y = get("CIFAR10", split)
            self.assertEqual(y.tolist(), labels.tolist())
            # spot-check every 97th image against an independent per-patch construction
            for i in range(0, len(imgs), 97):
                img = (imgs[i].astype(np.float64) / 255.0 - m[:, None, None]) / s[:, None, None]  # C, H, W
                for t in (0, 9, 63):
                    py, px = divmod(t, 8)
                    patch = img[:, 4 * py:4 * py + 4, 4 * px:4 * px + 4].transpose(1, 2, 0).reshape(48)  # dy, dx, c
                    np.testing.assert_allclose(X[i, t].numpy(), patch, atol=1e-5, rtol=0)

    def test_kws_test_clip_features_from_raw(self):
        clips = meta("KWS", "test")["clips"]
        X, _ = get("KWS", "test")
        norm = MAN["tasks"]["KWS"]["normalization"]
        m, s = np.array(norm["mean"]), np.array(norm["std"])
        want = {c["source"]: i for i, c in enumerate(clips) if i % 97 == 0}
        tf_mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=10, melkwargs=dict(
            n_fft=640, win_length=640, hop_length=320, n_mels=40, center=False))
        seen = 0
        with tarfile.open(RAW / "speech_commands_test_set_v0.02.tar.gz", "r|gz") as tf:
            for mem in tf:
                name = mem.name[2:] if mem.name.startswith("./") else mem.name
                if mem.isfile() and name in want:
                    sr, a = wavfile.read(io.BytesIO(tf.extractfile(mem).read()))
                    a = a.astype(np.float32) / 32768.0
                    a = np.pad(a, (0, max(0, 16000 - len(a))))[:16000]
                    f = tf_mfcc(torch.from_numpy(a)).T.double().numpy()
                    np.testing.assert_allclose(X[want[name]].double().numpy(), (f - m) / s, atol=1e-4, rtol=0)
                    seen += 1
        self.assertEqual(seen, len(want))
        self.assertGreater(seen, 40)


class Gating(unittest.TestCase):
    def test_test_split_raises_before_any_file_is_opened(self):
        for task in TASKS:
            with self.subTest(task=task):
                with self.assertRaises(data.TestSplitAccess) as cm:
                    data.load_task(task, "test")
                self.assertIsInstance(cm.exception, PermissionError)
                with mock.patch("builtins.open", side_effect=AssertionError("file opened")), \
                        mock.patch("io.open", side_effect=AssertionError("file opened (io.open / pathlib)")), \
                        mock.patch("torch.load", side_effect=AssertionError("torch.load called")):
                    with self.assertRaises(data.TestSplitAccess):
                        data.load_task(task, "test")
                    with self.assertRaises(data.TestSplitAccess):
                        data.load_task(task, "test", allow_test=1)
                    with self.assertRaises(data.TestSplitAccess):
                        data.load_meta(task, "test")
                with self.assertRaises(data.TestSplitAccess):
                    data.load_task(task, "test", data_root="/nonexistent", manifest_path="/nonexistent/m.json")
                X, y = data.load_task(task, "test", allow_test=True)
                self.assertEqual(len(X), MAN["tasks"][task]["splits"]["test"]["n"])

    def test_planted_corruption_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            (tmp / "HAR").mkdir()
            shutil.copyfile(ROOT / "data" / "HAR" / "val.pt", tmp / "HAR" / "val.pt")
            shutil.copyfile(data.MANIFEST, tmp / "m.json")
            X0, y0 = data.load_task("HAR", "val", data_root=tmp, manifest_path=tmp / "m.json")  # intact copy loads
            self.assertTrue(torch.equal(X0, get("HAR", "val")[0]))
            # re-serialize unchanged and register that file's hash, so that only the value change below differs
            d = torch.load(tmp / "HAR" / "val.pt", weights_only=True)
            torch.save(d, tmp / "HAR" / "val.pt")
            man = json.loads((tmp / "m.json").read_text())
            man["tasks"]["HAR"]["outputs"]["val.pt"]["sha256"] = data._sha256(tmp / "HAR" / "val.pt")
            (tmp / "m.json").write_text(json.dumps(man))
            X1, _ = data.load_task("HAR", "val", data_root=tmp, manifest_path=tmp / "m.json")
            self.assertTrue(torch.equal(X1, X0))
            d["X"][7, 3, 11] += 1.0
            torch.save(d, tmp / "HAR" / "val.pt")
            with self.assertRaises(data.DataIntegrityError):
                data.load_task("HAR", "val", data_root=tmp, manifest_path=tmp / "m.json")


if __name__ == "__main__":
    unittest.main()
