"""S3 data pipelines for the three frozen tasks (the preregistration in force via lb.prereg, section tasks;
identical in v1 and v2).

Downloads the four raw archives into data/raw/ (reused only after their sha256 matches the sidecar written at download
time and the previous manifest), builds data/<TASK>/{train,val,test}.pt (dict X float32 [N, n_tokens, token_dim],
y int64 [N]) plus per-sample ID files data/<TASK>/<split>_meta.json, and writes research/s3/data_manifest.json.
The Speech Commands training archive is streamed with tarfile and never extracted to disk. CPU only, 4 threads.
Run: CUDA_VISIBLE_DEVICES="" python src/lb/prepare_data.py [--tasks HAR KWS CIFAR10] [--download-only]
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse
import datetime
import hashlib
import io
import json
import pathlib
import pickle
import platform
import re
import sys
import tarfile
import urllib.request
import warnings
import zipfile

import numpy as np
import scipy
import torch
import torchaudio
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lb import prereg  # noqa: E402

torch.set_num_threads(4)

RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "research" / "s3" / "data_manifest.json"
UA = "loopbudget-S3/1.0"
SPLITS = ("train", "val", "test")

SOURCES = {
    "HAR": {"name": "human+activity+recognition+using+smartphones.zip",
            "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
            "bytes": None},
    "KWS_train": {"name": "speech_commands_v0.02.tar.gz",
                  "url": "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", "bytes": 2428923189},
    "KWS_test": {"name": "speech_commands_test_set_v0.02.tar.gz",
                 "url": "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz", "bytes": 112563277},
    "CIFAR10": {"name": "cifar-10-python.tar.gz", "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                "bytes": 170498071},
}

HAR_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z",
                "total_acc_x", "total_acc_y", "total_acc_z"]
HAR_CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
KWS_KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
KWS_CLASSES = KWS_KEYWORDS + ["_unknown_", "_silence_"]
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
SR = 16000


class RawMismatch(RuntimeError):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    print(f"[{utc()}] {msg}", flush=True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


# ----------------------------------------------------------------------------------------------- raw downloads
def fetch(key, prev_manifest):
    src = SOURCES[key]
    path = RAW / src["name"]
    side = RAW / (src["name"] + ".json")
    prev = prev_manifest.get("raw", {}).get(src["name"])
    if path.exists():
        if not side.exists():
            raise RawMismatch(f"{path} exists without its download sidecar {side.name}; delete it or restore the sidecar")
        rec = json.loads(side.read_text())
        got = sha256_file(path)
        size = path.stat().st_size
        if got != rec["sha256"] or size != rec["bytes"]:
            raise RawMismatch(f"{path.name}: sha256 {got} / {size} B != sidecar {rec['sha256']} / {rec['bytes']} B")
        if prev is not None and prev["sha256"] != got:
            raise RawMismatch(f"{path.name}: sha256 {got} != previous manifest {prev['sha256']}")
        log(f"reuse {path.name}: {size} B sha256 {got} (matches sidecar"
            + (" and previous manifest)" if prev else "; no previous manifest entry)"))
        return path, rec
    RAW.mkdir(parents=True, exist_ok=True)
    part = RAW / (src["name"] + ".part")
    t0 = utc()
    log(f"download {src['url']}")
    req = urllib.request.Request(src["url"], headers={"User-Agent": UA})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        final_url = r.geturl()
        clen = r.headers.get("Content-Length")
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            h.update(b)
            n += len(b)
    if clen is not None and int(clen) != n:
        raise RawMismatch(f"{src['name']}: got {n} B, Content-Length {clen}")
    if src["bytes"] is not None and src["bytes"] != n:
        raise RawMismatch(f"{src['name']}: got {n} B, expected {src['bytes']}")
    part.rename(path)
    rec = {"url": src["url"], "final_url": final_url, "bytes": n, "sha256": h.hexdigest(),
           "retrieval_started_utc": t0, "retrieved_utc": utc(), "user_agent": UA}
    side.write_text(json.dumps(rec, indent=1) + "\n")
    log(f"downloaded {src['name']}: {n} B sha256 {rec['sha256']} final_url {final_url}")
    return path, rec


# ----------------------------------------------------------------------------------------------- writers
def save_split(task, split, X, y, meta, cfg):
    X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.int64)
    assert X.ndim == 3 and X.shape[1:] == (cfg["n_tokens"], cfg["token_dim"]), (task, split, tuple(X.shape))
    assert y.shape == (X.shape[0],) and int(y.min()) >= 0 and int(y.max()) < cfg["classes"]
    assert bool(torch.isfinite(X).all())
    d = ROOT / "data" / task
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    p = d / f"{split}.pt"
    tmp = d / f"{split}.pt.tmp"
    torch.save({"X": X, "y": y}, tmp)
    tmp.rename(p)
    out[f"{split}.pt"] = {"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256_file(p),
                          "X_shape": list(X.shape), "X_dtype": str(X.dtype).replace("torch.", ""),
                          "y_shape": list(y.shape), "y_dtype": str(y.dtype).replace("torch.", "")}
    m = d / f"{split}_meta.json"
    m.write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    out[f"{split}_meta.json"] = {"path": str(m.relative_to(ROOT)), "bytes": m.stat().st_size, "sha256": sha256_file(m)}
    counts = np.bincount(y.numpy(), minlength=cfg["classes"]).tolist()
    log(f"{task}/{split}: X {tuple(X.shape)} {X.dtype}, y {tuple(y.shape)}; class counts {counts}; "
        f"sha256 {out[f'{split}.pt']['sha256']}")
    return out, {"n": int(X.shape[0]), "class_counts": counts}


# ----------------------------------------------------------------------------------------------- HAR
def build_har(raw_path, cfg):
    z = zipfile.ZipFile(raw_path)
    names = [n for n in z.namelist() if "__MACOSX" not in n]
    nested = None
    if not any("Inertial Signals/" in n for n in names):
        cands = [n for n in names if n.endswith("UCI HAR Dataset.zip")]
        assert len(cands) == 1, cands
        nested = cands[0]
        z = zipfile.ZipFile(io.BytesIO(z.read(nested)))
        names = [n for n in z.namelist() if "__MACOSX" not in n]

    def read(suffix):
        hit = [n for n in names if n.endswith(suffix)]
        assert len(hit) == 1, (suffix, hit)
        return np.loadtxt(io.BytesIO(z.read(hit[0])))

    parts = {}
    for part in ("train", "test"):
        sig = np.stack([read(f"{part}/Inertial Signals/{c}_{part}.txt") for c in HAR_CHANNELS], axis=-1)
        subj = read(f"{part}/subject_{part}.txt").astype(np.int64)
        lab = read(f"{part}/y_{part}.txt").astype(np.int64) - 1
        assert sig.shape == (len(subj), 128, 9) and lab.shape == subj.shape and set(lab.tolist()) == set(range(6))
        parts[part] = (sig, subj, lab)
        log(f"HAR official {part}: {sig.shape}, volunteers {sorted(set(subj.tolist()))}")
    sig_tr, subj_tr, lab_tr = parts["train"]
    train_vols = sorted(set(subj_tr.tolist()))
    val_vols = train_vols[-4:]
    fit_vols = train_vols[:-4]
    is_val = np.isin(subj_tr, val_vols)
    rows = np.arange(len(subj_tr))
    split_data = {"train": (sig_tr[~is_val], subj_tr[~is_val], lab_tr[~is_val], rows[~is_val], "train"),
                  "val": (sig_tr[is_val], subj_tr[is_val], lab_tr[is_val], rows[is_val], "train"),
                  "test": (parts["test"][0], parts["test"][1], parts["test"][2], np.arange(len(parts["test"][1])), "test")}
    mean = split_data["train"][0].mean(axis=(0, 1))
    std = split_data["train"][0].std(axis=(0, 1))
    outputs, stats = {}, {}
    for split in SPLITS:
        sig, subj, lab, src_rows, src_part = split_data[split]
        X = ((sig - mean) / std).reshape(len(sig), 16, 8 * 9)
        meta = {"task": "HAR", "split": split, "volunteer": subj.tolist(),
                "source": f"official {src_part} rows (0-based)", "source_row": src_rows.tolist()}
        o, s = save_split("HAR", split, X, lab, meta, cfg)
        outputs.update(o)
        s["volunteers"] = sorted(set(subj.tolist()))
        stats[split] = s
    return {
        "processing": {
            "raw_file": raw_path.name, "nested_zip": nested,
            "channels_in_order": HAR_CHANNELS, "steps": 128,
            "split_rule": "train/val from official train by volunteer: val = the 4 highest-numbered training volunteers; test = official test",
            "labels": "y_*.txt minus 1", "class_names": HAR_CLASSES,
            "tokens": "(N,128,9) -> (N,16,72): 8 consecutive steps per token, step-major (index = step*9 + channel)",
            "normalization": "per-channel z-score, mean and population std (ddof 0) over all train samples and 128 steps; float64 then cast to float32"},
        "normalization": {"over": "train split only", "ddof": 0, "axis": "channel", "mean": mean.tolist(), "std": std.tolist()},
        "splits": stats,
        "ids": {"kind": "volunteer", "training_volunteers_official": train_vols, "train": fit_vols, "val": val_vols,
                "test": sorted(set(parts["test"][1].tolist()))},
        "outputs": outputs,
    }


# ----------------------------------------------------------------------------------------------- KWS
_MFCC = None


def mfcc(wave_f32):
    global _MFCC
    if _MFCC is None:
        _MFCC = torchaudio.transforms.MFCC(sample_rate=SR, n_mfcc=10, melkwargs=dict(
            n_fft=640, win_length=640, hop_length=320, n_mels=40, center=False))
    with torch.no_grad():
        f = _MFCC(torch.from_numpy(np.ascontiguousarray(wave_f32, dtype=np.float32)))
    assert f.shape == (10, 49), f.shape
    return f.T.numpy().astype(np.float32)  # (49, 10)


def decode_wav(data, name, stats):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sr, a = wavfile.read(io.BytesIO(data))
    if w:
        stats["wav_warnings"] = stats.get("wav_warnings", 0) + 1
        stats.setdefault("wav_warning_files", []).append(f"{name}: {w[0].message}")
    if sr != SR or a.dtype != np.int16 or a.ndim != 1:
        raise RuntimeError(f"{name}: sr {sr}, dtype {a.dtype}, shape {a.shape} is not 16 kHz mono int16")
    return a.astype(np.float32) / 32768.0


def fix_len(a, stats):
    if len(a) < SR:
        stats["padded"] = stats.get("padded", 0) + 1
        return np.pad(a, (0, SR - len(a)))
    if len(a) > SR:
        stats["truncated"] = stats.get("truncated", 0) + 1
        return a[:SR]
    return a


def _norm(name):
    return name[2:] if name.startswith("./") else name


def stream_wavs(path):
    with tarfile.open(path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            yield _norm(m.name), tf.extractfile(m).read()


def speaker_of(fname):
    """Speaker = the filename part before "_nohash_"; test-archive _unknown_ names carry a "<word>_" prefix, which is
    dropped. Only 8-hex-digit IDs are accepted; anything else (e.g. test _silence_ clips "4664.wav") is None."""
    if "_nohash_" not in fname:
        return None
    s = fname.split("_nohash_")[0].split("_")[-1]
    return s if re.fullmatch(r"[0-9a-f]{8}", s) else None


def build_kws(train_path, test_path, cfg):
    feats, noise, lists, other_files = {}, {}, {}, []
    st_train = {}
    n = 0
    log("KWS: streaming the training archive (no extraction to disk)")
    for name, data in stream_wavs(train_path):
        parts = name.split("/")
        if name in ("validation_list.txt", "testing_list.txt"):
            lists[name] = [ln.strip() for ln in data.decode().splitlines() if ln.strip()]
        elif len(parts) == 2 and parts[0] == "_background_noise_" and name.endswith(".wav"):
            noise[parts[1]] = decode_wav(data, name, st_train)
        elif len(parts) == 2 and name.endswith(".wav"):
            feats[name] = mfcc(fix_len(decode_wav(data, name, st_train), st_train))
            n += 1
            if n % 10000 == 0:
                log(f"KWS: {n} training-archive clips featurized")
        else:
            other_files.append(name)
    log(f"KWS: training archive done: {len(feats)} clips, {len(noise)} noise files, lists {sorted(lists)}, "
        f"other files {other_files}, length fixes {st_train}")
    assert set(lists) == {"validation_list.txt", "testing_list.txt"}, sorted(lists)
    val_list, test_list = set(lists["validation_list.txt"]), set(lists["testing_list.txt"])
    missing = {k: sum(1 for p in v if p not in feats) for k, v in (("validation_list.txt", val_list), ("testing_list.txt", test_list))}
    folders = sorted({p.split("/")[0] for p in feats})
    assert all(k in folders for k in KWS_KEYWORDS), folders
    unknown_words = [f for f in folders if f not in KWS_KEYWORDS]

    rng_unknown = np.random.default_rng(0)
    rng_silence = np.random.default_rng(1)
    noise_names = sorted(noise)
    split_rows = {}
    split_counts_info = {}
    for split in ("train", "val"):
        if split == "train":
            members = [p for p in sorted(feats) if p not in val_list and p not in test_list]
        else:
            members = [p for p in sorted(feats) if p in val_list]
        kw = [p for p in members if p.split("/")[0] in KWS_KEYWORDS]
        pool = [p for p in members if p.split("/")[0] not in KWS_KEYWORDS]
        n_extra = int(round(len(kw) / 8))
        pick = np.sort(rng_unknown.choice(len(pool), size=n_extra, replace=False))
        unk = [pool[i] for i in pick]
        rows = []
        for p in sorted(kw + unk):
            w = p.split("/")[0]
            lab = KWS_CLASSES.index(w) if w in KWS_KEYWORDS else KWS_CLASSES.index("_unknown_")
            rows.append((feats[p], lab, {"source": p, "word": w, "speaker": speaker_of(p.split("/")[1])}))
        for _ in range(n_extra):
            fi = int(rng_silence.integers(len(noise_names)))
            a = noise[noise_names[fi]]
            off = int(rng_silence.integers(0, len(a) - SR + 1))
            scale = float(rng_silence.uniform(0.0, 1.0))
            clip = a[off:off + SR] * np.float32(scale)
            rows.append((mfcc(clip), KWS_CLASSES.index("_silence_"),
                         {"source": f"_background_noise_/{noise_names[fi]}", "word": "_silence_", "speaker": None,
                          "offset": off, "scale": scale}))
        split_rows[split] = rows
        split_counts_info[split] = {"n_keywords": len(kw), "n_unknown_pool": len(pool), "n_unknown": n_extra,
                                    "n_silence": n_extra}
        log(f"KWS {split}: keywords {len(kw)}, unknown {n_extra} of pool {len(pool)}, silence {n_extra}")

    log("KWS: streaming the test archive")
    st_test = {}
    test_rows, test_other = [], []
    for name, data in stream_wavs(test_path):
        parts = name.split("/")
        if len(parts) == 2 and name.endswith(".wav"):
            if parts[0] not in KWS_CLASSES:
                raise RuntimeError(f"test archive folder {parts[0]} is not one of the 12 classes")
            test_rows.append((mfcc(fix_len(decode_wav(data, name, st_test), st_test)), KWS_CLASSES.index(parts[0]),
                              {"source": name, "word": parts[0], "speaker": speaker_of(parts[1])}))
        else:
            test_other.append(name)
    test_rows.sort(key=lambda r: r[2]["source"])
    split_rows["test"] = test_rows
    log(f"KWS test: {len(test_rows)} clips; other entries {test_other}; length fixes {st_test}")

    Xtr = np.stack([r[0] for r in split_rows["train"]]).astype(np.float64)
    mean = Xtr.mean(axis=(0, 1))
    std = Xtr.std(axis=(0, 1))
    outputs, stats, ids = {}, {}, {"kind": "speaker = filename part before _nohash_ (word prefix of test _unknown_ names dropped; 8 hex digits required; silence clips have none)"}
    for split in SPLITS:
        rows = split_rows[split]
        X = (np.stack([r[0] for r in rows]).astype(np.float64) - mean) / std
        y = np.array([r[1] for r in rows], dtype=np.int64)
        meta = {"task": "KWS", "split": split, "clips": [r[2] for r in rows]}
        o, s = save_split("KWS", split, X, y, meta, cfg)
        outputs.update(o)
        spk = [r[2]["speaker"] for r in rows]
        s["n_speaker_recoverable"] = sum(1 for v in spk if v is not None)
        s["n_speaker_not_recoverable"] = sum(1 for v in spk if v is None)
        s["n_speaker_not_recoverable_non_silence"] = sum(1 for r in rows if r[2]["speaker"] is None and r[1] != KWS_CLASSES.index("_silence_"))
        if split in split_counts_info:
            s.update(split_counts_info[split])
        stats[split] = s
        ids[split] = sorted({v for v in spk if v is not None})
    d = ROOT / "data" / "KWS"
    for k in ("validation_list.txt", "testing_list.txt"):
        (d / k).write_text("\n".join(lists[k]) + "\n")
        outputs[k] = {"path": str((d / k).relative_to(ROOT)), "bytes": (d / k).stat().st_size, "sha256": sha256_file(d / k),
                      "note": "copy of the official list from the training archive (sorted as in the archive)"}
    return {
        "processing": {
            "raw_files": [train_path.name, test_path.name], "class_names": KWS_CLASSES,
            "split_rule": "train = training-archive clips in neither validation_list.txt nor testing_list.txt; val = clips in validation_list.txt; testing_list.txt clips unused; test = all clips of the separate 12-class test archive",
            "unknown": "all non-keyword word folders except _background_noise_; per split sampled without replacement from the path-sorted pool with one numpy default_rng(0) used for train first, then val; n_unknown = round(n_keywords / 8) (Python round, half to even)",
            "silence": "n_silence = n_unknown; one numpy default_rng(1) used for train first, then val; per clip: file = integers(n_files) over the name-sorted _background_noise_ wavs, offset = integers(0, len - 16000 + 1), scale = uniform(0, 1); clip = crop * scale",
            "order_within_split": "keyword and unknown clips sorted by archive path, then silence clips in generation order (train/val); test sorted by archive path",
            "audio": "scipy.io.wavfile int16 16 kHz mono; float32 = int16 / 32768; zero-pad or truncate to 16000 samples",
            "features": "torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=10, melkwargs=dict(n_fft=640, win_length=640, hop_length=320, n_mels=40, center=False)) per clip on a 1-D waveform (10, 49) -> transposed (49, 10)",
            "normalization": "per-coefficient z-score, mean and population std (ddof 0) over all train clips and 49 frames; float64 then cast to float32",
            "training_archive_entries_not_used": other_files, "noise_files": noise_names,
            "noise_lengths_samples": {k: int(len(noise[k])) for k in noise_names},
            "list_entries_missing_from_archive": missing,
            "length_fixes_training_archive": st_train, "length_fixes_test_archive": st_test,
            "test_archive_entries_not_used": test_other,
            "unknown_words": unknown_words},
        "normalization": {"over": "train split only", "ddof": 0, "axis": "MFCC coefficient", "mean": mean.tolist(), "std": std.tolist()},
        "splits": stats,
        "ids": ids,
        "outputs": outputs,
    }


# ----------------------------------------------------------------------------------------------- CIFAR-10
def build_cifar(raw_path, cfg):
    batches = {}
    with tarfile.open(raw_path, "r:gz") as tf:
        for m in tf.getmembers():
            base = m.name.split("/")[-1]
            if base in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
                batches[base] = pickle.load(tf.extractfile(m), encoding="bytes")
    tr = [batches[f"data_batch_{i}"] for i in range(1, 6)]
    Xall = np.concatenate([b[b"data"] for b in tr]).reshape(-1, 3, 32, 32)
    yall = np.concatenate([np.asarray(b[b"labels"]) for b in tr]).astype(np.int64)
    Xte = batches["test_batch"][b"data"].reshape(-1, 3, 32, 32)
    yte = np.asarray(batches["test_batch"][b"labels"], dtype=np.int64)
    assert Xall.shape == (50000, 3, 32, 32) and Xte.shape == (10000, 3, 32, 32)
    idx = {"train": np.arange(0, 45000), "val": np.arange(45000, 50000), "test": np.arange(10000)}
    src = {"train": (Xall, yall), "val": (Xall, yall), "test": (Xte, yte)}
    xtr = Xall[:45000].astype(np.float64) / 255.0
    mean = xtr.mean(axis=(0, 2, 3))
    std = xtr.std(axis=(0, 2, 3))
    del xtr
    outputs, stats = {}, {}
    for split in SPLITS:
        Xs, ys = src[split]
        x = Xs[idx[split]].astype(np.float64) / 255.0
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        x = x.transpose(0, 2, 3, 1)  # N, H, W, C
        x = x.reshape(-1, 8, 4, 8, 4, 3).transpose(0, 1, 3, 2, 4, 5).reshape(-1, 64, 48)
        meta = {"task": "CIFAR10", "split": split, "source": "data_batch_1..5 concatenated in file order" if split != "test" else "test_batch",
                "source_index": idx[split].tolist()}
        o, s = save_split("CIFAR10", split, x.astype(np.float32), ys[idx[split]], meta, cfg)
        outputs.update(o)
        stats[split] = s
        del x
    return {
        "processing": {
            "raw_file": raw_path.name, "class_names": CIFAR_CLASSES,
            "split_rule": "train = images 0..44999 of data_batch_1..5 concatenated in file order; val = images 45000..49999; test = test_batch",
            "normalization": "pixels / 255, then per-channel z-score with mean and population std (ddof 0) over the 45000 training images (equivalent to z-scoring raw 0..255 values); float64 then cast to float32",
            "tokens": "4 x 4 patches in row-major patch order (64 tokens); token values (dy, dx, channel) flattened in that order (48)"},
        "normalization": {"over": "train split only (45000 images)", "ddof": 0, "axis": "RGB channel", "units": "pixel / 255",
                          "mean": mean.tolist(), "std": std.tolist()},
        "splits": stats,
        "ids": {"kind": "index into data_batch_1..5 concatenated (train, val) or test_batch (test); ranges are [start, stop)",
                "train": [0, 45000], "val": [45000, 50000], "test": [0, 10000], "test_source": "test_batch"},
        "outputs": outputs,
    }


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["HAR", "KWS", "CIFAR10"], choices=["HAR", "KWS", "CIFAR10"])
    ap.add_argument("--download-only", action="store_true")
    a = ap.parse_args()
    log(f"start; tasks {a.tasks}; torch threads {torch.get_num_threads()}; CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'")
    p = prereg.load()
    prev = load_manifest()
    keys = {"HAR": ["HAR"], "KWS": ["KWS_train", "KWS_test"], "CIFAR10": ["CIFAR10"]}
    raws = {}
    for t in a.tasks:
        for k in keys[t]:
            raws[k] = fetch(k, prev)
    if a.download_only:
        log("download-only: done")
        return 0
    man = prev if prev else {}
    man.update({"schema": "loopbudget S3 data manifest v1", "generated_utc": utc(),
                "script": "src/lb/prepare_data.py", "script_sha256": sha256_file(pathlib.Path(__file__)),
                "prereg_yaml": prereg.YAML_PATH.name, "prereg_yaml_sha256": prereg.yaml_sha256(),
                "env": {"python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                        "numpy": np.__version__, "scipy": scipy.__version__, "torch_threads": torch.get_num_threads(),
                        "device": "cpu"}})
    man.setdefault("raw", {})
    man.setdefault("tasks", {})
    for k, (path, rec) in raws.items():
        man["raw"][path.name] = rec
    builders = {"HAR": lambda: build_har(raws["HAR"][0], prereg.task_cfg("HAR", p)),
                "KWS": lambda: build_kws(raws["KWS_train"][0], raws["KWS_test"][0], prereg.task_cfg("KWS", p)),
                "CIFAR10": lambda: build_cifar(raws["CIFAR10"][0], prereg.task_cfg("CIFAR10", p))}
    for t in a.tasks:
        log(f"build {t}")
        res = builders[t]()
        res["task_cfg"] = prereg.task_cfg(t, p)
        old = prev.get("tasks", {}).get(t, {}).get("outputs", {})
        for f, rec in res["outputs"].items():
            if f in old:
                log(f"{t}/{f}: sha256 {'unchanged' if old[f]['sha256'] == rec['sha256'] else 'CHANGED'} vs previous manifest")
        man["tasks"][t] = res
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1) + "\n")
    tmp.rename(MANIFEST)
    log(f"manifest written: {MANIFEST.relative_to(ROOT)} sha256 {sha256_file(MANIFEST)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
