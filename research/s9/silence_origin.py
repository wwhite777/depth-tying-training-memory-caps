"""Origin of the official Speech Commands v0.02 test-set _silence_ clips (S9, 2026-10-15; asked by an external review).

Stage "fetch": streams the two public archives once from their original URLs (plain User-Agent, no e-mail), verifies
byte count and sha256 against research/s3/data_manifest.json (fail-closed), and extracts only the six
_background_noise_/*.wav recordings (training archive) and the _silence_/*.wav clips (test archive) into --workdir.
Stage "analyse": for every test clip and every recording, the cosine similarity (normalised cross-correlation without
centring) between the clip and every window of the recording, by FFT; the best window gives recording, offset, NCC and
the least-squares gain. A clip is called a crop when NCC >= --ncc (default 0.999). It also reports whether each crop's
interval overlaps the training-split silence crops of the same recording (data/KWS/train_meta.json) or the
validation-split ones (val_meta.json). Stage "selftest": planted crops in synthetic recordings must be recovered exactly
and random clips rejected; exit 1 otherwise.

  python research/s9/silence_origin.py fetch --workdir <scratch>/kws_silence
  python research/s9/silence_origin.py analyse --workdir <scratch>/kws_silence --out research/s9/silence_origin_<UTC>.json
  python research/s9/silence_origin.py selftest
Exit 0 = done; 1 = selftest failure; 2 = refused (hash mismatch, missing input, existing output).
"""
import argparse
import bisect
import collections
import datetime
import hashlib
import io
import json
import pathlib
import sys
import tarfile
import urllib.request
import wave

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
UA = "loopbudget-silence-origin/1.0"
TRAIN, TEST = "speech_commands_v0.02.tar.gz", "speech_commands_test_set_v0.02.tar.gz"
SR = 16000


class Hashing(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.h, self.n = raw, hashlib.sha256(), 0

    def readable(self):
        return True

    def readinto(self, b):
        data = self.raw.read(len(b))
        if not data:
            return 0
        b[:len(data)] = data
        self.h.update(data)
        self.n += len(data)
        return len(data)


def fetch(workdir):
    man = json.loads((ROOT / "research/s3/data_manifest.json").read_text())["raw"]
    workdir.mkdir(parents=True, exist_ok=True)
    rec = {}
    for name, want in ((TRAIN, "_background_noise_/"), (TEST, "_silence_/")):
        m = man[name]
        req = urllib.request.Request(m["url"], headers={"User-Agent": UA})
        n_ex = 0
        with urllib.request.urlopen(req, timeout=300) as r:
            hs = Hashing(r)
            buf = io.BufferedReader(hs, buffer_size=1 << 22)
            with tarfile.open(fileobj=buf, mode="r|gz") as tf:
                for mem in tf:
                    if mem.isfile() and want in mem.name and mem.name.endswith(".wav"):
                        dst = workdir / want.strip("/") / pathlib.Path(mem.name).name
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(tf.extractfile(mem).read())
                        n_ex += 1
            while buf.read(1 << 22):
                pass
        ok = hs.n == m["bytes"] and hs.h.hexdigest() == m["sha256"]
        rec[name] = {"url": m["url"], "bytes": hs.n, "sha256": hs.h.hexdigest(), "manifest_sha256": m["sha256"],
                     "matches_manifest": ok, "extracted": n_ex, "member_filter": want}
        print(f"{name}: {hs.n} B sha256 {hs.h.hexdigest()} manifest {'MATCH' if ok else 'MISMATCH'}; extracted {n_ex}")
        if not ok:
            (workdir / "fetch_receipt.json").write_text(json.dumps(rec, indent=1))
            print("REFUSED: archive differs from the manifest", file=sys.stderr)
            return 2
    (workdir / "fetch_receipt.json").write_text(json.dumps(rec, indent=1))
    return 0


def read_wav(p):
    with wave.open(str(p)) as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != SR:
            raise ValueError(f"{p}: not 16-bit mono {SR} Hz")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)


def best_match(rec_fft, rec_cum, nrec, nfft, clip):
    """Best window of one recording for one clip: (ncc, offset, gain)."""
    L = len(clip)
    if L > nrec:
        return -1.0, -1, 0.0
    cn = float(np.sqrt(np.dot(clip, clip)))
    if cn == 0:
        return 0.0, -1, 0.0
    dots = np.fft.irfft(rec_fft * np.fft.rfft(clip[::-1], nfft), nfft)[L - 1:nrec]
    win = np.sqrt(np.maximum(rec_cum[L:nrec + 1] - rec_cum[:nrec - L + 1], 1e-300))
    ncc = dots / (cn * win)
    k = int(np.argmax(ncc))
    gain = float(dots[k] / (win[k] ** 2))
    return float(ncc[k]), k, gain


def match_all(recordings, clips, ncc_min):
    prep = {}
    for name, x in recordings.items():
        nfft = 1 << (len(x) + SR).bit_length()
        prep[name] = (np.fft.rfft(x, nfft), np.concatenate([[0.0], np.cumsum(x * x)]), len(x), nfft)
    rows = []
    for cname, c in clips.items():
        best = None
        for rname, (F, cum, n, nfft) in prep.items():
            ncc, off, gain = best_match(F, cum, n, nfft, c)
            if best is None or ncc > best[1]:
                best = (rname, ncc, off, gain)
        rows.append({"clip": cname, "recording": best[0], "ncc": best[1], "offset": best[2], "gain": best[3],
                     "length": len(c), "crop": best[1] >= ncc_min})
    return rows


def intervals(meta_path):
    by = collections.defaultdict(list)
    for c in json.loads(meta_path.read_text())["clips"]:
        if c["word"] == "_silence_":
            by[pathlib.Path(c["source"]).name].append(int(c["offset"]))
    return {k: sorted(v) for k, v in by.items()}


def overlaps(starts, off, length):
    i = bisect.bisect_left(starts, off - SR + 1)
    return i < len(starts) and starts[i] < off + length


def analyse(workdir, out, ncc_min):
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    rec_dir, clip_dir = workdir / "_background_noise_", workdir / "_silence_"
    recordings = {p.name: read_wav(p) for p in sorted(rec_dir.glob("*.wav"))}
    clips = {p.name: read_wav(p) for p in sorted(clip_dir.glob("*.wav"))}
    if len(recordings) != 6 or not clips:
        print(f"REFUSED: {len(recordings)} recordings, {len(clips)} clips", file=sys.stderr)
        return 2
    rows = match_all(recordings, clips, ncc_min)
    tr = intervals(ROOT / "data/KWS/train_meta.json")
    va = intervals(ROOT / "data/KWS/val_meta.json")
    for r in rows:
        r["overlaps_train_crop"] = r["crop"] and overlaps(tr.get(r["recording"], []), r["offset"], r["length"])
        r["overlaps_val_crop"] = r["crop"] and overlaps(va.get(r["recording"], []), r["offset"], r["length"])
    crops = [r for r in rows if r["crop"]]
    summary = {
        "n_clips": len(rows), "n_crops": len(crops), "ncc_threshold": ncc_min,
        "ncc_min_over_crops": min((r["ncc"] for r in crops), default=None),
        "ncc_max_over_non_crops": max((r["ncc"] for r in rows if not r["crop"]), default=None),
        "crops_by_recording": dict(collections.Counter(r["recording"] for r in crops)),
        "gain_range_over_crops": [min((r["gain"] for r in crops), default=None), max((r["gain"] for r in crops), default=None)],
        "clip_lengths": dict(collections.Counter(r["length"] for r in rows)),
        "crops_overlapping_train_crops": sum(r["overlaps_train_crop"] for r in rows),
        "crops_overlapping_val_crops": sum(r["overlaps_val_crop"] for r in rows),
        "recordings": {k: len(v) for k, v in recordings.items()},
    }
    rec = {"kind": "Speech Commands v0.02 test-set silence origin", "written_utc":
           datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "fetch_receipt": json.loads((workdir / "fetch_receipt.json").read_text()),
           "meta_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                           for p in ("data/KWS/train_meta.json", "data/KWS/val_meta.json")},
           "summary": summary, "clips": rows}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


def selftest():
    rng = np.random.default_rng(7)
    recs = {f"r{i}.wav": rng.normal(0, 1000, SR * (30 + 7 * i)) for i in range(6)}
    clips, truth = {}, {}
    for j in range(20):
        rn = f"r{j % 6}.wav"
        off = int(rng.integers(0, len(recs[rn]) - SR))
        g = float(rng.uniform(0.05, 1.0))
        clips[f"c{j}.wav"] = np.round(recs[rn][off:off + SR] * g)
        truth[f"c{j}.wav"] = (rn, off, g)
    for j in range(5):
        clips[f"rand{j}.wav"] = rng.normal(0, 500, SR)
    rows = {r["clip"]: r for r in match_all(recs, clips, 0.999)}
    ok = all(rows[c]["crop"] and rows[c]["recording"] == t[0] and rows[c]["offset"] == t[1]
             and abs(rows[c]["gain"] - t[2]) < 1e-3 for c, t in truth.items())
    ok &= not any(rows[f"rand{j}.wav"]["crop"] for j in range(5))
    print(f"selftest: {'PASS' if ok else 'FAIL'} (20 planted crops recovered, 5 random clips rejected)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("fetch", "analyse", "selftest"))
    ap.add_argument("--workdir")
    ap.add_argument("--out")
    ap.add_argument("--ncc", type=float, default=0.999)
    a = ap.parse_args()
    if a.stage == "selftest":
        return selftest()
    wd = pathlib.Path(a.workdir)
    if a.stage == "fetch":
        return fetch(wd)
    out = pathlib.Path(a.out)
    return analyse(wd, out if out.is_absolute() else ROOT / out, a.ncc)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
