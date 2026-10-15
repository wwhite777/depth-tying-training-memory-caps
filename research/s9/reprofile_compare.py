"""Compare the re-profiles of the 22 finite-cap selections under the transfer environment (torch 2.10.0+cpu;
result/s9/reprofile_cpu/<name>/profile.json) with the archived registry profiles (torch 2.10.0+cu128 on CPU;
result/s4/profile_registry/<name>/profile.json). Every measurement (microbatch x checkpointing) must agree in peak_bytes,
baseline_bytes, peak_phase, closure_error, trace_len and trace_sha256 (the hash of the full signed allocation trace),
and the minimum and its knob must agree. Exit 0 = all identical; 1 = any difference; 2 = error.
  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s9/reprofile_compare.py --out research/s9/reprofile_compare_<UTC>.json \
      [--pair NEW_DIR OLD_DIR ...]   (pairs override the default 22; used for the planted-failure check)
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIELDS = ("peak_bytes", "baseline_bytes", "peak_phase", "closure_error", "trace_len", "trace_sha256")


def pairs_default():
    sel = json.loads((ROOT / "research/s4/selection.json").read_text())
    cids = sorted({c[cl]["config_id"] for c in sel["cells"] if c["cap_kib"] != "unbounded" for cl in ("tied", "untied")})
    return [(ROOT / "result/s9/reprofile_cpu" / c.replace("|", "__"), ROOT / "result/s4/profile_registry" / c.replace("|", "__"))
            for c in cids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", nargs=2, action="append")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    pairs = [(pathlib.Path(n), pathlib.Path(o)) for n, o in a.pair] if a.pair else pairs_default()
    rows, all_ok, n_meas = [], True, 0
    for new_d, old_d in pairs:
        N = json.loads((new_d / "profile.json").read_text())
        O = json.loads((old_d / "profile.json").read_text())
        key = lambda m: (m["microbatch"], m["checkpointing"])
        nm, om = {key(m): m for m in N["measurements"]}, {key(m): m for m in O["measurements"]}
        diffs = []
        if N["config_id"] != O["config_id"]:
            diffs.append("config_id")
        if set(nm) != set(om):
            diffs.append("knob set")
        for k in sorted(set(nm) & set(om)):
            n_meas += 1
            for f in FIELDS:
                if nm[k].get(f) != om[k].get(f):
                    diffs.append(f"{k} {f}")
        if (N["min_peak_bytes"], N["argmin_knob"]) != (O["min_peak_bytes"], O["argmin_knob"]):
            diffs.append("minimum or argmin")
        ok = not diffs
        all_ok &= ok
        rows.append({"config_id": O["config_id"], "identical": ok, "differences": diffs[:20],
                     "min_peak_bytes": [N["min_peak_bytes"], O["min_peak_bytes"]],
                     "argmin_knob": [N["argmin_knob"], O["argmin_knob"]], "n_measurements": len(nm),
                     "code_freeze_equal": N.get("code_freeze") == O.get("code_freeze")})
    rec = {"kind": "re-profile of the 22 finite-cap selections under the transfer environment",
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "fields": FIELDS, "n_configs": len(rows), "n_measurements": n_meas,
           "n_identical_configs": sum(r["identical"] for r in rows), "all_identical": all_ok, "configs": rows}
    out.write_text(json.dumps(rec, indent=1))
    print(f"{rec['n_identical_configs']} of {len(rows)} configurations identical over {n_meas} measurements; "
          f"{'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
