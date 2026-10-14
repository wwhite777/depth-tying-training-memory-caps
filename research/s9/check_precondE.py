"""Precondition E of research/s9/TRANSFER_PLAN_v1.md: each microbatch-64 rerun in result/s9/transfer_env/<run>/ must equal
the archived run result/s4/confirm/<run>/ bitwise in test_acc, final_val_acc, train_loss_per_epoch, val_acc_per_epoch and
steps (floats compared with ==; JSON stores repr-exact doubles).

  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s9/check_precondE.py --out research/s9/precondE_<UTC>.json \
      [--pair ENV_RUN_DIR ARCHIVED_RUN_DIR ...]
Without --pair, every run directory under result/s9/transfer_env/ is paired with the archived run of the same name.
Exit 0 = every pair identical; 1 = a pair differs or a run is not finished; 2 = error (missing file, existing output).
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIELDS = ("test_acc", "final_val_acc", "train_loss_per_epoch", "val_acc_per_epoch", "steps", "status")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ENV_RUN", "ARCHIVED_RUN"))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        print(f"ERROR: {out} exists", file=sys.stderr)
        return 2
    if a.pair:
        pairs = [(ROOT / e, ROOT / r) for e, r in a.pair]
    else:
        env = sorted(p for p in (ROOT / "result/s9/transfer_env").iterdir() if p.is_dir())
        pairs = [(p, ROOT / "result/s4/confirm" / p.name) for p in env]
    if not pairs:
        print("ERROR: no run to compare", file=sys.stderr)
        return 2
    rows, ok_all = [], True
    for e, r in pairs:
        ej, rj = e / "result.json", r / "result.json"
        if not rj.is_file():
            print(f"ERROR: archived run missing: {rj}", file=sys.stderr)
            return 2
        if not ej.is_file() or not (e / "DONE.json").is_file():
            rows.append({"env_run": str(e.relative_to(ROOT)), "archived_run": str(r.relative_to(ROOT)), "finished": False})
            ok_all = False
            continue
        E, A = json.loads(ej.read_text()), json.loads(rj.read_text())
        same = {f: E.get(f) == A.get(f) for f in FIELDS}
        knob = (E.get("microbatch"), E.get("checkpointing")) == (A.get("microbatch"), A.get("checkpointing")) == (64, False)
        cfg = (E.get("config_id"), E.get("seed"), E.get("lr"), E.get("mode")) == (A.get("config_id"), A.get("seed"), A.get("lr"), A.get("mode"))
        ok = all(same.values()) and knob and cfg
        ok_all &= ok
        rows.append({"env_run": str(e.relative_to(ROOT)), "archived_run": str(r.relative_to(ROOT)), "finished": True,
                     "identical": ok, "fields_equal": same, "knob_m64_nockpt": knob, "same_config_seed_lr_mode": cfg,
                     "test_acc": [E.get("test_acc"), A.get("test_acc")], "torch": [E.get("torch"), A.get("torch")],
                     "wall_seconds": [E.get("wall_seconds"), A.get("wall_seconds")],
                     "sha256": {"env": sha(ej), "archived": sha(rj)}})
    rec = {"kind": "precondition E (TRANSFER_PLAN_v1.md)", "written_utc": datetime.datetime.now(datetime.timezone.utc)
           .strftime("%Y-%m-%dT%H:%M:%SZ"), "script_sha256": sha(pathlib.Path(__file__)), "fields": FIELDS,
           "pairs": rows, "all_identical": ok_all}
    out.write_text(json.dumps(rec, indent=1) + "\n")
    for row in rows:
        print(row["env_run"], "finished" if row["finished"] else "NOT FINISHED",
              "IDENTICAL" if row.get("identical") else "DIFFERS" if row["finished"] else "")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"precondition E: {'PASS' if ok_all else 'FAIL'}; record {shown}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # an unexpected error is never a verdict
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
