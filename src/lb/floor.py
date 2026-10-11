"""S3 floor instantiation (the preregistration in force (lb.prereg; PREREGISTERED_loopbudget_v4.yaml): floor.rule, floor.instantiation, code_freeze_s3).

  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.floor --runs DIR --lr research/s3/v4/lr_selected.json \
      --out research/s3/v4/floor_values.json

Requires exactly the floor runs, tasks x lb.train.FLOOR_SEEDS (3 x 5 = 15; the count is derived, not fixed): per
task, the untied reference configuration with adam at that task's selected lr
(lr_selected.json: selections[task]["adam"]["untied"]), seeds 101-105 (lb.train.FLOOR_SEEDS), mode floor; status,
accuracy and hashes are checked as in lb.select_lr. SD_task = sample standard deviation (ddof 1) of final_val_acc over
the five seeds; floor_task = max(1.0, 2 x SD_task) percentage points; if any floor run of a task diverged, that task's
floor is 1.0. Output written with open "x" (never overwritten). A missing seed, a duplicate, an unexpected run, a wrong
lr, a wrong hash or mode is an ERROR: exit 2, no output.
"""
import argparse
import json
import pathlib
import statistics
import sys

from lb import freeze, prereg, train
from lb.select_lr import SelectionError, check_run, harness_code, inputs_sha, read_results, utc_now, write_x

FLOOR_MIN = 1.0


def load_lr_selection(path, p):
    """{task: selected adam lr of the untied reference} from lr_selected.json (prereg hash checked)."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise SelectionError(f"{path} is missing")
    d = json.loads(path.read_text())
    if d.get("kind") != "lr_selection":
        raise SelectionError(f"{path}: kind {d.get('kind')!r} != 'lr_selection'")
    if d.get("prereg_sha256") != prereg.yaml_sha256():
        raise SelectionError(f"{path}: prereg_sha256 {d.get('prereg_sha256')} != YAML in force")
    out = {}
    for task in p["tasks"]:
        try:
            lr = float(d["selections"][task]["adam"]["untied"])
        except (KeyError, TypeError, ValueError) as e:
            raise SelectionError(f"{path}: no selected adam lr for the untied reference of {task}") from e
        if not any(lr == float(v) for v in p["training"]["lr_grid"]["adam"]):
            raise SelectionError(f"{path}: {task} lr {lr} not in lr_grid[adam]")
        out[task] = lr
    return out


def floor_value(accs, diverged):
    sd = statistics.stdev(accs)
    return {"sd": sd, "two_sd": 2.0 * sd, "floor": FLOOR_MIN if diverged else max(FLOOR_MIN, 2.0 * sd)}


def compute(runs_dir, lr_path):
    p = prereg.load()
    if "seeds 101-105" not in p["floor"]["rule"]:
        raise SelectionError("floor.rule does not state seeds 101-105")
    yaml_sha = prereg.yaml_sha256()
    record_sha = freeze.verify("harness")
    harness = harness_code(freeze.load_record("harness"))
    lrs = load_lr_selection(lr_path, p)
    cid_task = {train.reference_config_id(t, "untied", "adam", p): t for t in p["tasks"]}
    found = {}
    for path, r in read_results(runs_dir):
        check_run(path, r, "floor", harness, yaml_sha)
        cid, seed, lr = r.get("config_id"), r.get("seed"), r.get("lr")
        if cid not in cid_task or seed not in train.FLOOR_SEEDS:
            raise SelectionError(f"{path}: unexpected run {cid!r} seed {seed!r}")
        task = cid_task[cid]
        if not isinstance(lr, (int, float)) or float(lr) != lrs[task]:
            raise SelectionError(f"{path}: lr {lr!r} != selected lr {lrs[task]} for {task}")
        if (task, seed) in found:
            raise SelectionError(f"{path}: duplicate of {found[(task, seed)][0]}")
        found[(task, seed)] = (path, r)
    expected = {(t, s) for t in p["tasks"] for s in train.FLOOR_SEEDS}
    missing = sorted(expected - set(found))
    if missing:
        raise SelectionError(f"{len(missing)} of {len(expected)} floor runs missing: {missing}")
    if len(found) != len(expected):
        raise SelectionError(f"{len(found)} floor runs != {len(expected)}")
    floors = {}
    for task in p["tasks"]:
        runs = [found[(task, s)][1] for s in train.FLOOR_SEEDS]
        accs = [float(r["final_val_acc"]) for r in runs]
        diverged = [s for s, r in zip(train.FLOOR_SEEDS, runs) if r["status"] == "diverged"]
        floors[task] = {"config_id": train.reference_config_id(task, "untied", "adam", p), "lr": lrs[task],
                        "seeds": list(train.FLOOR_SEEDS), "final_val_acc": accs, "diverged_seeds": diverged,
                        **floor_value(accs, bool(diverged)),
                        "results": [freeze._key(found[(task, s)][0]) for s in train.FLOOR_SEEDS]}
    paths = sorted([path for path, _ in found.values()] + [pathlib.Path(lr_path), freeze.record_path("harness")])
    return {"kind": "floor", "rule": p["floor"]["rule"], "instantiation": p["floor"]["instantiation"],
            "unit": "percentage_points", "floors": floors, "floor_values": {t: floors[t]["floor"] for t in floors},
            "prereg_sha256": yaml_sha, "code_freeze": record_sha, "inputs_sha256": inputs_sha(paths),
            "written_utc": utc_now()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 floor values from the floor runs (tasks x seeds 101-105)")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if pathlib.Path(a.out).exists():
            raise FileExistsError(f"{a.out} exists; refusing to overwrite")
        obj = compute(a.runs, a.lr)
        h = write_x(a.out, obj)
    except (SelectionError, freeze.CodeFreezeError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(a.out), "sha256": h, "floor_values": obj["floor_values"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
