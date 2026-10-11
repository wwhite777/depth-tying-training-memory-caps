"""Launcher spec files (JSONL, format in lb.launch) for the S3 runs; output dirs under result/s3/v<version>/ (version =
the top-level "version" of the preregistration in force, lb.prereg: result/s3/v4/ under v4; the v3 runs stay in
result/s3/{lrcal,floor,bars}/).

  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.specs --kind lrcal --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.specs --kind floor --lr research/s3/v4/lr_selected.json --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.specs --kind bars --out SPECS.jsonl

Counts follow the YAML (v4 values in brackets):
lrcal: tasks x optimizers x 2 reference configurations x len(lr_grid[optimizer]) x seeds.selection, mode lrcal
       [3 x 4 x 2 x 5 x 1 = 120].
floor: per task, the untied reference with adam at its selected lr, seeds 101-105 (lb.train.FLOOR_SEEDS), mode floor
       [3 x 5 = 15].
bars:  per task, the majority bar plus the linear bar at each adam grid lr, mode dev, seed 0 [3 x (1 + 5) = 18].
run_id = <kind>/<task>__<structure>__<optimizer>__lr<lr>__s<seed>; out = result/s3/v<version>/<run_id>. The majority
bar has optimizer "none" and lr 0. Spec files are written with open "x" (never overwritten).
"""
import argparse
import json
import pathlib
import sys

from lb import prereg, train
from lb.floor import load_lr_selection
from lb.select_lr import SelectionError

KINDS = ("lrcal", "floor", "bars")


def out_root(p=None):
    """result/s3/v<version> (relative to the project root) of the YAML in force."""
    p = p or prereg.load()
    return f"result/s3/v{int(p['version'])}"


def _spec(kind, config_id, lr, seed, mode, p=None):
    cfg = config_id.split("|")
    task, structure = cfg[0], cfg[1]
    optimizer = cfg[4] if len(cfg) == 5 else ("adam" if structure == "linear" else "none")
    run_id = f"{kind}/{task}__{structure}__{optimizer}__lr{float(lr):g}__s{seed}"
    return {"run_id": run_id, "module": "lb.train", "out": f"{out_root(p)}/{run_id}",
            "args": ["--config-id", config_id, "--lr", repr(float(lr)), "--seed", str(seed), "--mode", mode]}


def lrcal_specs(p=None):
    p = p or prereg.load()
    out = []
    for task in p["tasks"]:
        for opt in p["registry"]["optimizers"]:
            for cls in ("untied", "tied"):
                cid = train.reference_config_id(task, cls, opt, p)
                for lr in p["training"]["lr_grid"][opt]:
                    for seed in p["seeds"]["selection"]:
                        out.append(_spec("lrcal", cid, lr, seed, "lrcal", p))
    return out


def floor_specs(lr_path, p=None):
    p = p or prereg.load()
    lrs = load_lr_selection(lr_path, p)
    return [_spec("floor", train.reference_config_id(task, "untied", "adam", p), lrs[task], seed, "floor", p)
            for task in p["tasks"] for seed in train.FLOOR_SEEDS]


def bars_specs(p=None):
    p = p or prereg.load()
    out = []
    for task in p["tasks"]:
        out.append(_spec("bars", f"{task}|majority", 0.0, 0, "dev", p))
        for lr in p["training"]["lr_grid"]["adam"]:
            out.append(_spec("bars", f"{task}|linear", lr, 0, "dev", p))
    return out


def build(kind, lr_path=None):
    if kind == "lrcal":
        specs = lrcal_specs()
    elif kind == "floor":
        if lr_path is None:
            raise SelectionError("--kind floor needs --lr lr_selected.json")
        specs = floor_specs(lr_path)
    elif kind == "bars":
        specs = bars_specs()
    else:
        raise ValueError(f"kind {kind!r} not in {KINDS}")
    ids = [s["run_id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise SelectionError(f"{kind}: duplicate run_id")
    return specs


def write(specs, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="write launcher spec files for the S3 runs")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", default=None)
    a = ap.parse_args(argv)
    try:
        specs = build(a.kind, a.lr)
        write(specs, a.out)
    except (SelectionError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"kind": a.kind, "n_specs": len(specs), "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
