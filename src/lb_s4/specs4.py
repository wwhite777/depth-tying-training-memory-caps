"""Launcher spec files (JSONL, format in lb.launch) for the S4 runs; output dirs under result/s4/ (the preregistration
in force through lb.prereg.load(): registry, seeds, training.lr_selection, memory_knobs, selection, classical_bars).

  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb_s4.specs4 --kind profile_registry --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb_s4.specs4 --kind select --lr research/s3/v4/lr_selected.json --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb_s4.specs4 --kind confirm --selection SELECTION.json --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb_s4.specs4 --kind top_up --selection SELECTION.json --out SPECS.jsonl
  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb_s4.specs4 --kind bars_confirm --bars result/s3/v4/bars --out SPECS.jsonl

profile_registry: the 744 registry configurations, lb.profiler in min-over-knobs mode (no --microbatch);
                  out result/s4/profile_registry/<name>.
select:           the 744 registry configurations, lb.train mode select, seed(s) seeds.selection, microbatch 64, at
                  lr_selected.json's lr for (task, optimizer, class); class tied iff structure in registry.tied_class
                  (training.lr_selection); out result/s4/select/<name>__lr<lr>__s<seed>.
confirm:          every DISTINCT configuration selected in any cell of select_cells' output (21 finite + 3 unbounded
                  cells), at its select-run lr recorded there, seeds seeds.confirmation, mode confirm, microbatch 64;
                  out result/s4/confirm/<name>__lr<lr>__s<seed>.
top_up:           the same configurations and lrs, seeds seeds.top_up (judge.top_up), mode confirm, microbatch 64;
                  PIN: out under result/s4/confirm/ as well (the seeds are disjoint), so that judge_inputs n1
                  --confirm result/s4/confirm sees confirmation and top-up runs in one tree.
bars_confirm:     per task, the majority bar plus the linear bar at the adam lr whose S3 bars run (lb.specs bars:
                  mode dev, seed 0; exactly the runs of lb.specs.bars_specs under the YAML in force, tasks x
                  (1 + len(lr_grid[adam])) = 18 under v4, validated as in lb.select_lr) has the best final_val_acc,
                  ties to the smaller lr; seeds seeds.confirmation, mode confirm; out result/s4/bars_confirm/....
S3 inputs are CLI arguments (--lr, --bars; no default): under v4 research/s3/v4/lr_selected.json and
result/s3/v4/bars (lb.specs writes the S3 runs under result/s3/v<version>/).
<name> = config_id with "|" replaced by "__". run_id = the out path relative to result/s4. Spec files are written with
open "x" (never overwritten). Exit 0 written; 2 ERROR (nothing written).
"""
import argparse
import json
import pathlib
import sys

from lb import freeze, prereg, specs as lbspecs
from lb.select_lr import SelectionError, check_run
from lb_s4.common import (S4Error, harness_and_records, load_lr_selected, load_selection, read_tree, registry_index,
                          selected_configs)

KINDS = ("profile_registry", "select", "confirm", "top_up", "bars_confirm")
OUT_ROOT = "result/s4"
MICROBATCH = 64


def name_of(config_id):
    return config_id.replace("|", "__")


def _train_spec(subdir, config_id, lr, seed, mode):
    rel = f"{subdir}/{name_of(config_id)}__lr{float(lr):g}__s{seed}"
    return {"run_id": rel, "module": "lb.train", "out": f"{OUT_ROOT}/{rel}",
            "args": ["--config-id", config_id, "--lr", repr(float(lr)), "--seed", str(seed), "--mode", mode,
                     "--microbatch", str(MICROBATCH)]}


def profile_registry_specs():
    """The 744 registry configurations (registry.configurations_per_task x tasks), min over knobs."""
    out = []
    for cid in registry_index():
        rel = f"profile_registry/{name_of(cid)}"
        out.append({"run_id": rel, "module": "lb.profiler", "out": f"{OUT_ROOT}/{rel}", "args": ["--config-id", cid]})
    return out


def select_specs(lr_path, p=None):
    """The 744 registry configurations in mode select at lr_selected.json's lr for (task, optimizer, class)."""
    p = p or prereg.load()
    lrs = load_lr_selected(lr_path, p, prereg.yaml_sha256())
    return [_train_spec("select", cid, lrs[(e["task"], e["optimizer"], e["class"])], seed, "select")
            for cid, e in registry_index().items() for seed in p["seeds"]["selection"]]


def _selected(selection_path):
    sel = load_selection(selection_path, prereg.yaml_sha256())
    return selected_configs(sel, selection_path)


def confirm_specs(selection_path, p=None):
    """Distinct selected configurations x seeds.confirmation at their select-run lr, mode confirm."""
    p = p or prereg.load()
    return [_train_spec("confirm", cid, lr, int(s), "confirm")
            for cid, lr in sorted(_selected(selection_path).items()) for s in p["seeds"]["confirmation"]]


def top_up_specs(selection_path, p=None):
    """Distinct selected configurations x seeds.top_up at their select-run lr, mode confirm (out under confirm/)."""
    p = p or prereg.load()
    return [_train_spec("confirm", cid, lr, int(s), "confirm")
            for cid, lr in sorted(_selected(selection_path).items()) for s in p["seeds"]["top_up"]]


def best_linear_lrs(bars_dir, p=None):
    """{task: adam lr of the best S3 linear bar run}: exactly the lb.specs.bars_specs runs of the YAML in force
    (tasks x (1 + len(lr_grid[adam])); 18 under v4; mode dev, seed 0), each
    validated by lb.select_lr.check_run (status, finite accuracy, prereg hash, harness code); best final_val_acc,
    ties to the smaller lr."""
    p = p or prereg.load()
    yaml_sha = prereg.yaml_sha256()
    harness, _ = harness_and_records("harness")
    expected = {}
    for s in lbspecs.bars_specs(p):
        a = dict(zip(s["args"][::2], s["args"][1::2]))
        expected[(a["--config-id"], float(a["--lr"]), int(a["--seed"]))] = s["run_id"]
    found = {}
    for path, r in read_tree(bars_dir, "result.json"):
        try:
            check_run(path, r, "dev", harness, yaml_sha)
        except SelectionError as e:
            raise S4Error(str(e)) from e
        lr = r.get("lr")
        key = (r.get("config_id"), float(lr) if isinstance(lr, (int, float)) else lr, r.get("seed"))
        if key not in expected:
            raise S4Error(f"{path}: unexpected bars run {key}")
        if key in found:
            raise S4Error(f"{path}: duplicate bars run {key} (also {found[key][0]})")
        found[key] = (path, r)
    missing = sorted(set(expected) - set(found), key=str)
    if missing:
        raise S4Error(f"{len(missing)} of {len(expected)} S3 bars runs missing, e.g. {missing[:3]}")
    out = {}
    for task in p["tasks"]:
        lin = [(k[1], r["final_val_acc"]) for k, (_, r) in found.items() if k[0] == f"{task}|linear"]
        out[task] = min(lin, key=lambda x: (-float(x[1]), x[0]))[0]
    return out


def bars_confirm_specs(bars_dir, p=None):
    p = p or prereg.load()
    lrs = best_linear_lrs(bars_dir, p)
    out = []
    for task in p["tasks"]:
        for s in p["seeds"]["confirmation"]:
            out.append(_train_spec("bars_confirm", f"{task}|majority", 0.0, int(s), "confirm"))
            out.append(_train_spec("bars_confirm", f"{task}|linear", lrs[task], int(s), "confirm"))
    return out


def build(kind, lr_path=None, selection_path=None, bars_dir=None):
    need = {"select": ("--lr", lr_path), "confirm": ("--selection", selection_path),
            "top_up": ("--selection", selection_path), "bars_confirm": ("--bars", bars_dir)}
    if kind in need and need[kind][1] is None:
        raise S4Error(f"--kind {kind} needs {need[kind][0]}")
    if kind == "profile_registry":
        specs = profile_registry_specs()
    elif kind == "select":
        specs = select_specs(lr_path)
    elif kind == "confirm":
        specs = confirm_specs(selection_path)
    elif kind == "top_up":
        specs = top_up_specs(selection_path)
    elif kind == "bars_confirm":
        specs = bars_confirm_specs(bars_dir)
    else:
        raise S4Error(f"kind {kind!r} not in {KINDS}")
    ids = [s["run_id"] for s in specs]
    outs = [s["out"] for s in specs]
    if len(set(ids)) != len(ids) or len(set(outs)) != len(outs):
        raise S4Error(f"{kind}: duplicate run_id or out dir")
    return specs


def write(specs, out):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        for s in specs:
            f.write(json.dumps(s) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="write launcher spec files for the S4 runs")
    ap.add_argument("--kind", required=True, choices=KINDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", default=None)
    ap.add_argument("--selection", default=None)
    ap.add_argument("--bars", default=None)
    a = ap.parse_args(argv)
    try:
        if pathlib.Path(a.out).exists():
            raise FileExistsError(f"{a.out} exists; refusing to overwrite")
        specs = build(a.kind, a.lr, a.selection, a.bars)
        write(specs, a.out)
    except (S4Error, freeze.CodeFreezeError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"kind": a.kind, "n_specs": len(specs), "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
