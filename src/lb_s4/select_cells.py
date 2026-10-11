"""S4 cell selection (the preregistration in force through lb.prereg.load(): selection.rule, selection.empty_class,
memory_knobs.feasibility_rule, caps_kib, control_cap, seeds.selection, training.lr_selection, code_freeze_s3).

  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m lb_s4.select_cells \
      --profiles result/s4/profile_registry --runs result/s4/select --lr research/s3/v4/lr_selected.json --out FILE

Per task, class (tied, untied) and cap (caps_kib, plus the control_cap "unbounded"):
  feasible iff min_peak_bytes <= cap_kib x 1024 (memory_knobs.feasibility_rule); every configuration is feasible at
  the unbounded cap; the selected configuration is the argmax of final_val_acc (seed 0, mode select, final epoch)
  over the feasible configurations of the class (selection.rule); ties go to the lower measured min_peak_bytes, then
  fewer parameters (selection.rule), then the lexicographically smallest config_id (PIN: final deterministic rule; the
  YAML is silent after "fewer parameters"); a class without a feasible configuration is null (selection.empty_class).
  A diverged select run keeps its accuracy (YAML failures; lb.train counts non-finite logits as wrong).
Fail-closed (exit 2, nothing written): exactly one profile and one select run per registry configuration (744 each;
missing, duplicate or extra is an ERROR); every input carries the prereg hash in force; select runs carry code_freeze
equal to lb.freeze.verify("harness", "predictor") now and code_sha256 (harness files) equal to the harness record;
profiles likewise (lb_s4.common.load_profiles); every select run has mode select, a seed in seeds.selection, status ok
or diverged, a finite final_val_acc, full length (max_steps null, truncated false; PIN: "final epoch"), microbatch 64
without checkpointing (memory_knobs.accuracy_invariance: each configuration is trained once at microbatch 64),
n_params equal to the registry's, and the lr selected for its (task, optimizer, class) in lr_selected.json.
Output (open "x"): per cell the selected tied and untied configuration with final_val_acc, min_peak_bytes,
argmin_knob, slack_bytes (cap x 1024 minus peak; null at the unbounded cap), n_params and lr; a descriptive
feasibility table (count of feasible configurations per task, class and cap); the sha256 of every input.
"""
import argparse
import json
import pathlib
import sys

from lb import freeze, prereg
from lb.select_lr import SelectionError, check_run
from lb_s4.common import (CLASSES, S4Error, check_complete, harness_and_records, inputs_sha, is_int,
                          load_lr_selected,
                          load_profiles, read_tree, registry_index, utc_now, write_x)

FINAL_TIE_BREAK = "lexicographically smallest config_id (PIN; the YAML selection.rule is silent after fewer parameters)"


def caps_with_control(p):
    """caps_kib in YAML order, then control_cap ("unbounded")."""
    return [int(c) for c in p["caps_kib"]] + [p["control_cap"]]


def is_feasible(peak_bytes, cap_kib, control):
    """memory_knobs.feasibility_rule: min-over-knobs peak <= cap x 1024 bytes; always true at the control cap."""
    return True if cap_kib == control else peak_bytes <= int(cap_kib) * 1024


def selection_key(c):
    """selection.rule order: higher final_val_acc, then lower min_peak_bytes, then fewer n_params; then (PIN)
    the lexicographically smallest config_id."""
    return (-float(c["final_val_acc"]), int(c["min_peak_bytes"]), int(c["n_params"]), c["config_id"])


def choose(cands):
    """The selected candidate (min of selection_key) or None for an empty list (selection.empty_class)."""
    return min(cands, key=selection_key) if cands else None


def load_select_runs(root, p, lrs, harness, records, yaml_sha):
    """{config_id: (path, result)}: exactly one select run per registry configuration, validated (module docstring)."""
    reg = registry_index()
    seeds = list(p["seeds"]["selection"])
    found = {}
    for path, r in read_tree(root, "result.json"):
        try:
            check_run(path, r, "select", harness, yaml_sha)
        except SelectionError as e:
            raise S4Error(str(e)) from e
        if r.get("code_freeze") != records:
            raise S4Error(f"{path}: code_freeze {r.get('code_freeze')} != current records {records}")
        check_complete(path, r)
        cid = r.get("config_id")
        if cid not in reg:
            raise S4Error(f"{path}: unexpected run {cid!r} (not a registry configuration)")
        if r.get("seed") not in seeds:
            raise S4Error(f"{path}: seed {r.get('seed')!r} not in seeds.selection {seeds}")
        if r.get("microbatch") != 64 or r.get("checkpointing") is not False:
            raise S4Error(f"{path}: microbatch {r.get('microbatch')!r} / checkpointing {r.get('checkpointing')!r}; "
                          "select runs train at microbatch 64 without checkpointing")
        e = reg[cid]
        want = lrs[(e["task"], e["optimizer"], e["class"])]
        if not isinstance(r.get("lr"), (int, float)) or float(r["lr"]) != want:
            raise S4Error(f"{path}: lr {r.get('lr')!r} != selected lr {want} for ({e['task']}, {e['optimizer']}, "
                          f"{e['class']})")
        if not is_int(r.get("n_params")) or r["n_params"] != e["n_params"]:
            raise S4Error(f"{path}: n_params {r.get('n_params')!r} != registry {e['n_params']}")
        if cid in found:
            raise S4Error(f"{path}: duplicate select run for {cid} (also {found[cid][0]})")
        found[cid] = (path, r)
    missing = sorted(set(reg) - set(found))
    if missing:
        raise S4Error(f"{len(missing)} of {len(reg)} select runs missing, e.g. {missing[:3]}")
    return found


def candidates(profiles, runs):
    """One row per registry configuration joining its profile and its select run."""
    out = []
    for cid, e in registry_index().items():
        ppath, prof = profiles[cid]
        rpath, r = runs[cid]
        out.append({"config_id": cid, "task": e["task"], "class": e["class"], "structure": e["structure"],
                    "d": e["d"], "T": e["T"], "optimizer": e["optimizer"],
                    "final_val_acc": float(r["final_val_acc"]), "status": r["status"],
                    "min_peak_bytes": int(prof["min_peak_bytes"]), "argmin_knob": prof["argmin_knob"],
                    "n_params": int(e["n_params"]), "lr": float(r["lr"]),
                    "result": freeze._key(rpath), "profile": freeze._key(ppath)})
    return out


def select_all(rows, p):
    """Cells (task x cap, control last) and the descriptive feasibility table."""
    control = p["control_cap"]
    cells, table = [], []
    for task in p["tasks"]:
        for cap in caps_with_control(p):
            cell = {"task": task, "cap_kib": cap, "cap_bytes": None if cap == control else int(cap) * 1024}
            for cls in CLASSES:
                pool = [c for c in rows if c["task"] == task and c["class"] == cls]
                feas = [c for c in pool if is_feasible(c["min_peak_bytes"], cap, control)]
                best = choose(feas)
                if best is not None:
                    best = dict(best, slack_bytes=None if cap == control else int(cap) * 1024 - best["min_peak_bytes"])
                cell[cls] = best
                table.append({"task": task, "class": cls, "cap_kib": cap, "n_feasible": len(feas), "n_total": len(pool)})
            cell["infeasible_class"] = [cls for cls in CLASSES if cell[cls] is None]
            cells.append(cell)
    return cells, table


def run(profiles_dir, runs_dir, lr_path):
    """Validate every input and return the output object (nothing is written)."""
    p = prereg.load()
    yaml_sha = prereg.yaml_sha256()
    harness, records = harness_and_records("harness", "predictor")
    lrs = load_lr_selected(lr_path, p, yaml_sha)
    profiles = load_profiles(profiles_dir, p, harness, records, yaml_sha)
    runs = load_select_runs(runs_dir, p, lrs, harness, records, yaml_sha)
    cells, table = select_all(candidates(profiles, runs), p)
    paths = ([pp for pp, _ in profiles.values()] + [rp for rp, _ in runs.values()]
             + [pathlib.Path(lr_path), freeze.record_path("harness"), freeze.record_path("predictor")])
    return {"kind": "cell_selection", "rule": p["selection"]["rule"], "empty_class": p["selection"]["empty_class"],
            "feasibility_rule": p["memory_knobs"]["feasibility_rule"], "final_tie_break": FINAL_TIE_BREAK,
            "caps_kib": [int(c) for c in p["caps_kib"]], "control_cap": p["control_cap"],
            "n_cells": len(cells), "cells": cells, "feasibility": table,
            "n_profiles": len(profiles), "n_select_runs": len(runs),
            "prereg_sha256": yaml_sha, "code_freeze": records, "inputs_sha256": inputs_sha(paths),
            "written_utc": utc_now()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="S4 selection per task, class and cap")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--lr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if pathlib.Path(a.out).exists():
            raise FileExistsError(f"{a.out} exists; refusing to overwrite")
        obj = run(a.profiles, a.runs, a.lr)
        h = write_x(a.out, obj)
    except (S4Error, freeze.CodeFreezeError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(a.out), "sha256": h, "n_cells": obj["n_cells"],
                      "infeasible": [[c["task"], c["cap_kib"], c["infeasible_class"]] for c in obj["cells"]
                                     if c["infeasible_class"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
