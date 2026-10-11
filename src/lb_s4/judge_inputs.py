"""Inputs of the G6 judge (src/lb/judge.py; its module docstring defines both schemas) from the S4 artifacts
(the preregistration in force through lb.prereg.load(): estimand, seeds, floor, selection.empty_class, conjuncts
N1.1 to N2.2, judge.top_up, code_freeze_s3).

  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m lb_s4.judge_inputs n1 \
      --selection FILE --confirm result/s4/confirm --floors research/s3/v4/floor_values.json --out N1.json
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m lb_s4.judge_inputs n2 \
      --predictions research/s3/v4/predictions_registry.json --profiles result/s4/profile_registry --out N2.json

n1: floors = floor_values.json "floor_values" (kind floor; prereg hash and recorded harness code_freeze checked;
  floor.rule / floor.instantiation are applied by lb.floor, not here). One cell per (task, finite cap) and one
  unbounded cell per task, in select_cells order; tied_config / untied_config = the selected configurations (null for
  an infeasible class, selection.empty_class). Seeds (judge.top_up): seeds.confirmation, or seeds.confirmation +
  seeds.top_up when top-up runs exist for EVERY selected configuration; a partial top-up is an ERROR. Accuracies are
  test_acc (percent) of the confirm runs, paired through the single seeds list (estimand.delta: paired by seed).
  PIN: in a cell with exactly one infeasible class, the feasible class keeps its paired accuracy list and the null
  class has an empty list (lb.judge ignores both lists of such a cell).
  Confirm runs: exactly one result.json per (selected configuration, seed); mode confirm, status ok or diverged,
  finite final_val_acc and test_acc, prereg hash, code_sha256 (harness files) equal to the harness record,
  code_freeze equal to lb.freeze.verify("harness", "predictor") now, the select-run lr recorded in the selection,
  microbatch 64 without checkpointing, full length (max_steps null, truncated false); a missing, duplicate or
  extra run is an ERROR.
n2: per registry configuration (744), measured_min_peak_bytes = the profile's min_peak_bytes; predicted and the three
  graded baselines' min-over-knobs bytes from the predictions file (parse_predictions isolates its schema, card S3-P).
  The predictions file must carry the prereg hash in force and, if it records code_freeze, exactly the harness and
  predictor records as they verify now (PIN: both names required when code_freeze is present).
Every ERROR: exit 2, nothing written. Outputs written with open "x"; provenance under the key "_provenance"
(unknown top-level keys are ignored by lb.judge).
"""
import argparse
import json
import pathlib
import sys

from lb import freeze, judge, prereg
from lb.select_lr import SelectionError, check_run
from lb_s4.common import (CLASSES, S4Error, check_complete, check_prereg, check_recorded_freeze,
                          harness_and_records, inputs_sha,
                          is_int, is_num, load_profiles, load_selection, read_json, read_tree, registry_index,
                          selected_configs, utc_now, write_x)

BASELINES = judge.BASELINES


# ------------------------------------------------------------------------------------------------ N1
def load_floors(path, p, yaml_sha):
    """{task: floor} from floor_values.json (lb.floor output)."""
    d = read_json(path, "floor values")
    if d.get("kind") != "floor":
        raise S4Error(f"{path}: kind {d.get('kind')!r} != 'floor'")
    check_prereg(path, d, yaml_sha)
    check_recorded_freeze(path, d.get("code_freeze"), required=("harness",))
    fv = d.get("floor_values")
    if not isinstance(fv, dict) or set(fv) != set(p["tasks"]):
        raise S4Error(f"{path}: floor_values tasks {sorted(fv) if isinstance(fv, dict) else fv} != YAML tasks")
    for t, v in fv.items():
        if not is_num(v) or v < 1.0:
            raise S4Error(f"{path}: floor {t} = {v!r} is not a finite number >= 1.0")
    return {t: float(fv[t]) for t in p["tasks"]}


def load_confirm_runs(root, p, selected, harness, records, yaml_sha):
    """{(config_id, seed): (path, result)} for the confirm runs of the selected configurations (module docstring)."""
    allowed = [int(s) for s in p["seeds"]["confirmation"]] + [int(s) for s in p["seeds"]["top_up"]]
    found = {}
    for path, r in read_tree(root, "result.json"):
        try:
            check_run(path, r, "confirm", harness, yaml_sha)
        except SelectionError as e:
            raise S4Error(str(e)) from e
        if r.get("code_freeze") != records:
            raise S4Error(f"{path}: code_freeze {r.get('code_freeze')} != current records {records}")
        check_complete(path, r)
        cid, seed = r.get("config_id"), r.get("seed")
        if cid not in selected:
            raise S4Error(f"{path}: unexpected confirm run {cid!r} (not a selected configuration)")
        if not is_int(seed) or seed not in allowed:
            raise S4Error(f"{path}: seed {seed!r} not in seeds.confirmation + seeds.top_up {allowed}")
        if not isinstance(r.get("lr"), (int, float)) or float(r["lr"]) != selected[cid]:
            raise S4Error(f"{path}: lr {r.get('lr')!r} != select-run lr {selected[cid]} of {cid}")
        if r.get("microbatch") != 64 or r.get("checkpointing") is not False:
            raise S4Error(f"{path}: microbatch {r.get('microbatch')!r} / checkpointing {r.get('checkpointing')!r}")
        if not is_num(r.get("test_acc")):
            raise S4Error(f"{path}: test_acc {r.get('test_acc')!r} is not a finite number")
        if (cid, seed) in found:
            raise S4Error(f"{path}: duplicate confirm run for ({cid}, seed {seed}) (also {found[(cid, seed)][0]})")
        found[(cid, seed)] = (path, r)
    return found


def seed_set(found, selected, p):
    """judge.top_up: confirmation seeds, or confirmation + top_up when every selected configuration has every top-up
    seed; a missing confirmation run or a partial top-up is an ERROR."""
    conf = [int(s) for s in p["seeds"]["confirmation"]]
    top = [int(s) for s in p["seeds"]["top_up"]]
    missing = sorted((c, s) for c in selected for s in conf if (c, s) not in found)
    if missing:
        raise S4Error(f"{len(missing)} confirm runs missing, e.g. {missing[:3]}")
    have = {c: sorted(s for s in top if (c, s) in found) for c in selected}
    if all(not v for v in have.values()):
        return conf
    if all(v == top for v in have.values()):
        return conf + top
    partial = sorted(c for c, v in have.items() if v != top)
    raise S4Error(f"partial top-up: {len(partial)} of {len(selected)} selected configurations lack top-up seeds "
                  f"{top}, e.g. {partial[:3]}")


def build_n1(selection_path, confirm_dir, floors_path):
    p = prereg.load()
    yaml_sha = prereg.yaml_sha256()
    harness, records = harness_and_records("harness", "predictor")
    sel = load_selection(selection_path, yaml_sha)
    floors = load_floors(floors_path, p, yaml_sha)
    selected = selected_configs(sel, selection_path)
    want = {(t, c) for t in p["tasks"] for c in [int(x) for x in p["caps_kib"]] + [p["control_cap"]]}
    got = [(c.get("task"), c.get("cap_kib")) for c in sel["cells"]]
    if len(got) != len(set(got)) or set(got) != want:
        raise S4Error(f"{selection_path}: cells {sorted(map(str, got))[:4]} are not exactly tasks x (caps + control)")
    found = load_confirm_runs(confirm_dir, p, selected, harness, records, yaml_sha)
    seeds = seed_set(found, selected, p)
    extra = sorted(k for k in found if k[1] not in seeds)
    if extra:
        raise S4Error(f"confirm runs outside the seed set {seeds}: {extra[:3]}")
    cells = []
    for c in sel["cells"]:
        tc = c["tied"]["config_id"] if c.get("tied") else None
        uc = c["untied"]["config_id"] if c.get("untied") else None
        cells.append({"task": c["task"], "cap_kib": c["cap_kib"], "tied_config": tc, "untied_config": uc,
                      "seeds": list(seeds),
                      "tied_test_acc": [] if tc is None else [float(found[(tc, s)][1]["test_acc"]) for s in seeds],
                      "untied_test_acc": [] if uc is None else [float(found[(uc, s)][1]["test_acc"]) for s in seeds]})
    paths = [pp for pp, _ in found.values()] + [pathlib.Path(selection_path), pathlib.Path(floors_path),
                                                 freeze.record_path("harness"), freeze.record_path("predictor")]
    return {"prereg_sha256": yaml_sha, "floors": floors, "cells": cells,
            "_provenance": {"kind": "judge_input_n1", "seed_set": seeds, "top_up_included": len(seeds) > len(
                p["seeds"]["confirmation"]), "n_selected_configs": len(selected), "n_confirm_runs": len(found),
                "code_freeze": records, "inputs_sha256": inputs_sha(paths), "written_utc": utc_now()}}


# ------------------------------------------------------------------------------------------------ N2
def _bytes(v, where):
    """A baseline or prediction value: an int, or an object carrying the min-over-knobs value as "min_bytes"."""
    if isinstance(v, dict):
        v = v.get("min_bytes")
    if not is_int(v) or v < 0:
        raise S4Error(f"{where}: {v!r} is not a non-negative integer byte count")
    return v


def parse_predictions(doc, path):
    """THE ONLY READER OF THE predictions_registry.json SCHEMA (card S3-P; src/lb/predictor.py predict_registry):
    {"configs": [{"config_id", "accounting": {"min_peak_bytes": int, ...},
                  "baselines": {name: int | {"min_bytes": int, ...}}}]}.
    Returns {config_id: {"predicted_min_peak_bytes": int, "baselines": {name: int}}}."""
    configs = doc.get("configs")
    if not isinstance(configs, list) or not configs:
        raise S4Error(f"{path}: configs is not a non-empty list")
    out = {}
    for k, c in enumerate(configs):
        where = f"{path}: configs[{k}]"
        cid = c.get("config_id") if isinstance(c, dict) else None
        if not isinstance(cid, str):
            raise S4Error(f"{where}: no config_id")
        if cid in out:
            raise S4Error(f"{where}: duplicate prediction for {cid}")
        acc = c.get("accounting")
        pred = _bytes(acc.get("min_peak_bytes") if isinstance(acc, dict) else None, f"{where} accounting.min_peak_bytes")
        bl = c.get("baselines")
        if not isinstance(bl, dict) or set(bl) != set(BASELINES):
            raise S4Error(f"{where}: baselines {sorted(bl) if isinstance(bl, dict) else bl} != {sorted(BASELINES)}")
        out[cid] = {"predicted_min_peak_bytes": pred, "baselines": {b: _bytes(bl[b], f"{where} {b}") for b in BASELINES}}
    return out


def build_n2(predictions_path, profiles_dir):
    p = prereg.load()
    yaml_sha = prereg.yaml_sha256()
    harness, records = harness_and_records("harness", "predictor")
    doc = read_json(predictions_path, "predictions")
    check_prereg(predictions_path, doc, yaml_sha)
    if "code_freeze" in doc:
        check_recorded_freeze(predictions_path, doc["code_freeze"], required=("harness", "predictor"))
    caps = [int(c) for c in p["caps_kib"]]
    if "caps_kib" in doc and doc["caps_kib"] != caps:
        raise S4Error(f"{predictions_path}: caps_kib {doc['caps_kib']} != YAML caps_kib {caps}")
    preds = parse_predictions(doc, predictions_path)
    reg = registry_index()
    if set(preds) != set(reg):
        miss, extra = sorted(set(reg) - set(preds)), sorted(set(preds) - set(reg))
        raise S4Error(f"{predictions_path}: predictions cover {len(preds)} configurations; missing {len(miss)} "
                      f"{miss[:3]}, extra {len(extra)} {extra[:3]}")
    profiles = load_profiles(profiles_dir, p, harness, records, yaml_sha)
    configs = [{"config_id": cid, "measured_min_peak_bytes": int(profiles[cid][1]["min_peak_bytes"]),
                "predicted_min_peak_bytes": preds[cid]["predicted_min_peak_bytes"],
                "baselines": dict(preds[cid]["baselines"])} for cid in reg]
    paths = [pp for pp, _ in profiles.values()] + [pathlib.Path(predictions_path), freeze.record_path("harness"),
                                                    freeze.record_path("predictor")]
    return {"prereg_sha256": yaml_sha, "caps_kib": caps, "configs": configs,
            "_provenance": {"kind": "judge_input_n2", "n_configs": len(configs), "code_freeze": records,
                            "inputs_sha256": inputs_sha(paths), "written_utc": utc_now()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="write the G6 judge's N1 or N2 input from S4 artifacts")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("n1")
    a1.add_argument("--selection", required=True)
    a1.add_argument("--confirm", required=True)
    a1.add_argument("--floors", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("n2")
    a2.add_argument("--predictions", required=True)
    a2.add_argument("--profiles", required=True)
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if pathlib.Path(a.out).exists():
            raise FileExistsError(f"{a.out} exists; refusing to overwrite")
        obj = build_n1(a.selection, a.confirm, a.floors) if a.cmd == "n1" else build_n2(a.predictions, a.profiles)
        h = write_x(a.out, obj)
    except (S4Error, freeze.CodeFreezeError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"cmd": a.cmd, "out": str(a.out), "sha256": h}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
