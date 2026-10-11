"""Synthetic S4 fixtures for test/lb_s4 (no run is trained or profiled): profiles, select runs, lr_selected.json,
floor_values.json, confirm runs, predictions and S3 bars runs, written into temporary directories. The code-freeze
records are written by lb.freeze.write_record into a patched temporary RECORD_DIR (never the real records).
"""
import contextlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lb import freeze, prereg  # noqa: E402
from lb.select_lr import harness_code  # noqa: E402
from lb_s4.common import expected_knobs, registry_index  # noqa: E402
from lb_s4.specs4 import name_of  # noqa: E402

P = prereg.load()
YAML = prereg.yaml_sha256()
REG = registry_index()
TASKS = list(P["tasks"])
CAPS = [int(c) for c in P["caps_kib"]]
CONTROL = P["control_cap"]
CONF = [int(s) for s in P["seeds"]["confirmation"]]
TOP = [int(s) for s in P["seeds"]["top_up"]]
PRED_SHA = "f" * 64          # a predictor*.py entry in code_sha256: ignored by the harness comparison


class Env:
    """Temporary directory + patched freeze.RECORD_DIR holding freshly written harness and predictor records."""

    def __init__(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._p = mock.patch.object(freeze, "RECORD_DIR", self.tmp / "records")
        self._p.start()
        freeze.write_record("harness")
        freeze.write_record("predictor")
        self.harness = harness_code(freeze.load_record("harness"))
        self.records = freeze.verify("harness", "predictor")
        self.code = dict(self.harness, **{"predictor.py": PRED_SHA})

    def close(self):
        self._p.stop()
        self._td.cleanup()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def edit_json(path, **kw):
    d = json.loads(Path(path).read_text())
    d.update(kw)
    Path(path).write_text(json.dumps(d))


@contextlib.contextmanager
def restored(*paths):
    """Snapshot files (or their absence) and restore them on exit: tests mutate a shared fixture tree."""
    snap = {Path(p): (Path(p).read_bytes() if Path(p).exists() else None) for p in paths}
    try:
        yield
    finally:
        for p, b in snap.items():
            if b is None:
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)


# ------------------------------------------------------------------------------------------------ S3 artifacts
def lr_selections():
    """Class-dependent lrs so that provenance errors are visible: tied = grid[0], untied = grid[1]."""
    g = P["training"]["lr_grid"]
    return {t: {o: {"tied": float(g[o][0]), "untied": float(g[o][1])} for o in P["registry"]["optimizers"]}
            for t in TASKS}


def lr_of(cid, sel=None):
    e = REG[cid]
    return (sel or lr_selections())[e["task"]][e["optimizer"]][e["class"]]


def write_lr_selected(path, env, selections=None, **override):
    d = {"kind": "lr_selection", "prereg_sha256": YAML, "code_freeze": {"harness": env.records["harness"]},
         "selections": selections or lr_selections()}
    d.update(override)
    return dump(path, d)


def write_floors(path, env, floors, **override):
    d = {"kind": "floor", "prereg_sha256": YAML, "code_freeze": {"harness": env.records["harness"]},
         "floor_values": dict(floors)}
    d.update(override)
    return dump(path, d)


# ------------------------------------------------------------------------------------------------ profiles, runs
def default_peak(cid):
    """150000 + 12 x n_params bytes; HAR untied + 120000, so that HAR's untied class is infeasible at 256 KiB only."""
    e = REG[cid]
    return 150000 + 12 * e["n_params"] + (120000 if e["task"] == "HAR" and e["class"] == "untied" else 0)


def default_acc(cid):
    return round(random.Random("acc|" + cid).uniform(40.0, 90.0), 6)


def profile_doc(cid, peak, env, **override):
    knobs = sorted(expected_knobs(REG[cid]["structure"], P))
    ms = [{"microbatch": m, "checkpointing": c, "peak_bytes": peak + (0 if (m, c) == (1, False) else 100 + 10 * i)}
          for i, (m, c) in enumerate(knobs)]
    d = {"config_id": cid, "measurements": ms, "min_peak_bytes": peak,
         "argmin_knob": {"microbatch": 1, "checkpointing": False},
         "prereg_sha256": YAML, "code_sha256": dict(env.code), "code_freeze": dict(env.records)}
    d.update(override)
    return d


def write_profiles(root, env, peak=default_peak):
    for cid in REG:
        dump(Path(root) / name_of(cid) / "profile.json", profile_doc(cid, peak(cid), env))


def run_doc(cid, env, mode, seed, lr, acc, **override):
    d = {"config_id": cid, "lr": float(lr), "seed": seed, "mode": mode, "status": "ok", "final_val_acc": acc,
         "test_acc": None, "microbatch": 64, "checkpointing": False, "max_steps": None, "truncated": False,
         "n_params": REG[cid]["n_params"] if cid in REG else 0,
         "prereg_sha256": YAML, "code_sha256": dict(env.code), "code_freeze": dict(env.records)}
    d.update(override)
    return d


def write_select_runs(root, env, acc=default_acc):
    for cid in REG:
        dump(Path(root) / name_of(cid) / "result.json", run_doc(cid, env, "select", 0, lr_of(cid), acc(cid)))


def confirm_path(root, cid, seed):
    return Path(root) / f"{name_of(cid)}__s{seed}" / "result.json"


def write_confirm(root, env, selected, seeds, test_acc):
    """selected: {config_id: lr}; test_acc(cid, seed) -> percent."""
    for cid, lr in selected.items():
        for s in seeds:
            dump(confirm_path(root, cid, s), run_doc(cid, env, "confirm", s, lr, 50.0, test_acc=test_acc(cid, s)))


def write_predictions(path, env, measured, **override):
    """Accounting = measured exactly; baselines: two that always fit and one that never fits (predictor schema of
    src/lb/predictor.py, the third baseline as an object with min_bytes)."""
    configs = [{"config_id": cid, "n_params": REG[cid]["n_params"],
                "accounting": {"min_peak_bytes": measured(cid), "argmin_knob": {"microbatch": 1, "checkpointing": False}},
                "baselines": {"param_count_with_state_multiplier": 0,
                              "persistent_weights_gradients_optimizer": 10 ** 9,
                              "static_four_component_sum": {"min_bytes": 0, "argmin_knob": {}}}} for cid in REG]
    d = {"name": "predictions_registry", "prereg_sha256": YAML, "code_freeze": dict(env.records),
         "caps_kib": CAPS, "configs": configs}
    d.update(override)
    return dump(path, d)


def write_bars(root, env, linear_acc):
    """The S3 bars runs of lb.specs.bars_specs (tasks x (1 + len(lr_grid[adam])) = 18 under v4; mode dev, seed 0);
    linear_acc: {(task, lr): acc}."""
    for t in TASKS:
        dump(Path(root) / f"{t}__majority" / "result.json",
             run_doc(f"{t}|majority", env, "dev", 0, 0.0, 30.0, code_freeze=None))
        for lr in P["training"]["lr_grid"]["adam"]:
            dump(Path(root) / f"{t}__linear__{lr}" / "result.json",
                 run_doc(f"{t}|linear", env, "dev", 0, float(lr), linear_acc[(t, float(lr))], code_freeze=None))


def expected_cells(peak=default_peak, acc=default_acc):
    """Independent brute-force re-derivation of the selection from the fixture functions:
    {(task, cap): {cls: config_id or None}}."""
    out = {}
    for t in TASKS:
        for cap in CAPS + [CONTROL]:
            out[(t, cap)] = {}
            for cls in ("tied", "untied"):
                pool = [c for c, e in REG.items() if e["task"] == t and e["class"] == cls
                        and (cap == CONTROL or peak(c) <= cap * 1024)]
                pool.sort(key=lambda c: (-acc(c), peak(c), REG[c]["n_params"], c))
                out[(t, cap)][cls] = pool[0] if pool else None
    return out
