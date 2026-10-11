"""Registry enumeration from the frozen preregistration (read only through lb.prereg.load(), which verifies the hash).

enumerate(task): the 248 registry configurations of a task (structure x width x depth x optimizer).
calibration_grid(task): the same rules at widths 40 and 56 (no wide widths), for the feasibility predictor's calibration.
"""
import torch

from lb import prereg
from lb.models import TIED_CLASS, UNTIED_CLASS, build_model

DEPTHS = {"shared": (2, 4, 8), "two_groups": (4, 8), "reversible_shared": (2, 4, 8),
          "untied": (2, 4, 8), "reversible_untied": (2, 4, 8), "wide_single": (1,)}
TIED_PER_TASK = 128
UNTIED_PER_TASK = 120
CALIBRATION_WIDTHS = (40, 56)


def _check_rules(reg):
    if tuple(reg["tied_class"]) != TIED_CLASS or tuple(reg["untied_class"]) != UNTIED_CLASS:
        raise prereg.PreregMismatch("tied/untied class lists differ from lb.models")
    for s, ts in DEPTHS.items():
        rule = reg["rules"][s]
        want = "T = 1" if ts == (1,) else "T in {" + ", ".join(str(t) for t in ts) + "}"
        if want not in rule:
            raise prereg.PreregMismatch(f"rule for {s!r} does not state {want!r}: {rule!r}")
    if sorted({t for ts in DEPTHS.values() for t in ts}) != sorted(reg["depths"]):
        raise prereg.PreregMismatch("depth sets differ from registry.depths")


def _n_params(cfg, structure, d, T, cache):
    key = (structure, d, T)
    if key not in cache:
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            cache[key] = build_model(cfg, structure, d, T).n_params
    return cache[key]


def _grid(task, widths, wide_widths, p):
    reg = p["registry"]
    _check_rules(reg)
    cfg = prereg.task_cfg(task, p)
    cache, out = {}, []
    for s in list(reg["tied_class"]) + list(reg["untied_class"]):
        ws = list(widths) + (list(wide_widths) if s == "wide_single" else [])
        for d in ws:
            for T in DEPTHS[s]:
                for opt in reg["optimizers"]:
                    out.append({"config_id": f"{task}|{s}|d{d}|T{T}|{opt}", "task": task, "structure": s,
                                "d": int(d), "T": int(T), "optimizer": opt, "tied": s in TIED_CLASS,
                                "n_params": _n_params(cfg, s, int(d), int(T), cache)})
    return out


def enumerate(task):  # noqa: A001 (name fixed by the S3-M card)
    p = prereg.load()
    reg = p["registry"]
    out = _grid(task, reg["widths"], reg["wide_widths"], p)
    n_tied = sum(1 for c in out if c["tied"])
    if len(out) != int(reg["configurations_per_task"]):
        raise AssertionError(f"{task}: {len(out)} configurations != {reg['configurations_per_task']}")
    if (n_tied, len(out) - n_tied) != (TIED_PER_TASK, UNTIED_PER_TASK):
        raise AssertionError(f"{task}: class sizes {n_tied}/{len(out) - n_tied} != {TIED_PER_TASK}/{UNTIED_PER_TASK}")
    if len({c["config_id"] for c in out}) != len(out):
        raise AssertionError(f"{task}: duplicate config_id")
    return out


def calibration_grid(task):
    p = prereg.load()
    return _grid(task, CALIBRATION_WIDTHS, (), p)
