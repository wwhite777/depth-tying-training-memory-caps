"""Analysis of the cap-respecting transfer check (research/s9/TRANSFER_PLAN_v1.md, readouts R1-R4 and cost).
Exploratory, outside the frozen protocol; recomputes nothing of the frozen A1 verdict.

  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s9/transfer_analysis.py --design {seed1,seeds123} \
      --out OUT.json [--partial] [--comparator {archived,env}] [--root DIR]

Reads only (under --root, default the project root): result/s9/transfer/, result/s9/transfer_env/, result/s4/confirm/,
research/s4/selection.json, research/s4/judge_final_20261011T054347Z.json (cells) and the probe estimates
research/s9/transfer_probe/timing_20261014T070439Z.json (cost readout). Writes OUT.json (refuses to overwrite) with
the sha256 of every input read; prints a short summary.

Runs. Feasible run: result/s9/transfer/<stem>__lr<lr>__s<seed>__m1[_ckpt]. Comparator: --comparator archived (default)
result/s4/confirm/<stem>__lr<lr>__s<seed>; --comparator env (the plan's fallback when precondition E fails)
result/s9/transfer_env/<stem>__lr<lr>__s<seed>, fresh microbatch-64 reruns. Precondition E: the seed-1 reruns
result/s9/transfer_env/<archived name> of KWS|shared|d32|T2|adafactor and CIFAR10|shared|d24|T2|adafactor; test_acc,
final_val_acc and train_loss_per_epoch must equal the archived run exactly.

Refusal (exit 2) without --partial: any expected run missing (no DONE.json or result.json) or with a status other than
"ok" or with a record that does not match its spec; or precondition E not passed while --comparator archived.
With --partial: incomplete or non-ok runs are listed and left out of every statistic; the R1 verdict is reported
only when all 22 configurations are present (verdict_on_available otherwise).

Readouts (percentage points; delta = test_acc(feasible) - test_acc(comparator) per configuration and seed):
  R1  per-configuration mean of delta over seeds; mean over configurations, sample SD (ddof 1), se = SD/sqrt(n);
      90% t interval and two-sided 95% t interval, df = n - 1 (21 for 22 configurations).
      SUPPORTED if the 90% interval lies strictly inside (-0.5, +0.5) (two one-sided tests, alpha 0.05, margin 0.5);
      CONTRADICTED if the 95% interval excludes 0 and |mean| >= 0.5; UNRESOLVED otherwise. Per task: the same
      statistics, descriptive (no verdict).
  R2  |delta| per run / SD (ddof 1) of that configuration's archived test_acc over seeds 1-12 (microbatch 64);
      mean of the ratios; reference 2/sqrt(pi) = 1.128 (expected |X - Y| / sigma for independent N(0, sigma^2) runs).
  R3  per finite cell (judge_final n1.cells with finite_cap): D_feas = mean tied - mean distinct at the feasible
      setting, D_comp the same from the comparator runs, over the seeds run for all four arms; diff = D_feas - D_comp;
      sign change when one is > 0 and the other < 0 (zero signs listed separately).
  R4  per run: (L_feas[-1] - L_comp[-1]) / L_comp[-1] on train_loss_per_epoch, and the first epoch (1-based) at which
      the two curves differ by exact float comparison (None when identical).
  Cost per run: wall_seconds against the probe's additive estimate; summed core-hours (single-thread runs).
"""
import argparse
import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import scipy
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
SELECTION = "research/s4/selection.json"
JUDGE = "research/s4/judge_final_20261011T054347Z.json"
PROBE = "research/s9/transfer_probe/timing_20261014T070439Z.json"
CONFIRM = "result/s4/confirm"
TRANSFER = "result/s9/transfer"
ENV = "result/s9/transfer_env"
PRECOND_E = ("KWS|shared|d32|T2|adafactor", "CIFAR10|shared|d24|T2|adafactor")
SEEDS = {"seed1": (1,), "seeds123": (1, 2, 3)}
SD_SEEDS = tuple(range(1, 13))
MARGIN = 0.5
TASKS = ("HAR", "KWS", "CIFAR10")
REF_ABS_RATIO = 2.0 / math.sqrt(math.pi)
E_FIELDS = ("test_acc", "final_val_acc", "train_loss_per_epoch")


def stem(config_id):
    return config_id.replace("|", "__")


def fmt_lr(lr):
    return repr(float(lr))


def archived_name(cid, lr, seed):
    return f"{stem(cid)}__lr{fmt_lr(lr)}__s{seed}"


class Reader:
    """Reads JSON inputs under root and records the sha256 of every file read."""

    def __init__(self, root):
        self.root, self.sha = pathlib.Path(root), {}

    def json(self, rel):
        p = self.root / rel
        b = p.read_bytes()
        self.sha[str(rel)] = hashlib.sha256(b).hexdigest()
        return json.loads(b)

    def run(self, rel, expect):
        """(record or None, problem or None). expect: {field: value} the record must carry."""
        d = self.root / rel
        if not (d / "DONE.json").exists() or not (d / "result.json").exists():
            return None, "missing"
        r = self.json(f"{rel}/result.json")
        bad = {k: (r.get(k), v) for k, v in expect.items() if r.get(k) != v}
        if bad:
            return r, f"record mismatch {bad}"
        if r.get("status") != "ok":
            return r, f"status {r.get('status')!r}"
        return r, None


def selected_configs(sel):
    cfgs = {}
    for c in sel["cells"]:
        if c["cap_kib"] == "unbounded":
            continue
        for cls in ("tied", "untied"):
            e = c[cls]
            prev = cfgs.setdefault(e["config_id"], e)
            if prev["lr"] != e["lr"] or prev["argmin_knob"] != e["argmin_knob"]:
                raise SystemExit(f"selection.json disagrees across cells for {e['config_id']}")
    return cfgs


def t_stats(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {"n": n, "mean": float(x.mean()) if n else None}
    if n < 2:
        out.update(sd=None, se=None, df=None, ci90=None, ci95=None)
        return out
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    df = n - 1
    t90, t95 = float(stats.t.ppf(0.95, df)), float(stats.t.ppf(0.975, df))
    m = out["mean"]
    out.update(sd=sd, se=se, df=df, t_0_95=t90, t_0_975=t95, ci90=[m - t90 * se, m + t90 * se],
               ci95=[m - t95 * se, m + t95 * se])
    return out


def verdict(st, margin=MARGIN):
    if st["ci90"] is None:
        return None
    lo90, hi90 = st["ci90"]
    lo95, hi95 = st["ci95"]
    if lo90 > -margin and hi90 < margin:
        return "SUPPORTED"
    if (lo95 > 0 or hi95 < 0) and abs(st["mean"]) >= margin:
        return "CONTRADICTED"
    return "UNRESOLVED"


def first_diff_epoch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i + 1
    return None if len(a) == len(b) else min(len(a), len(b)) + 1


def analyse(root, design, comparator, partial):
    rd = Reader(root)
    sel = rd.json(SELECTION)
    judge = rd.json(JUDGE)
    probe = {r["config_id"]: r for r in rd.json(PROBE)["rows"]}
    cfgs = selected_configs(sel)
    problems = []
    if len(cfgs) != 22:
        raise SystemExit(f"{len(cfgs)} configurations in the finite cells, expected 22")

    # Precondition E
    pre = {"configs": list(PRECOND_E), "runs": {}, "passed": True}
    for cid in PRECOND_E:
        name = archived_name(cid, cfgs[cid]["lr"], 1)
        exp = {"config_id": cid, "seed": 1, "microbatch": 64, "checkpointing": False}
        new, pnew = rd.run(f"{ENV}/{name}", exp)
        old, pold = rd.run(f"{CONFIRM}/{name}", exp)
        if pnew or pold:
            pre["runs"][name] = {"status": "not_available", "rerun": pnew, "archived": pold}
            pre["passed"] = False
            continue
        eq = {f: new[f] == old[f] for f in E_FIELDS}
        pre["runs"][name] = {"status": "pass" if all(eq.values()) else "fail", "equal": eq,
                             "first_loss_diff_epoch": first_diff_epoch(new["train_loss_per_epoch"],
                                                                       old["train_loss_per_epoch"]),
                             "wall_seconds": new["wall_seconds"]}
        pre["passed"] &= all(eq.values())
    pre["result"] = "PASS" if pre["passed"] else "FAIL"
    if not pre["passed"] and comparator == "archived":
        problems.append("precondition E not passed (plan fallback: --comparator env)")

    # Runs
    comp_dir = CONFIRM if comparator == "archived" else ENV
    runs, excluded = [], []
    for cid, e in sorted(cfgs.items()):
        ckpt = bool(e["argmin_knob"]["checkpointing"])
        for s in SEEDS[design]:
            base = archived_name(cid, e["lr"], s)
            fname = base + "__m1" + ("_ckpt" if ckpt else "")
            f, pf = rd.run(f"{TRANSFER}/{fname}", {"config_id": cid, "seed": s, "microbatch": 1,
                                                   "checkpointing": ckpt})
            c, pc = rd.run(f"{comp_dir}/{base}", {"config_id": cid, "seed": s, "microbatch": 64,
                                                  "checkpointing": False})
            if pf or pc:
                excluded.append({"config_id": cid, "seed": s, "feasible": pf, "comparator": pc})
                problems.append(f"{cid} s{s}: feasible {pf}; comparator {pc}")
                continue
            lf, lc = f["train_loss_per_epoch"], c["train_loss_per_epoch"]
            est = probe[cid]["estimate"]["additive_s"]
            runs.append({"config_id": cid, "task": e["task"], "class": e["class"], "seed": s, "run": fname,
                         "checkpointing": ckpt, "test_feasible": f["test_acc"], "test_comparator": c["test_acc"],
                         "delta_pp": f["test_acc"] - c["test_acc"],
                         "final_loss_feasible": lf[-1], "final_loss_comparator": lc[-1],
                         "final_loss_rel_diff": (lf[-1] - lc[-1]) / lc[-1],
                         "first_loss_diff_epoch": first_diff_epoch(lf, lc), "n_epochs": len(lf),
                         "wall_seconds": f["wall_seconds"], "estimate_additive_s": est,
                         "wall_over_estimate": f["wall_seconds"] / est,
                         "comparator_wall_seconds": c["wall_seconds"]})

    complete = not problems
    if not complete and not partial:
        return None, problems

    # R1
    per_cfg = {}
    for r in runs:
        per_cfg.setdefault(r["config_id"], []).append(r["delta_pp"])
    cfg_rows = [{"config_id": k, "task": cfgs[k]["task"], "class": cfgs[k]["class"], "n_seeds": len(v),
                 "mean_delta_pp": float(np.mean(v))} for k, v in sorted(per_cfg.items())]
    st = t_stats([r["mean_delta_pp"] for r in cfg_rows])
    v = verdict(st)
    r1 = {"margin_pp": MARGIN, "rule": "SUPPORTED: 90% t interval strictly inside (-0.5, 0.5); CONTRADICTED: 95% "
          "interval excludes 0 and |mean| >= 0.5; else UNRESOLVED", "per_config": cfg_rows, **st,
          "verdict": v if (st["n"] == 22 and complete) else None,
          "verdict_on_available": v,
          "per_task": {t: t_stats([r["mean_delta_pp"] for r in cfg_rows if r["task"] == t]) for t in TASKS}}

    # R2
    r2_rows = []
    for r in runs:
        e = cfgs[r["config_id"]]
        accs = []
        for s in SD_SEEDS:
            a, pa = rd.run(f"{CONFIRM}/{archived_name(r['config_id'], e['lr'], s)}",
                           {"config_id": r["config_id"], "seed": s, "microbatch": 64, "checkpointing": False})
            if pa:
                raise SystemExit(f"archived seed {s} of {r['config_id']}: {pa}")
            accs.append(a["test_acc"])
        sd = float(np.std(accs, ddof=1))
        r2_rows.append({"config_id": r["config_id"], "seed": r["seed"], "abs_delta_pp": abs(r["delta_pp"]),
                        "archived_sd_pp": sd, "ratio": abs(r["delta_pp"]) / sd if sd > 0 else None})
    ratios = [x["ratio"] for x in r2_rows if x["ratio"] is not None]
    r2 = {"sd": "ddof 1 over archived seeds 1-12, microbatch 64", "runs": r2_rows,
          "mean_ratio": float(np.mean(ratios)) if ratios else None, "n": len(ratios),
          "reference_full_decorrelation": REF_ABS_RATIO}

    # R3
    by = {(r["config_id"], r["seed"]): r for r in runs}
    cells = [c for c in judge["n1"]["cells"] if c["finite_cap"]]
    if len(cells) != 21:
        raise SystemExit(f"{len(cells)} finite cells in judge_final, expected 21")
    r3_rows = []
    for c in cells:
        tc, uc = c["tied_config"], c["untied_config"]
        seeds = [s for s in SEEDS[design] if (tc, s) in by and (uc, s) in by]
        if not seeds:
            r3_rows.append({"task": c["task"], "cap_kib": c["cap_kib"], "tied": tc, "distinct": uc, "seeds": []})
            continue
        df_ = np.mean([by[(tc, s)]["test_feasible"] for s in seeds]) - np.mean([by[(uc, s)]["test_feasible"]
                                                                                 for s in seeds])
        dc = np.mean([by[(tc, s)]["test_comparator"] for s in seeds]) - np.mean([by[(uc, s)]["test_comparator"]
                                                                                   for s in seeds])
        r3_rows.append({"task": c["task"], "cap_kib": c["cap_kib"], "tied": tc, "distinct": uc, "seeds": seeds,
                        "D_feasible_pp": float(df_), "D_comparator_pp": float(dc), "diff_pp": float(df_ - dc),
                        "sign_change": bool(df_ * dc < 0), "zero_sign": bool(df_ == 0 or dc == 0)})
    r3 = {"cells": r3_rows, "n_cells": len(r3_rows),
          "n_sign_changes": sum(bool(x.get("sign_change")) for x in r3_rows),
          "sign_change_cells": [[x["task"], x["cap_kib"]] for x in r3_rows if x.get("sign_change")],
          "zero_sign_cells": [[x["task"], x["cap_kib"]] for x in r3_rows if x.get("zero_sign")]}

    # R4
    rel = [abs(r["final_loss_rel_diff"]) for r in runs]
    fde = [r["first_loss_diff_epoch"] for r in runs]
    r4 = {"runs": [{k: r[k] for k in ("config_id", "seed", "final_loss_rel_diff", "first_loss_diff_epoch",
                                      "n_epochs")} for r in runs],
          "max_abs_final_rel_diff": max(rel) if rel else None,
          "median_abs_final_rel_diff": float(np.median(rel)) if rel else None,
          "n_identical_curves": sum(x is None for x in fde),
          "first_diff_epoch_counts": {str(k): fde.count(k) for k in sorted(set(fde), key=lambda z: (z is None, z))}}

    # Cost
    wall = [r["wall_seconds"] for r in runs]
    env_walls = {k: v["wall_seconds"] for k, v in pre["runs"].items() if "wall_seconds" in v}
    if comparator == "env":
        env_walls.update({archived_name(r["config_id"], cfgs[r["config_id"]]["lr"], r["seed"]):
                          r["comparator_wall_seconds"] for r in runs})
    env_wall = list(env_walls.values())
    cost = {"method": "additive: W + K x (t1 - t64)",
            "runs": [{k: r[k] for k in ("config_id", "seed", "wall_seconds", "estimate_additive_s",
                                        "wall_over_estimate")} for r in runs],
            "core_hours_feasible": sum(wall) / 3600, "estimate_core_hours": sum(r["estimate_additive_s"]
                                                                               for r in runs) / 3600,
            "core_hours_env_reruns": sum(env_wall) / 3600, "n_env_reruns": len(env_wall),
            "core_hours_total": (sum(wall) + sum(env_wall)) / 3600,
            "median_wall_over_estimate": float(np.median([r["wall_over_estimate"] for r in runs])) if runs else None}

    res = {"kind": "s9_transfer_analysis", "design": design, "comparator": comparator, "partial": partial,
           "complete": complete, "labelling": "exploratory, outside the frozen protocol",
           "precondition_E": pre, "n_runs_used": len(runs), "excluded": excluded, "problems": problems,
           "R1": r1, "R2": r2, "R3": r3, "R4": r4, "cost": cost, "runs": runs,
           "scipy": scipy.__version__, "numpy": np.__version__, "argv": sys.argv,
           "inputs_sha256": dict(sorted(rd.sha.items()))}
    return res, problems


def summary(res):
    r1 = res["R1"]
    ci90 = "n/a" if r1["ci90"] is None else f"[{r1['ci90'][0]:+.3f}, {r1['ci90'][1]:+.3f}]"
    ci95 = "n/a" if r1["ci95"] is None else f"[{r1['ci95'][0]:+.3f}, {r1['ci95'][1]:+.3f}]"
    lines = [f"design {res['design']} comparator {res['comparator']} complete {res['complete']} "
             f"runs used {res['n_runs_used']} excluded {len(res['excluded'])}",
             f"precondition E: {res['precondition_E']['result']}",
             f"R1 mean delta {r1['mean']:+.3f} pp over {r1['n']} configs; 90% {ci90}; 95% {ci95}; "
             f"verdict {r1['verdict']} (on available: {r1['verdict_on_available']})" if r1["n"] else "R1: no runs",
             f"R2 mean |delta|/SD {res['R2']['mean_ratio']} (reference {REF_ABS_RATIO:.3f})",
             f"R3 sign changes {res['R3']['n_sign_changes']} of {res['R3']['n_cells']} cells "
             f"{res['R3']['sign_change_cells']}",
             f"R4 max |rel final loss diff| {res['R4']['max_abs_final_rel_diff']}; identical curves "
             f"{res['R4']['n_identical_curves']}",
             f"cost {res['cost']['core_hours_feasible']:.2f} core-h feasible vs estimate "
             f"{res['cost']['estimate_core_hours']:.2f}; total {res['cost']['core_hours_total']:.2f}"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True, choices=tuple(SEEDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--comparator", choices=("archived", "env"), default="archived")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    res, problems = analyse(a.root, a.design, a.comparator, a.partial)
    if res is None:
        print("REFUSED (exit 2): incomplete or failed inputs; rerun with --partial to inspect", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 2
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print(summary(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
