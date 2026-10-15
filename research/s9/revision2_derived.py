"""Post hoc numbers for the second external-review revision (S9, 2026-10-15). They are computed only from frozen or
recorded outputs and are labelled post hoc in the paper:
  (a) Holm over the 14 distinct finite pairs pooled over both looks (28 tests per event family), next to the final-look
      14-pair family already reported (derived_numbers section 17);
  (b) heterogeneity of the transfer check: per-configuration and per-run shifts, runs beyond +-0.5 points, the per-task
      reading of the loss differences, and the UCI-HAR cell contrasts at both settings;
  (c) a seed-block sensitivity of the pooled shift: one mean over the 22 configurations per seed (seeds 1-3), t interval
      with 2 degrees of freedom.
Inputs: research/s4/judge_20261011T045319Z.json (first look), research/s4/judge_final_20261011T054347Z.json (final look),
research/s9/transfer_analysis_20261014T180351Z.json. Output: research/s9/revision2_derived_<UTC>.json.
  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s9/revision2_derived.py --out research/s9/revision2_derived_<UTC>.json
"""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
J1 = "research/s4/judge_20261011T045319Z.json"
J2 = "research/s4/judge_final_20261011T054347Z.json"
TA = "research/s9/transfer_analysis_20261014T180351Z.json"


def holm(p):
    """Holm-adjusted p values in input order (monotone, capped at 1)."""
    order = np.argsort(p)
    m, adj, run = len(p), [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def pairs(judge):
    """Distinct (tied, distinct) pairs of the finite cells with their cell labels and statistics."""
    out = collections.OrderedDict()
    for c in judge["n1"]["cells"]:
        if not c["in_holm_family"]:
            continue
        k = (c["task"], c["tied_config"], c["untied_config"])
        if k not in out:
            out[k] = {"cells": [], "p_plus": c["p_plus"], "p_minus": c["p_minus"], "p_zero": c["p_zero"],
                      "mean": c["mean"]}
        out[k]["cells"].append(c["cap_kib"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    srcs = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (J1, J2, TA)}
    j1, j2, ta = (json.loads((ROOT / p).read_text()) for p in (J1, J2, TA))
    v = {}

    # (a) 14 pairs x 2 looks
    p1, p2 = pairs(j1), pairs(j2)
    assert list(p1) == list(p2) and len(p2) == 14, "the two looks must have the same 14 pairs"
    names = [f"{k[0]}|{'-'.join(str(c) for c in p2[k]['cells'])}" for k in p2]
    fam = {}
    for ev in ("p_plus", "p_minus", "p_zero"):
        ps = [p1[k][ev] for k in p1] + [p2[k][ev] for k in p2]
        adj = holm(ps)
        sig = {}
        for i, nm in enumerate(names * 2):
            look = "first" if i < 14 else "final"
            if adj[i] < 0.05:
                sig[f"{look}::{nm}"] = {"p_holm_28": adj[i], "p_raw": ps[i]}
        fam[ev] = {"n_tests": len(ps), "significant_at_0.05": sig, "min_adjusted": min(adj)}
    v["holm_14_pairs_both_looks"] = fam

    # (b) transfer heterogeneity
    runs = ta["runs"]
    d = [r["delta_pp"] for r in runs]
    v["transfer_run_delta_range"] = [min(d), max(d)]
    v["transfer_runs_beyond_0_5"] = {"n": sum(abs(x) > 0.5 for x in d), "of": len(d)}
    pc = ta["R1"]["per_config"]
    v["transfer_config_mean_range"] = [min(x["mean_delta_pp"] for x in pc), max(x["mean_delta_pp"] for x in pc)]
    v["transfer_har_config_means"] = {x["config_id"]: x["mean_delta_pp"] for x in pc if x["task"] == "HAR"}
    rel = collections.defaultdict(list)
    for r in runs:
        rel[r["task"]].append(abs(r["final_loss_rel_diff"]))
    v["transfer_loss_rel_diff_median_by_task"] = {t: float(np.median(x)) for t, x in rel.items()}
    v["transfer_loss_rel_diff_median_all"] = float(np.median([abs(r["final_loss_rel_diff"]) for r in runs]))
    har = [r for r in runs if r["task"] == "HAR"]
    v["transfer_har_comparator_final_loss_max"] = max(r["final_loss_comparator"] for r in har)
    big = max(runs, key=lambda r: abs(r["final_loss_rel_diff"]))
    v["transfer_largest_loss_rel_diff"] = {k: big[k] for k in ("config_id", "seed", "final_loss_feasible",
                                                               "final_loss_comparator", "final_loss_rel_diff",
                                                               "delta_pp")}
    cells = {(c["task"], c["cap_kib"]): c for c in ta["R3"]["cells"]}
    v["transfer_har_cells"] = {f"{c[1]}": {"D_comparator_pp": cells[c]["D_comparator_pp"],
                                           "D_feasible_pp": cells[c]["D_feasible_pp"]}
                               for c in sorted(cells) if c[0] == "HAR"}

    # (c) seed blocks
    by_seed = collections.defaultdict(list)
    for r in runs:
        by_seed[r["seed"]].append(r["delta_pp"])
    assert sorted(by_seed) == [1, 2, 3] and all(len(x) == 22 for x in by_seed.values())
    m = np.array([np.mean(by_seed[s]) for s in (1, 2, 3)])
    se = float(m.std(ddof=1) / np.sqrt(3))
    t90 = float(stats.t.ppf(0.95, 2))
    v["transfer_seed_block"] = {"seed_means": m.tolist(), "mean": float(m.mean()), "df": 2,
                                "ci90": [float(m.mean() - t90 * se), float(m.mean() + t90 * se)]}

    rec = {"kind": "post hoc numbers of the second external-review revision", "labelling": "post hoc",
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "sources_sha256": srcs, "scipy": __import__("scipy").__version__, "values": v}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(v, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
