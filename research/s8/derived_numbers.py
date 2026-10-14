"""Derived numbers for the S8 manuscript that the frozen judge does not print, computed from the frozen S3/S4 files.

Run from the project root (the S8 project venv was deleted on 2026-10-11; numpy, scipy and pyyaml suffice, e.g. the docs venv):
  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s8/derived_numbers.py --out research/s8/derived_numbers_<UTC>.json
Sections 1-12 are those of derived_numbers_20261011T111328Z.json (unchanged values); sections 13-24 were added on
2026-10-14 for the revision after two external reviews (boundary-pair base rate and false fits, static-estimate ratios by
class, persistent shares, structure of the distinct pairs, unadjusted and 14-pair Holm p, margins, 97.5% bounds, knob
settings and microbatch-64 peaks, Speech Commands silence overlap, fp32 knob-equivalence values, optimizer counts, runs
before protocol version 4).
Reads only frozen outputs (no training, no profiling). Exit 0 = written; exit 2 = a source is missing, a hash differs
from the value recorded in DECISION_LOG.md, or a consistency assertion fails (nothing is written).
"""
import argparse
import ast
import collections
import datetime
import glob
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys

import yaml
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = {
    "judge_final": ("research/s4/judge_final_20261011T054347Z.json",
                    "e0bc089578fbff704af37098269b2caa334bc12ff65094c7eacec242476e35b1"),
    "judge_first": ("research/s4/judge_20261011T045319Z.json",
                    "ed6375cf8b3f837dbe0886ad345fef847b66f31803fe0199aa64b173263d6d45"),
    "selection": ("research/s4/selection.json",
                  "c50acf65dbe9e1b2f4416903c5f50fb71b99126803f8dc6d7b7f7cec13402904"),
    "n2_input": ("research/s4/n2_input.json",
                 "f240c26497a8a79a4f14cd565a9046ea897655cb020e938d2612b4e6b9fcc515"),
    "calibration": ("research/s3/v4/predictor_calibration.json", None),
    "lr_selected": ("research/s3/v4/lr_selected.json", None),
    "floor_values": ("research/s3/v4/floor_values.json", None),
    "data_manifest": ("research/s3/data_manifest.json", None),
    "train_test_log": ("test/lb/test_train_20261010T085319Z_v4_conductor.log", None),
    "models_test_log": ("test/lb/test_models_20261010T085319Z_v4_conductor.log", None),
}


class Fail(Exception):
    pass


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_sources():
    out, hashes = {}, {}
    for k, (rel, want) in SOURCES.items():
        p = ROOT / rel
        if not p.exists():
            raise Fail(f"missing source {rel}")
        h = sha(p)
        if want and h != want:
            raise Fail(f"hash mismatch {rel}: {h} != {want}")
        hashes[rel] = h
        out[k] = p.read_text() if rel.endswith(".log") else json.loads(p.read_text())
    return out, hashes


def holm(pvals):
    """Holm step-down adjusted p-values for a dict name -> raw p."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    m, run, adj = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def cells(j):
    return {(c["task"], c["cap_kib"]): c for c in j["n1"]["cells"]}


# sections 13-24 (added 2026-10-14, card S9-A): extra sources, each hashed into sources_sha256 when read
EXTRA = {
    "registry_summary": ("research/s4/profile_registry_summary.json", None),
    "prereg_v4": ("PREREGISTERED_loopbudget_v4.yaml", None),
    "kws_train_meta": ("data/KWS/train_meta.json", None),
    "kws_val_meta": ("data/KWS/val_meta.json", None),
    "kws_test_meta": ("data/KWS/test_meta.json", None),
    "train_test_source": ("test/lb/test_train.py", None),
}
TIED_STRUCT = ("shared", "two_groups", "reversible_shared")
UNTIED_STRUCT = ("untied", "reversible_untied", "wide_single")
BASE_NAMES = {"B1": "param_count_with_state_multiplier", "B2": "persistent_weights_gradients_optimizer",
              "B3": "static_four_component_sum"}
TASKS = ("HAR", "KWS", "CIFAR10")


def read_hashed(rel, hashes, want=None):
    p = ROOT / rel
    if not p.exists():
        raise Fail(f"missing source {rel}")
    b = p.read_bytes()
    h = hashlib.sha256(b).hexdigest()
    if want and h != want:
        raise Fail(f"hash mismatch {rel}: {h} != {want}")
    hashes[rel] = h
    return b


def cls_of(cid):
    s = cid.split("|")[1]
    if s in TIED_STRUCT:
        return "tied"
    if s in UNTIED_STRUCT:
        return "untied"
    raise Fail(f"unknown structure in {cid}")


def mmm(xs):
    xs = sorted(xs)
    return {"min": xs[0], "median": statistics.median(xs), "max": xs[-1], "n": len(xs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    try:
        S, hashes = load_sources()
        fin, first = cells(S["judge_final"]), cells(S["judge_first"])
        finite = [k for k, c in fin.items() if c["in_holm_family"]]
        if len(finite) != 21:
            raise Fail(f"expected 21 finite cells, got {len(finite)}")
        R = {}

        # 1. two-look sensitivity: Holm over both looks (42 tests) per event family
        two = {}
        for fam in ("p_minus", "p_plus", "p_zero"):
            raw = {}
            for k in finite:
                raw[f"look1::{k[0]}|{k[1]}"] = first[k][fam]
                raw[f"look2::{k[0]}|{k[1]}"] = fin[k][fam]
            adj = holm(raw)
            sig = sorted(n for n, v in adj.items() if v < 0.05)
            two[fam] = {"n_tests": len(raw), "significant_at_0.05": sig,
                        "CIFAR10|406_look2": adj["look2::CIFAR10|406"],
                        "CIFAR10|406_look2_single_look_holm": fin[("CIFAR10", 406)][fam + "_holm"]}
        R["two_look_holm"] = two

        # 2. distinct configuration pairs among cells
        pair = {k: (c["tied_config"], c["untied_config"]) for k, c in fin.items()}
        R["distinct_pairs_finite"] = len({pair[k] for k in finite})
        R["distinct_pairs_all24"] = len(set(pair.values()))
        R["distinct_pairs_finite_per_task"] = {t: len({pair[k] for k in finite if k[0] == t}) for t in ("HAR", "KWS", "CIFAR10")}
        R["repeated_pairs"] = sorted([[f"{k[0]}|{k[1]}" for k in finite if pair[k] == p] for p in {pair[k] for k in finite}
                                      if sum(1 for k in finite if pair[k] == p) > 1])

        # 3. per-cell display values and summaries
        def rng(t):
            ms = [fin[k]["mean"] for k in finite if k[0] == t]
            return [min(ms), max(ms)]
        R["mean_range_per_task"] = {t: rng(t) for t in ("HAR", "KWS", "CIFAR10")}
        R["n_finite_upper_below_floor"] = sum(1 for k in finite if fin[k]["upper_bound_below_floor"])
        R["not_below_floor"] = [[k[0], k[1], fin[k]["mean"], fin[k]["upper_bound_95"]] for k in finite
                                if not fin[k]["upper_bound_below_floor"]]
        R["E_minus_cells"] = [[k[0], k[1], fin[k]["p_minus_holm"]] for k in finite if fin[k]["E_minus"]]
        R["zero_sign_cells"] = [[k[0], k[1], fin[k]["p_zero_holm"]] for k in finite if fin[k]["zero_sign"]]
        R["E_plus_any"] = any(fin[k]["E_plus"] for k in fin)
        R["unbounded_means"] = {t: fin[(t, "unbounded")]["mean"] for t in ("HAR", "KWS", "CIFAR10")}
        R["n_seeds_final"] = sorted({c["n"] for c in fin.values()})
        R["first_look_status"] = S["judge_first"]["n1_1"]["status"]
        R["first_look_seeds"] = S["judge_first"]["n1"]["seed_set"]
        R["final_disposition"] = S["judge_final"]["disposition"]
        R["floors"] = S["judge_final"]["n1"]["floors"]

        # 4. selection: feasibility counts, optimizers, validation accuracies
        sel = S["selection"]
        fe = {(r["task"], r["class"], r["cap_kib"]): r["n_feasible"] for r in sel["feasibility"]}
        caps = sel["caps_kib"]
        more = [[t, c] for t in ("HAR", "KWS", "CIFAR10") for c in caps if fe[(t, "tied", c)] > fe[(t, "untied", c)]]
        notmore = [[t, c, fe[(t, "tied", c)], fe[(t, "untied", c)]] for t in ("HAR", "KWS", "CIFAR10") for c in caps
                   if fe[(t, "tied", c)] <= fe[(t, "untied", c)]]
        R["feasible_tied_more_caps"] = len(more)
        R["feasible_exceptions"] = notmore
        R["feasible_counts"] = {f"{t}|{cl}|{c}": fe[(t, cl, c)] for (t, cl, c) in fe}
        R["any_empty_class"] = any(v == 0 for v in fe.values())
        opt = collections.Counter()
        non_ada = []
        for c in sel["cells"]:
            for cl in ("tied", "untied"):
                opt[c[cl]["optimizer"]] += 1
                if c[cl]["optimizer"] != "adafactor":
                    non_ada.append([c["task"], c["cap_kib"], cl, c[cl]["config_id"]])
        R["optimizer_counts_48"] = dict(opt)
        R["non_adafactor"] = non_ada
        ids = {c[cl]["config_id"] for c in sel["cells"] for cl in ("tied", "untied")}
        R["distinct_selected_configs"] = len(ids)
        R["har_val"] = {str(c["cap_kib"]): [c["tied"]["final_val_acc"], c["untied"]["final_val_acc"]]
                        for c in sel["cells"] if c["task"] == "HAR"}
        R["har_tied_val_higher_every_cap"] = all(v[0] > v[1] for v in R["har_val"].values())
        R["selected_peaks"] = {f"{c['task']}|{c['cap_kib']}": [c["tied"]["min_peak_bytes"], c["untied"]["min_peak_bytes"]]
                               for c in sel["cells"]}

        # 5. N2: boundary pairs, baseline error directions, discordant pairs per configuration
        n2 = S["n2_input"]
        capb = [c * 1024 for c in n2["caps_kib"]]
        pairs = [(c, B) for c in n2["configs"] for B in capb
                 if abs(c["measured_min_peak_bytes"] - B) <= 0.2 * B]
        per_cfg = collections.Counter(c["config_id"] for c, _ in pairs)
        R["boundary_pairs"] = len(pairs)
        R["boundary_configs"] = len(per_cfg)
        R["boundary_configs_by_count"] = {str(k): v for k, v in sorted(collections.Counter(per_cfg.values()).items())}
        meas = [c["measured_min_peak_bytes"] for c in n2["configs"]]
        R["measured_min_peak_range"] = [min(meas), max(meas)]
        R["accounting_exact"] = sum(1 for c in n2["configs"] if c["predicted_min_peak_bytes"] == c["measured_min_peak_bytes"])
        base = {}
        for b in ("param_count_with_state_multiplier", "persistent_weights_gradients_optimizer", "static_four_component_sum"):
            ff = fn = right = 0
            disc = collections.Counter()
            for c, B in pairs:
                mfit = c["measured_min_peak_bytes"] <= B
                bfit = c["baselines"][b] <= B
                afit = c["predicted_min_peak_bytes"] <= B
                right += bfit == mfit
                ff += bfit and not mfit
                fn += (not bfit) and mfit
                if afit == mfit and bfit != mfit:
                    disc[c["config_id"]] += 1
            ratio = sorted(c["baselines"][b] / c["measured_min_peak_bytes"] for c in n2["configs"])
            base[b] = {"right": right, "false_fit": ff, "false_nofit": fn,
                       "under_predicted_configs": sum(1 for c in n2["configs"]
                                                      if c["baselines"][b] < c["measured_min_peak_bytes"]),
                       "ratio_min": ratio[0], "ratio_median": statistics.median(ratio), "ratio_max": ratio[-1],
                       "discordant_configs": len(disc), "max_discordant_per_config": max(disc.values())}
        R["baselines"] = base
        jn22 = {c["baseline"]: c for c in S["judge_final"]["n2_2"]["comparisons"]}
        for b, v in base.items():
            if v["false_fit"] + v["false_nofit"] != jn22[b]["b"] or jn22[b]["c"] != 0:
                raise Fail(f"discordant count for {b} disagrees with the judge")
        R["n2_2_p_holm"] = {b: jn22[b]["p_holm"] for b in jn22}

        # 6. peak phases (registry minimum knob; calibration every knob)
        def phases(pattern, every_knob):
            cnt, where = collections.Counter(), []
            files = sorted(glob.glob(str(ROOT / pattern)))
            for f in files:
                p = json.loads(pathlib.Path(f).read_text())
                ak = p["argmin_knob"]
                for x in p["measurements"]:
                    hit = x["microbatch"] == ak["microbatch"] and x["checkpointing"] == ak["checkpointing"]
                    if every_knob or hit:
                        cnt[x["peak_phase"]] += 1
                        if hit and x["peak_phase"] != "ph_backward":
                            where.append([p["config_id"], ak, p["min_peak_bytes"]])
            return len(files), dict(cnt), where
        n_reg, ph_reg, where_reg = phases("result/s4/profile_registry/*/profile.json", False)
        n_cal, ph_cal, _ = phases("result/s3/v4/calib_profiles/*/profile.json", True)
        if n_reg != 744 or n_cal != 360:
            raise Fail(f"profile counts {n_reg}, {n_cal}")
        R["peak_phase_registry_min_knob"] = ph_reg
        R["peak_phase_registry_non_backward"] = where_reg
        R["peak_phase_calibration_every_knob"] = ph_cal

        # 7. calibration summary
        cal = S["calibration"]
        R["calibration"] = {"n_profiles": cal["n_profiles"], "n_knob_measurements": cal["errors"]["per_knob"]["n"],
                            "per_knob_abs_max": cal["errors"]["per_knob"]["abs_max"],
                            "n_fitted_constants": cal["n_fitted_constants"]}

        # 8. classical bars (seeds 1-10)
        bars = collections.defaultdict(list)
        for f in sorted(glob.glob(str(ROOT / "result/s4/bars_confirm/*/result.json"))):
            r = json.loads(pathlib.Path(f).read_text())
            bars[r["config_id"]].append((r["seed"], r["test_acc"], r.get("n_params"), r.get("lr")))
        R["bars"] = {k: {"n": len(v), "seeds": sorted(s for s, *_ in v), "mean_test_acc": statistics.mean(a for _, a, *_ in v),
                         "n_params": v[0][2], "lr": v[0][3]} for k, v in sorted(bars.items())}
        lin_c = R["bars"]["CIFAR10|linear"]["n_params"]
        R["cifar_linear_adam_persistent_bytes"] = 16 * lin_c
        R["cifar_linear_exceeds_caps_kib"] = [c for c in caps if 16 * lin_c > c * 1024]

        # 9. learning rates at an edge; floors; data sizes
        lr = S["lr_selected"]
        R["lr_selected_raw_keys"] = list(lr.keys())[:12]
        R["data_sizes"] = {t: {s: S["data_manifest"]["tasks"][t]["splits"][s]["n"] for s in ("train", "val", "test")}
                           for t in ("HAR", "KWS", "CIFAR10")}

        # 10. equivalence-gate maxima (v4 conductor test logs)
        log = S["train_test_log"]
        def block(name):
            m = re.search(r'"%s": \{(.*?)\}' % re.escape(name), log, re.S)
            if not m:
                raise Fail(f"block {name} not in the train test log")
            return {k: float(v) for k, v in re.findall(r'"(\w+)": ([0-9.eE+-]+)', m.group(1))}
        R["f64_gradient_gate_max_per_optimizer"] = block("f64_gradient_gate_max_per_optimizer")
        R["f64_parameter_gate_max_per_optimizer"] = block("f64_parameter_gate_max_per_optimizer")
        m = re.search(r'"f64_gradient_gate_n_comparisons": (\d+)', log)
        R["f64_gradient_gate_n_comparisons"] = int(m.group(1)) if m else None
        if not re.search(r"^OK\s*$", log, re.M):
            raise Fail("train test log does not end OK")
        mlog = S["models_test_log"]
        m1 = re.search(r"'checkpoint_cases': (\d+), 'checkpoint_max_rel': ([0-9.eE+-]+)", mlog)
        if not m1:
            raise Fail("checkpoint report not in the models test log")
        R["checkpoint_cases"], R["checkpoint_max_rel"] = int(m1.group(1)), float(m1.group(2))

        # 11. mean test accuracy (seeds 1-12) of every selected configuration, per task and class
        conf = collections.defaultdict(list)
        for f in sorted(glob.glob(str(ROOT / "result/s4/confirm/*/result.json"))):
            r = json.loads(pathlib.Path(f).read_text())
            conf[r["config_id"]].append(r["test_acc"])
        if len(conf) != 25 or any(len(v) != 12 for v in conf.values()):
            raise Fail("confirmation runs are not 25 configurations x 12 seeds")
        rngs = {}
        for t in ("HAR", "KWS", "CIFAR10"):
            for cl in ("tied", "untied"):
                ids_tc = {c[cl]["config_id"] for c in sel["cells"] if c["task"] == t}
                m = [statistics.mean(conf[i]) for i in ids_tc]
                rngs[f"{t}|{cl}"] = [min(m), max(m)]
        R["selected_mean_test_acc_range"] = rngs
        R["selected_min_minus_linear_bar"] = {t: min(min(rngs[f"{t}|tied"]), min(rngs[f"{t}|untied"]))
                                              - R["bars"][f"{t}|linear"]["mean_test_acc"] for t in ("HAR", "KWS", "CIFAR10")}
        R["n_confirmation_runs"] = sum(len(v) for v in conf.values())

        # 12. compute: summed single-thread elapsed time of every run record (wall_seconds) and profile record
        #     (end_utc - start_utc); launcher overhead between records is not included
        def utc(s):
            return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        run_dirs = ["result/s3/lrcal", "result/s3/floor", "result/s3/bars", "result/s3/v4/lrcal", "result/s3/v4/floor",
                    "result/s3/v4/bars", "result/s4/select", "result/s4/confirm", "result/s4/bars_confirm"]
        prof_dirs = ["result/s3/calib_profiles", "result/s3/v4/calib_profiles", "result/s4/profile_registry"]
        hours = {}
        for d in run_dirs:
            fs = glob.glob(str(ROOT / d / "*/result.json"))
            if not fs:
                raise Fail(f"no run records in {d}")
            hours[d] = sum(json.loads(pathlib.Path(f).read_text())["wall_seconds"] for f in fs) / 3600
        for d in prof_dirs:
            fs = glob.glob(str(ROOT / d / "*/profile.json"))
            if not fs:
                raise Fail(f"no profile records in {d}")
            tot = 0.0
            for f in fs:
                p = json.loads(pathlib.Path(f).read_text())
                tot += (utc(p["end_utc"]) - utc(p["start_utc"])).total_seconds()
            hours[d] = tot / 3600
        R["compute_hours_by_folder"] = hours
        R["compute_hours_total_records"] = sum(hours.values())

        # ---------------- sections 13-24 (card S9-A, 2026-10-14); sections 1-12 above are unchanged ----------------
        D = {}  # new key -> short definition (written as the top-level "definitions" field)
        X = {k: read_hashed(rel, hashes, want) for k, (rel, want) in EXTRA.items()}
        reg = json.loads(X["registry_summary"])["profiles"]
        n2cfg = {c["config_id"]: c for c in n2["configs"]}
        if len(reg) != 744 or set(reg) != set(n2cfg) or any(
                reg[k]["min_peak_bytes"] != n2cfg[k]["measured_min_peak_bytes"] for k in reg):
            raise Fail("registry summary and n2_input disagree on the 744 configurations or their minimum peaks")

        def at_argmin(cid):
            p, ak = reg[cid], reg[cid]["argmin_knob"]
            hit = [x for x in p["measurements"]
                   if x["microbatch"] == ak["microbatch"] and x["checkpointing"] == ak["checkpointing"]]
            if (len(hit) != 1 or hit[0]["peak_bytes"] != p["min_peak_bytes"]
                    or min(x["peak_bytes"] for x in p["measurements"]) != p["min_peak_bytes"]):
                raise Fail(f"argmin measurement inconsistent for {cid}")
            m = hit[0]
            if m["baseline_bytes"] != m["W_bytes"] + m["G_bytes"] + m["O_bytes"]:
                raise Fail(f"baseline_bytes != W + G + O for {cid}")
            return m

        def peak64(cid):
            hit = [x["peak_bytes"] for x in reg[cid]["measurements"] if x["microbatch"] == 64 and not x["checkpointing"]]
            if len(hit) != 1:
                raise Fail(f"no unique microbatch-64 no-checkpointing measurement for {cid}")
            return hit[0]

        def parse(cid):
            t, s, d, T, o = cid.split("|")
            return {"task": t, "structure": s, "d": int(d[1:]), "T": int(T[1:]), "optimizer": o}

        sel_cells = {(c["task"], c["cap_kib"]): c for c in sel["cells"]}

        # 13. boundary pairs (|P - 1024B| <= 0.2 x 1024B, P = measured minimum peak, finite caps): error direction
        n_fit = sum(1 for c, B in pairs if c["measured_min_peak_bytes"] <= B)
        R["boundary_fit_base_rate"] = {"n_pairs": len(pairs), "n_fit": n_fit, "n_not_fit": len(pairs) - n_fit,
                                       "share_fit": n_fit / len(pairs)}
        D["boundary_fit_base_rate"] = ("boundary pairs (Eq. 6, finite caps) and how many fit by the measured minimum "
                                       "peak (P <= 1024 x cap); share_fit = n_fit / n_pairs")
        bd = {}
        for e in ("B1", "B2", "B3", "accounting"):
            anf = rf = ok = 0
            for c, B in pairs:
                mfit = c["measured_min_peak_bytes"] <= B
                pfit = (c["predicted_min_peak_bytes"] if e == "accounting" else c["baselines"][BASE_NAMES[e]]) <= B
                anf += pfit and not mfit
                rf += (not pfit) and mfit
                ok += pfit == mfit
            bd[e] = {"estimator": BASE_NAMES.get(e, "predicted_min_peak_bytes (lifetime-aware accounting)"),
                     "admitted_not_fit": anf, "rejected_fit": rf, "correct": ok, "accuracy_pct": 100 * ok / len(pairs)}
        if bd["accounting"]["admitted_not_fit"] or bd["accounting"]["rejected_fit"]:
            raise Fail("the accounting has boundary errors")
        for e in ("B1", "B2", "B3"):
            jb = jn22[BASE_NAMES[e]]
            if bd[e]["admitted_not_fit"] + bd[e]["rejected_fit"] != jb["b"] or jb["c"] != 0:
                raise Fail(f"{e} errors disagree with the judge's N2.2 discordant counts")
        R["boundary_errors_by_estimator"] = bd
        D["boundary_errors_by_estimator"] = ("per estimator over the boundary pairs: admitted_not_fit = predicted to fit "
                                             "but the measured peak exceeds the cap (false fit); rejected_fit = predicted "
                                             "not to fit but fits (false rejection); correct; accuracy_pct = 100 x "
                                             "correct / n_pairs; baseline errors equal the judge's N2.2 discordant b "
                                             "(c = 0), asserted")
        R["n2_2_discordant_judge"] = {e: {"b_for_accounting": jn22[BASE_NAMES[e]]["b"],
                                          "c_for_baseline": jn22[BASE_NAMES[e]]["c"]} for e in ("B1", "B2", "B3")}
        D["n2_2_discordant_judge"] = "N2.2 discordant counts copied from judge_final (b favours the accounting)"

        # 14. static-estimate ratio = baseline estimate / measured minimum peak, all 744 configurations, per class
        rat, allr = {}, []
        for e in ("B1", "B2", "B3"):
            per = {"tied": [], "untied": []}
            for c in n2["configs"]:
                per[cls_of(c["config_id"])].append(c["baselines"][BASE_NAMES[e]] / c["measured_min_peak_bytes"])
            both = per["tied"] + per["untied"]
            allr += both
            rat[e] = {"estimator": BASE_NAMES[e], "tied": mmm(per["tied"]), "untied": mmm(per["untied"]),
                      "overall_min": min(both), "overall_max": max(both)}
        R["static_estimate_ratio_by_class"] = rat
        R["static_estimate_ratio_overall_range"] = [min(allr), max(allr)]
        D["static_estimate_ratio_by_class"] = ("estimate / measured minimum peak over the 744 registry configurations; "
                                               "tied = shared, two_groups, reversible_shared; untied (distinct-parameter) "
                                               "= untied, reversible_untied, wide_single; min, median, max per class")
        D["static_estimate_ratio_overall_range"] = "min and max of the ratio over B1, B2, B3 and all 744 configurations"

        # 15. persistent share = baseline_bytes (W + G + O, L0 of Eq. 1) / min_peak_bytes at the argmin knob
        share = {cid: at_argmin(cid)["baseline_bytes"] / reg[cid]["min_peak_bytes"] for cid in reg}
        fin_ids = {cl: sorted({sel_cells[k][cl]["config_id"] for k in finite}) for cl in ("tied", "untied")}
        all_ids = {c[cl]["config_id"] for c in sel["cells"] for cl in ("tied", "untied")}
        ids_fin = set(fin_ids["tied"]) | set(fin_ids["untied"])
        if len(ids_fin) != 22 or len(all_ids - ids_fin) != 3:
            raise Fail(f"expected 22 finite and 3 unbounded-only selected configurations, got {len(ids_fin)}")
        R["persistent_share_selected_finite"] = {
            cl: {"min": min(share[i] for i in fin_ids[cl]), "max": max(share[i] for i in fin_ids[cl]),
                 "n_configs": len(fin_ids[cl])} for cl in ("tied", "untied")}
        D["persistent_share_selected_finite"] = ("share = baseline_bytes / min_peak_bytes at the argmin knob over the "
                                                 "distinct configurations selected in the 21 finite cells (42 class-cells, "
                                                 "22 configurations), min and max per class")
        R["persistent_share_unbounded_only"] = {i: share[i] for i in sorted(all_ids - ids_fin)}
        D["persistent_share_unbounded_only"] = "share of the selections that occur only in the unbounded cells"
        R["persistent_share_selected_per_config"] = {
            i: {"baseline_bytes": at_argmin(i)["baseline_bytes"], "min_peak_bytes": reg[i]["min_peak_bytes"],
                "share": share[i]} for i in sorted(all_ids)}
        D["persistent_share_selected_per_config"] = "the share and its two byte counts for each of the 25 selections"
        med, remmed, rem_all = {}, {}, [1 - v for v in share.values()]
        for t in TASKS:
            for cl in ("tied", "untied"):
                v = [share[i] for i in reg if i.split("|")[0] == t and cls_of(i) == cl]
                med[f"{t}|{cl}"] = statistics.median(v)
                remmed[f"{t}|{cl}"] = statistics.median(1 - x for x in v)
        R["persistent_share_registry_median"] = med
        R["persistent_share_registry_median_range"] = {
            cl: [min(v for k, v in med.items() if k.endswith(cl)), max(v for k, v in med.items() if k.endswith(cl))]
            for cl in ("|tied", "|untied")}
        D["persistent_share_registry_median"] = "median share per task x class over the 744 registry configurations"
        D["persistent_share_registry_median_range"] = "min and max of the three task medians per class"
        R["remainder_share_registry_range"] = [min(rem_all), max(rem_all)]
        R["remainder_share_registry_median"] = remmed
        R["remainder_share_registry_median_range"] = [min(remmed.values()), max(remmed.values())]
        D["remainder_share_registry_range"] = "remainder = 1 - share (activations and transients), min and max over 744"
        D["remainder_share_registry_median"] = "median remainder per task x class over the registry"
        D["remainder_share_registry_median_range"] = "min and max of the six task x class remainder medians"

        # 16. structure of the 14 distinct finite-cell pairs; n_params from result/s4/confirm/*/result.json
        conf_np, conf_knob = collections.defaultdict(set), collections.Counter()
        for f in sorted(glob.glob(str(ROOT / "result/s4/confirm/*/result.json"))):
            r = json.loads(read_hashed(str(pathlib.Path(f).relative_to(ROOT)), hashes))
            conf_np[r["config_id"]].add(r["n_params"])
            conf_knob[f"microbatch {r['microbatch']}, checkpointing {r['checkpointing']}"] += 1
        if any(len(v) != 1 for v in conf_np.values()):
            raise Fail("n_params differs across seeds of one configuration")
        npar = {k: next(iter(v)) for k, v in conf_np.items()}
        for c in sel["cells"]:
            for cl in ("tied", "untied"):
                if c[cl]["n_params"] != npar[c[cl]["config_id"]]:
                    raise Fail(f"n_params of {c[cl]['config_id']} differs between selection and confirm records")
        plist, pcaps = [], collections.defaultdict(list)
        for k in finite:
            if pair[k] not in pcaps:
                plist.append((k[0], pair[k]))
            pcaps[pair[k]].append(k[1])
        if len(plist) != 14:
            raise Fail(f"expected 14 distinct finite pairs, got {len(plist)}")
        ratios_np = [npar[u] / npar[t] for _, (t, u) in plist]
        wid = collections.Counter(">" if parse(u)["d"] > parse(t)["d"] else "=" if parse(u)["d"] == parse(t)["d"]
                                  else "<" for _, (t, u) in plist)
        R["finite_pairs_structure"] = {
            "n_pairs": len(plist),
            "tied_T_ge_4": sum(1 for _, (t, u) in plist if parse(t)["T"] >= 4),
            "untied_T_le_2": sum(1 for _, (t, u) in plist if parse(u)["T"] <= 2),
            "params_ratio_untied_over_tied_min": min(ratios_np), "params_ratio_untied_over_tied_max": max(ratios_np),
            "width_untied_gt_tied": wid[">"], "width_untied_eq_tied": wid["="], "width_untied_lt_tied": wid["<"],
            "n_params_source": "result/s4/confirm/*/result.json (n_params; identical across seeds and equal to "
                               "selection.json n_params, asserted)",
            "pairs": [{"task": tk, "caps_kib": pcaps[(t, u)], "tied": t, "untied": u, "tied_n_params": npar[t],
                       "untied_n_params": npar[u], "params_ratio": npar[u] / npar[t]} for tk, (t, u) in plist]}
        D["finite_pairs_structure"] = ("the distinct (tied, untied) selected pairs over the 21 finite cells: tied T >= 4, "
                                       "untied T <= 2 (wide_single has T = 1), parameter-count ratio untied / tied, "
                                       "and widths d compared")
        R["confirm_runs_knob"] = dict(conf_knob)
        D["confirm_runs_knob"] = "confirmation-run records by training knob (microbatch, checkpointing)"

        # 17. multiplicity readings (post hoc): Holm convention check, KWS cells, Holm over the 14 pairs
        fams = {"E_plus": "p_plus", "E_minus": "p_minus", "two_sided": "p_zero"}
        for fam in fams.values():
            adj = holm({f"{k[0]}|{k[1]}": fin[k][fam] for k in finite})
            if any(adj[f"{k[0]}|{k[1]}"] != fin[k][fam + "_holm"] for k in finite):
                raise Fail(f"derived_numbers.holm does not reproduce the judge's {fam}_holm")
        R["holm_convention_matches_judge"] = True
        D["holm_convention_matches_judge"] = ("derived_numbers.holm over the 21 finite cells reproduces judge_final's "
                                              "p_plus_holm, p_minus_holm and p_zero_holm exactly (asserted)")
        R["kws_upper_above_floor_p"] = {str(c): {"p_zero": fin[("KWS", c)]["p_zero"],
                                                 "p_zero_holm_21": fin[("KWS", c)]["p_zero_holm"],
                                                 "mean": fin[("KWS", c)]["mean"]} for c in (406, 645, 813)}
        D["kws_upper_above_floor_p"] = ("the three Speech Commands cells whose upper bound exceeds the floor: unadjusted "
                                        "two-sided p and its 21-cell Holm value, from judge_final")
        pp = {}
        for tk, pr in plist:
            ks = [k for k in finite if pair[k] == pr]
            lab = f"{tk}|" + "-".join(str(k[1]) for k in ks)
            for f in ("p_plus", "p_minus", "p_zero", "mean", "upper_bound_95", "floor"):
                if len({fin[k][f] for k in ks}) != 1:
                    raise Fail(f"cells of pair {lab} differ in {f}")
            pp[lab] = fin[ks[0]]
        h14 = {}
        for fn, fam in fams.items():
            adj = holm({lab: c[fam] for lab, c in pp.items()})
            h14[fn] = {"n_tests": len(adj), "min_adjusted": min(adj.values()),
                       "significant_at_0.05": {lab: {"p_holm_14": v, "p_raw": pp[lab][fam], "mean": pp[lab]["mean"]}
                                               for lab, v in sorted(adj.items(), key=lambda kv: kv[1]) if v < 0.05}}
        R["holm_over_14_pairs_final"] = h14
        D["holm_over_14_pairs_final"] = ("post hoc: Holm over the 14 distinct finite pairs instead of the 21 cells, final "
                                         "look, per event family (E_plus = p_plus, E_minus = p_minus, two_sided = p_zero); "
                                         "every pair with adjusted p < 0.05")
        R["pairs_upper_below_floor"] = {"n": sum(1 for c in pp.values() if c["upper_bound_below_floor"]),
                                        "of": len(pp),
                                        "not_below": [lab for lab, c in pp.items() if not c["upper_bound_below_floor"]]}
        D["pairs_upper_below_floor"] = "distinct finite pairs whose one-sided 95% upper bound U is below the floor"

        # 18. margins t_{0.95, n-1} x SD / sqrt(n), final look
        marg, mdiff = {}, 0.0
        for k in finite:
            c = fin[k]
            m = float(stats.t.ppf(0.95, c["n"] - 1)) * c["sd"] / math.sqrt(c["n"])
            mdiff = max(mdiff, abs((c["upper_bound_95"] - c["mean"]) - m))
            marg[f"{k[0]}|{k[1]}"] = m
        if mdiff > 1e-9:
            raise Fail(f"U - mean differs from the margin by {mdiff}")
        kw = [v for k, v in marg.items() if k.startswith("KWS|")]
        R["margin_95_final"] = {"min": min(marg.values()), "max": max(marg.values()), "KWS_min": min(kw),
                                "KWS_max": max(kw), "max_abs_diff_U_minus_mean": mdiff, "per_cell": marg}
        D["margin_95_final"] = ("t_{0.95, n-1} x SD / sqrt(n) per finite cell at n = 12 (scipy.stats.t.ppf); "
                                "max_abs_diff_U_minus_mean = largest |(U - mean) - margin| against judge_final")

        # 19. post hoc one-sided 97.5% bounds mean + t_{0.975, n-1} x SD / sqrt(n), both looks
        b975 = {}
        for look, J in (("first", first), ("final", fin)):
            vals = {}
            for k in finite:
                c = J[k]
                vals[f"{k[0]}|{k[1]}"] = (c["mean"] + float(stats.t.ppf(0.975, c["n"] - 1)) * c["sd"] / math.sqrt(c["n"]),
                                          c["floor"], c["n"])
            b975[look] = {"n_seeds": sorted({v[2] for v in vals.values()}),
                          "n_below_floor": sum(1 for v in vals.values() if v[0] < v[1]), "of": len(vals),
                          "not_below": {k: v[0] for k, v in vals.items() if not v[0] < v[1]}}
            if look == "final":
                b975[look]["KWS_406_645_813"] = [vals[f"KWS|{c}"][0] for c in (406, 645, 813)]
        R["upper_bound_97_5_post_hoc"] = b975
        D["upper_bound_97_5_post_hoc"] = ("post hoc one-sided 97.5% bound per finite cell, mean and SD from each judge "
                                          "file (first look n = 10, final n = 12); count below the floor")

        # 20. knob regime
        mb_reg = collections.Counter(reg[i]["argmin_knob"]["microbatch"] for i in reg)
        allow_ck = {i for i in reg if any(x["checkpointing"] for x in reg[i]["measurements"])}
        if allow_ck != {i for i in reg if not parse(i)["structure"].startswith("reversible")}:
            raise Fail("checkpointing-capable configurations are not exactly the non-reversible ones")
        cc = [(k, sel_cells[k][cl]["config_id"]) for k in finite for cl in ("tied", "untied")]
        r64 = [peak64(i) / (1024 * k[1]) for k, i in cc]
        R["knob_regime"] = {
            "registry_argmin_microbatch": {str(k): v for k, v in sorted(mb_reg.items())},
            "registry_checkpointing_allowed": len(allow_ck),
            "registry_checkpointing_on_at_argmin": sum(1 for i in allow_ck if reg[i]["argmin_knob"]["checkpointing"]),
            "classcells_42_argmin_microbatch": {str(k): v for k, v in sorted(collections.Counter(
                reg[i]["argmin_knob"]["microbatch"] for _, i in cc).items())},
            "classcells_42_checkpointing_on": sum(1 for _, i in cc if reg[i]["argmin_knob"]["checkpointing"]),
            "classcells_42_peak_mb64_nockpt_over_cap_min": min(r64),
            "classcells_42_peak_mb64_nockpt_over_cap_max": max(r64),
            "classcells_42_peak_mb64_nockpt_over_cap_gt1": sum(1 for v in r64 if v > 1), "n_classcells": len(cc)}
        D["knob_regime"] = ("argmin knob of the registry and of the 42 finite class-cells; checkpointing-capable = "
                            "configurations with a checkpointing measurement (asserted = non-reversible); peak at "
                            "microbatch 64 without checkpointing (the confirmation knob) / (1024 x cap)")

        # 21. Speech Commands silence clips
        meta = {s: json.loads(X[f"kws_{s}_meta"])["clips"] for s in ("train", "val", "test")}
        sil = {s: [c for c in meta[s] if c["word"] == "_silence_"] for s in meta}
        tr_off = collections.defaultdict(list)
        for c in sil["train"]:
            tr_off[c["source"]].append(c["offset"])
        ov = sum(1 for c in sil["val"] if any(abs(c["offset"] - o) < 16000 for o in tr_off[c["source"]]))
        R["kws_silence"] = {
            "counts": {s: len(v) for s, v in sil.items()},
            "val_overlapping_train_same_source": ov, "val_total": len(sil["val"]),
            "test_by_source_prefix": dict(collections.Counter(c["source"].split("/")[0] + "/" for c in sil["test"])),
            "test_with_offset": sum(1 for c in sil["test"] if "offset" in c)}
        D["kws_silence"] = ("silence clips (word _silence_) per split; validation clips whose [offset, offset + 16000) "
                            "overlaps a training clip interval of the same source file; test silence by source prefix")

        # 22. fp32 knob-equivalence evidence copied from the REPORT block of the v4 train test log
        i0 = log.index("REPORT ") + len("REPORT ")
        rep, _ = json.JSONDecoder().raw_decode(log, i0)
        ran = re.search(r"^Ran (\d+) tests", log, re.M)
        tree = ast.parse(X["train_test_source"].decode())
        names = [f"{c.name}.{f.name}" for c in tree.body if isinstance(c, ast.ClassDef)
                 and any(getattr(b, "attr", getattr(b, "id", "")) == "TestCase" for b in c.bases)
                 for f in c.body if isinstance(f, ast.FunctionDef) and f.name.startswith("test_")]
        if not ran or int(ran.group(1)) != len(names):
            raise Fail("test count in the log differs from the test methods in test/lb/test_train.py")
        gopt = collections.defaultdict(dict)
        for k, row in rep["fp32_gradient_reported"].items():
            o = k.split(" ")[0].split("|")[-1]
            for c, v in row.items():
                gopt[o][c] = max(gopt[o].get(c, v), v)
        R["fp32_knob_equivalence"] = {
            "log": SOURCES["train_test_log"][0], "tests_ran": int(ran.group(1)), "ok": True,
            "test_names_from_source": names,
            "parameter_max_rel_diff_per_optimizer": rep["fp32_parameter_reported_max_per_optimizer"],
            "gradient_max_rel_diff_all": rep["fp32_gradient_reported_max"],
            "gradient_max_rel_diff_per_task": rep["fp32_gradient_reported_max_per_task"],
            "gradient_max_rel_diff_per_optimizer_selected": dict(gopt),
            "n_gradient_entries": len(rep["fp32_gradient_reported"]),
            "n_parameter_entries": len(rep["fp32_parameter_reported"])}
        D["fp32_knob_equivalence"] = ("values copied from the REPORT block (test_fp32_reported): parameter max relative "
                                      "differences after three fp32 updates per optimizer, gradient maxima overall and per "
                                      "task; gradient_max_rel_diff_per_optimizer_selected = the largest logged entry per "
                                      "optimizer (a selection of logged values, no recomputation); test names parsed from "
                                      "test/lb/test_train.py (the log prints dots only), count asserted equal to 'Ran N'")

        # 23. optimizer counts
        unb = [c for c in sel["cells"] if c["cap_kib"] == "unbounded"]
        R["optimizer_counts_unbounded_6"] = dict(collections.Counter(c[cl]["optimizer"] for c in unb
                                                                     for cl in ("tied", "untied")))
        R["optimizer_unbounded_exceptions"] = [[c["task"], cl, c[cl]["config_id"]] for c in unb
                                               for cl in ("tied", "untied") if c[cl]["optimizer"] != "adafactor"]
        D["optimizer_counts_unbounded_6"] = "optimizers of the 6 unbounded class-cells (all 48: optimizer_counts_48)"
        D["optimizer_unbounded_exceptions"] = "unbounded class-cells not selecting Adafactor"

        # 24. runs before protocol version 4 (result/s3/{lrcal,floor,bars}; not result/s3/v4)
        frozen = str(yaml.safe_load(X["prereg_v4"])["date_frozen"])
        v3 = {}
        for g in ("lrcal", "floor", "bars"):
            fs = sorted(glob.glob(str(ROOT / f"result/s3/{g}/*/result.json")))
            if not fs:
                raise Fail(f"no v3 run records in result/s3/{g}")
            rs = [json.loads(read_hashed(str(pathlib.Path(f).relative_to(ROOT)), hashes)) for f in fs]
            last = max(r["end_utc"] for r in rs)
            v3[g] = {"n": len(rs), "by_mode": dict(collections.Counter(r["mode"] for r in rs)),
                     "n_test_acc_nonnull": sum(1 for r in rs if r.get("test_acc") is not None),
                     "n_test_acc_key_missing": sum(1 for r in rs if "test_acc" not in r),
                     "prereg_sha256": sorted({r.get("prereg_sha256") for r in rs}),
                     "latest_end_utc": last, "before_v4_freeze": utc(last) < utc(frozen)}
        R["runs_before_v4"] = {"v4_date_frozen": frozen, "groups": v3,
                               "any_test_acc": any(v["n_test_acc_nonnull"] for v in v3.values()),
                               "all_before_v4_freeze": all(v["before_v4_freeze"] for v in v3.values())}
        D["runs_before_v4"] = ("v3 run records under result/s3/{lrcal,floor,bars}: counts by mode, non-null test_acc, "
                               "latest end_utc against date_frozen of PREREGISTERED_loopbudget_v4.yaml")

        out = {"name": "s8_derived_numbers", "written_utc": datetime.datetime.now(datetime.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"), "script_sha256": sha(pathlib.Path(__file__)),
               "sources_sha256": hashes, "values": R, "definitions": D}
    except Fail as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    p = pathlib.Path(a.out)
    if p.exists():
        print(f"ERROR: {p} exists", file=sys.stderr)
        return 2
    p.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"written {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
