"""Derived numbers for the S8 manuscript that the frozen judge does not print, computed from the frozen S3/S4 files.

Run from the project root with the project venv:
  ~/envs/jeongwoncheol_loopbudget/bin/python research/s8/derived_numbers.py --out research/s8/derived_numbers_<UTC>.json
Reads only frozen outputs (no training, no profiling). Exit 0 = written; exit 2 = a source is missing, a hash differs
from the value recorded in DECISION_LOG.md, or a consistency assertion fails (nothing is written).
"""
import argparse
import collections
import datetime
import glob
import hashlib
import json
import pathlib
import re
import statistics
import sys

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

        out = {"name": "s8_derived_numbers", "written_utc": datetime.datetime.now(datetime.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%SZ"), "script_sha256": sha(pathlib.Path(__file__)),
               "sources_sha256": hashes, "values": R}
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
