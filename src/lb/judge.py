"""G6 judge for the loopbudget preregistration (the preregistration in force, read through lb.prereg.load(), which
verifies the frozen sha256). Computes N1.1, N1.2, the S4 disposition, N2.1 and N2.2 from the frozen rules; never by hand.

CLI:  python src/lb/judge.py --n1 FILE [--n2 FILE] --out FILE
      exit 0 for a computed verdict (status GO, KILL, INCONCLUSIVE or MISS_TOP_UP_REQUIRED), exit 2 for ERROR.

N1 input (JSON object):
  {"prereg_sha256": str (must equal lb.prereg.yaml_sha256()),
   "floors": {task: float percentage points}            exactly the YAML tasks; finite and >= 1.0 (YAML floor rule)
   "cells": [{"task": str, "cap_kib": int | "unbounded", "tied_config": str | null, "untied_config": str | null,
              "seeds": [int], "tied_test_acc": [float percent], "untied_test_acc": [float percent]}]}
  * Exactly one cell per (task, finite cap) for the 3 YAML tasks x 7 YAML caps; at most one "unbounded" cell per task.
  * tied_test_acc[i] and untied_test_acc[i] belong to seeds[i] (pairing by seed through the single seeds list).
  * A cell with a null config is infeasible for that class: no Delta, reported as infeasible, excluded from every test;
    its accuracy lists are not used (they must still be lists).
  * Every cell with a defined Delta (including unbounded cells) must carry the same seed set, without duplicates, and
    accuracy lists of the same length as seeds; accuracies finite and in [0, 100]. Otherwise ERROR.
  * The unbounded cell is reported but excluded from the Holm family and from N1.1.
N2 input (JSON object):
  {"prereg_sha256": str, "caps_kib": [int] (must equal the YAML caps_kib as a set),
   "configs": [{"config_id": str (unique), "measured_min_peak_bytes": int > 0, "predicted_min_peak_bytes": int >= 0,
                "baselines": {"param_count_with_state_multiplier": int >= 0,
                              "persistent_weights_gradients_optimizer": int >= 0,
                              "static_four_component_sum": int >= 0}}]}
Unknown top-level keys (for example "_synthetic") are ignored.

Frozen rules as implemented (pins marked PIN are interpretations reported to the conductor):
  Per defined cell: d_s = tied_s - untied_s; n, mean, sample SD (ddof 1), SE = SD / sqrt(n), df = n - 1 (n >= 2 else ERROR).
    E_plus  p = P(T_df >= (mean - floor) / SE)        (H1: Delta > floor)
    E_minus p = P(T_df <= (mean + floor) / SE)        (H1: Delta < -floor)
    zero    p = 2 P(T_df >= |mean| / SE)              (H1: Delta != 0)
    upper bound = mean + t_{0.95, df} SE (one-sided 95 percent).
    SD == 0 (all d_s identical): t = +inf or -inf in the direction of the numerator, p = 0 if the numerator lies in the
    H1 direction and 1 otherwise; PIN: a numerator of exactly 0 gives t undefined (reported null) and p = 1; the upper
    bound equals the mean.
  Holm over the finite-cap cells with a defined Delta (PIN: family size m = number of such cells, at most 21; infeasible
    cells carry no p value), separately for E_plus, E_minus and zero_sign; adjusted p_(j) = max_{i<=j} min(1, (m-i+1) p_(i));
    PIN: an event is significant iff adjusted p < alpha (strict).
  N1.1: PASS if any Holm-significant E_plus among finite cells; else FAIL if every finite cell with a defined Delta has
    upper bound < floor (strict); else MISS; ERROR if no finite cell has a defined Delta.
  N1.2 (descriptive, per task, finite caps ascending): lists of caps with significant E_plus, E_minus, zero_sign and a class:
    "multiple crossings" if E_plus and E_minus both present or the E_plus caps are not one contiguous run;
    "everywhere tested" if E_plus at every tested cap; "band" if one contiguous run; else "no event".
    PIN: contiguity and "every" are taken over the tested (defined-Delta) caps; an E_minus-only task is "no event"
    (no tied-advantage event; its E_minus caps are listed).
  Disposition: seed set must equal the YAML confirmation seeds or confirmation + top_up (PIN: applies to every N1.1
    result; any other seed set is ERROR). GO if PASS; KILL if FAIL; MISS with confirmation seeds -> MISS_TOP_UP_REQUIRED;
    MISS with confirmation + top_up -> INCONCLUSIVE.
  N2.1: rel_err = |pred - meas| / meas; boundary pairs = (config, cap) with |meas - cap*1024| <= 0.2 cap*1024; fit = peak
    <= cap*1024; PASS if share(rel_err <= 0.05) >= 0.95 and boundary fit accuracy >= 0.98; FAIL if share < 0.90 or
    accuracy < 0.95; else MISS; ERROR if there are no boundary pairs.
  N2.2: per baseline, exact McNemar on boundary pairs (b = predictor right and baseline wrong, c = predictor wrong and
    baseline right; p = min(1, 2 sum_{k <= min(b,c)} C(n,k) / 2^n), n = b + c, p = 1 if n = 0); Holm over the 3
    baselines; PASS if all three adjusted p < alpha with b > c; FAIL if any adjusted p < alpha with c > b; else MISS.
    ERROR if N2.1 has no boundary pairs.
  Overall status: ERROR if any validation failed or any of N1.1, disposition, N2.1, N2.2 is ERROR (PIN); otherwise the
    disposition. N2 not supplied: n2_1 and n2_2 are "NOT_RUN" and do not affect the status.
  Output JSON is strict (no NaN/Infinity): infinite t values are written as "+inf" / "-inf", undefined values as null.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import shlex
import sys
from fractions import Fraction
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy import stats  # noqa: E402

from lb import prereg as lbprereg  # noqa: E402

GATE = "G6 judge"
BASELINES = ("param_count_with_state_multiplier", "persistent_weights_gradients_optimizer", "static_four_component_sum")
VERDICTS = {"GO", "KILL", "INCONCLUSIVE", "MISS_TOP_UP_REQUIRED"}


class JudgeError(Exception):
    pass


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)) and math.isfinite(x)


def _req(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        raise JudgeError(f"missing field {key!r} in {where}")
    return obj[key]


# ---------------------------------------------------------------- statistics
def holm(pvals):
    """Holm step-down adjusted p values, returned in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p value (binomial, p = 1/2) on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(Fraction(math.comb(n, i)) for i in range(k + 1)) / Fraction(2) ** n
    return float(min(Fraction(1), 2 * tail))


def _one_sided(num, se, df, upper):
    """p for H1 'statistic > 0' (upper=True) or '< 0' (upper=False); t = num / se. SD == 0 handled explicitly."""
    if se == 0.0:
        if num == 0.0:
            return None, 1.0
        t = math.inf if num > 0 else -math.inf
        hit = (num > 0) if upper else (num < 0)
        return t, (0.0 if hit else 1.0)
    t = num / se
    p = float(stats.t.sf(t, df)) if upper else float(stats.t.cdf(t, df))
    return t, p


def cell_stats(d, floor):
    n = len(d)
    if n < 2:
        raise JudgeError(f"a defined cell needs at least 2 seeds, got {n}")
    if max(d) == min(d):
        mean, sd = float(d[0]), 0.0
    else:
        mean = math.fsum(d) / n
        sd = math.sqrt(math.fsum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    df = n - 1
    t_plus, p_plus = _one_sided(mean - floor, se, df, upper=True)
    t_minus, p_minus = _one_sided(mean + floor, se, df, upper=False)
    if se == 0.0:
        t_zero = None if mean == 0.0 else (math.inf if mean > 0 else -math.inf)
        p_zero = 1.0 if mean == 0.0 else 0.0
    else:
        t_zero = mean / se
        p_zero = float(min(1.0, 2.0 * stats.t.sf(abs(t_zero), df)))
    t_crit = float(stats.t.ppf(0.95, df))
    ub = mean + t_crit * se
    return {"n": n, "df": df, "mean": mean, "sd": sd, "se": se, "floor": floor, "t_crit_0.95": t_crit,
            "t_plus": t_plus, "t_minus": t_minus, "t_zero": t_zero, "p_plus": p_plus, "p_minus": p_minus,
            "p_zero": p_zero, "upper_bound_95": ub, "upper_bound_below_floor": ub < floor}


# ---------------------------------------------------------------- N1
def _validate_hash(doc, which):
    got = _req(doc, "prereg_sha256", which)
    exp = lbprereg.yaml_sha256()
    if got != exp:
        raise JudgeError(f"{which} prereg_sha256 {got!r} != frozen YAML sha256 {exp}")


def _validate_n1(n1, pre):
    if not isinstance(n1, dict):
        raise JudgeError("N1 input is not a JSON object")
    _validate_hash(n1, "N1 input")
    tasks = list(pre["tasks"].keys())
    caps = [int(c) for c in pre["caps_kib"]]
    control = pre["control_cap"]
    floors = _req(n1, "floors", "N1 input")
    if not isinstance(floors, dict):
        raise JudgeError("floors is not an object")
    for t in floors:
        if t not in tasks:
            raise JudgeError(f"floors: task {t!r} is outside the YAML tasks {tasks}")
    for t in tasks:
        f = _req(floors, t, "floors")
        if not _is_num(f):
            raise JudgeError(f"floors[{t}] = {f!r} is not a finite number")
        if f < 1.0:
            raise JudgeError(f"floors[{t}] = {f} is below the YAML minimum floor 1.0")
    cells = _req(n1, "cells", "N1 input")
    if not isinstance(cells, list):
        raise JudgeError("cells is not a list")
    seen, parsed, seedsets = set(), [], set()
    for k, c in enumerate(cells):
        where = f"cells[{k}]"
        task = _req(c, "task", where)
        if task not in tasks:
            raise JudgeError(f"{where}: task {task!r} is outside the YAML tasks {tasks}")
        cap = _req(c, "cap_kib", where)
        if cap == control:
            finite = False
        elif _is_int(cap) and cap in caps:
            finite = True
        else:
            raise JudgeError(f"{where}: cap_kib {cap!r} is neither a YAML cap {caps} nor {control!r}")
        if (task, cap) in seen:
            raise JudgeError(f"{where}: duplicate cell ({task}, {cap})")
        seen.add((task, cap))
        tc, uc = _req(c, "tied_config", where), _req(c, "untied_config", where)
        for name, v in (("tied_config", tc), ("untied_config", uc)):
            if v is not None and not isinstance(v, str):
                raise JudgeError(f"{where}: {name} must be a string or null")
        seeds = _req(c, "seeds", where)
        ta, ua = _req(c, "tied_test_acc", where), _req(c, "untied_test_acc", where)
        for name, v in (("seeds", seeds), ("tied_test_acc", ta), ("untied_test_acc", ua)):
            if not isinstance(v, list):
                raise JudgeError(f"{where}: {name} is not a list")
        feasible = tc is not None and uc is not None
        rec = {"task": task, "cap_kib": cap, "finite_cap": finite, "tied_config": tc, "untied_config": uc,
               "feasible": feasible, "infeasible_class": None if feasible else
               ("both" if tc is None and uc is None else ("tied" if tc is None else "untied"))}
        if feasible:
            if not all(_is_int(s) for s in seeds):
                raise JudgeError(f"{where}: seeds must be integers")
            if len(set(seeds)) != len(seeds):
                raise JudgeError(f"{where}: duplicate seeds {seeds}")
            if len(ta) != len(seeds) or len(ua) != len(seeds):
                raise JudgeError(f"{where}: seed mismatch: {len(seeds)} seeds, {len(ta)} tied and {len(ua)} untied accuracies")
            for name, arr in (("tied_test_acc", ta), ("untied_test_acc", ua)):
                for x in arr:
                    if not _is_num(x):
                        raise JudgeError(f"{where}: {name} contains a non-finite or non-numeric value {x!r}")
                    if not 0.0 <= x <= 100.0:
                        raise JudgeError(f"{where}: {name} value {x} is outside [0, 100] percent")
            seedsets.add(tuple(sorted(seeds)))
            rec["seeds"] = list(seeds)
            rec["d"] = [float(a) - float(b) for a, b in zip(ta, ua)]
        parsed.append(rec)
    missing = [(t, c) for t in tasks for c in caps if (t, c) not in seen]
    if missing:
        raise JudgeError(f"missing finite-cap cells {missing}; the family is the {len(tasks)} x {len(caps)} grid")
    if len(seedsets) > 1:
        raise JudgeError(f"seed mismatch: cells with a defined Delta carry different seed sets {sorted(seedsets)}")
    floors = {t: float(floors[t]) for t in tasks}
    return parsed, floors, (list(next(iter(seedsets))) if seedsets else None)


def _classify(tested_caps, plus, minus):
    if plus and minus:
        return "multiple crossings"
    if not plus:
        return "no event"
    idx = [tested_caps.index(c) for c in plus]
    contiguous = max(idx) - min(idx) + 1 == len(idx)
    if not contiguous:
        return "multiple crossings"
    if len(plus) == len(tested_caps):
        return "everywhere tested"
    return "band"


def judge_n1(n1, pre):
    alpha = float(pre["inference"]["alpha"])
    parsed, floors, seedset = _validate_n1(n1, pre)
    for rec in parsed:
        if rec["feasible"]:
            rec.update(cell_stats(rec.pop("d"), floors[rec["task"]]))
        rec["in_holm_family"] = rec["feasible"] and rec["finite_cap"]
    fam = [r for r in parsed if r["in_holm_family"]]
    for key in ("plus", "minus", "zero"):
        adj = holm([r["p_" + key] for r in fam])
        for r, a in zip(fam, adj):
            r[f"p_{key}_holm"] = a
    ev = {"plus": "E_plus", "minus": "E_minus", "zero": "zero_sign"}
    for r in parsed:
        for key, name in ev.items():
            r[name] = bool(r["in_holm_family"] and r[f"p_{key}_holm"] < alpha)
            if not r["in_holm_family"]:
                r.setdefault(f"p_{key}_holm", None)
    if not fam:
        n1_1 = {"status": "ERROR", "reason": "no finite-cap cell has a defined Delta"}
    elif any(r["E_plus"] for r in fam):
        n1_1 = {"status": "PASS", "cells": [[r["task"], r["cap_kib"]] for r in fam if r["E_plus"]]}
    elif all(r["upper_bound_below_floor"] for r in fam):
        n1_1 = {"status": "FAIL", "cells": []}
    else:
        n1_1 = {"status": "MISS", "cells_not_below_floor": [[r["task"], r["cap_kib"]] for r in fam
                                                            if not r["upper_bound_below_floor"]]}
    n1_2 = {}
    caps = [int(c) for c in pre["caps_kib"]]
    for t in pre["tasks"]:
        rows = sorted([r for r in fam if r["task"] == t], key=lambda r: r["cap_kib"])
        tested = [r["cap_kib"] for r in rows]
        plus = [r["cap_kib"] for r in rows if r["E_plus"]]
        minus = [r["cap_kib"] for r in rows if r["E_minus"]]
        zero = [r["cap_kib"] for r in rows if r["zero_sign"]]
        n1_2[t] = {"tested_caps": tested, "untested_caps": [c for c in caps if c not in tested],
                   "E_plus_caps": plus, "E_minus_caps": minus, "zero_sign_caps": zero,
                   "class": _classify(tested, plus, minus)}
    conf = sorted(int(s) for s in pre["seeds"]["confirmation"])
    full = sorted(conf + [int(s) for s in pre["seeds"]["top_up"]])
    reasons = []
    if n1_1["status"] == "ERROR":
        disposition = "ERROR"
        reasons.append("N1.1 ERROR: " + n1_1["reason"])
    elif seedset not in (conf, full):
        disposition = "ERROR"
        reasons.append(f"seed set {seedset} is neither the confirmation seeds {conf} nor confirmation + top_up {full}")
    elif n1_1["status"] == "PASS":
        disposition = "GO"
    elif n1_1["status"] == "FAIL":
        disposition = "KILL"
    else:
        disposition = "MISS_TOP_UP_REQUIRED" if seedset == conf else "INCONCLUSIVE"
    n1 = {"alpha": alpha, "floors": floors, "seed_set": seedset, "holm_family_size": len(fam),
          "n_cells_reported": len(parsed), "cells": parsed}
    return n1, n1_1, n1_2, disposition, reasons


# ---------------------------------------------------------------- N2
def _validate_n2(n2, pre):
    if not isinstance(n2, dict):
        raise JudgeError("N2 input is not a JSON object")
    _validate_hash(n2, "N2 input")
    caps = _req(n2, "caps_kib", "N2 input")
    if not isinstance(caps, list) or not all(_is_int(c) for c in caps):
        raise JudgeError("N2 caps_kib must be a list of integers")
    if sorted(caps) != sorted(int(c) for c in pre["caps_kib"]) or len(set(caps)) != len(caps):
        raise JudgeError(f"N2 caps_kib {caps} != YAML caps_kib {pre['caps_kib']}")
    configs = _req(n2, "configs", "N2 input")
    if not isinstance(configs, list) or not configs:
        raise JudgeError("N2 configs must be a non-empty list")
    ids = set()
    for k, c in enumerate(configs):
        where = f"configs[{k}]"
        cid = _req(c, "config_id", where)
        if not isinstance(cid, str) or cid in ids:
            raise JudgeError(f"{where}: config_id {cid!r} is not a unique string")
        ids.add(cid)
        meas = _req(c, "measured_min_peak_bytes", where)
        pred = _req(c, "predicted_min_peak_bytes", where)
        if not _is_int(meas) or meas <= 0:
            raise JudgeError(f"{where}: measured_min_peak_bytes {meas!r} must be a positive integer")
        if not _is_int(pred) or pred < 0:
            raise JudgeError(f"{where}: predicted_min_peak_bytes {pred!r} must be a non-negative integer")
        bl = _req(c, "baselines", where)
        for b in BASELINES:
            v = _req(bl, b, where + ".baselines")
            if not _is_int(v) or v < 0:
                raise JudgeError(f"{where}: baseline {b} = {v!r} must be a non-negative integer")
    return sorted(int(c) for c in caps), configs


def judge_n2(n2, pre):
    alpha = float(pre["inference"]["alpha"])
    caps, configs = _validate_n2(n2, pre)
    rel = [abs(c["predicted_min_peak_bytes"] - c["measured_min_peak_bytes"]) / c["measured_min_peak_bytes"] for c in configs]
    within = sum(1 for r in rel if r <= 0.05)
    share = within / len(configs)
    pairs = []
    for c in configs:
        meas = c["measured_min_peak_bytes"]
        for cap in caps:
            cb = cap * 1024
            if abs(meas - cb) <= 0.2 * cb:
                truth = meas <= cb
                pairs.append({"config_id": c["config_id"], "cap_kib": cap, "measured_fit": truth,
                              "predictor_right": (c["predicted_min_peak_bytes"] <= cb) == truth,
                              "baseline_right": {b: (c["baselines"][b] <= cb) == truth for b in BASELINES}})
    n2_1 = {"n_configs": len(configs), "n_within_5pct": within, "share_within_5pct": share,
            "max_rel_err": max(rel), "n_boundary_pairs": len(pairs)}
    if not pairs:
        n2_1.update(status="ERROR", reason="no boundary pairs")
        return n2_1, {"status": "ERROR", "reason": "no boundary pairs"}
    right = sum(1 for p in pairs if p["predictor_right"])
    acc = right / len(pairs)
    n2_1.update(n_boundary_right=right, boundary_fit_accuracy=acc)
    if share >= 0.95 and acc >= 0.98:
        n2_1["status"] = "PASS"
    elif share < 0.90 or acc < 0.95:
        n2_1["status"] = "FAIL"
    else:
        n2_1["status"] = "MISS"
    comps = []
    for b in BASELINES:
        bb = sum(1 for p in pairs if p["predictor_right"] and not p["baseline_right"][b])
        cc = sum(1 for p in pairs if not p["predictor_right"] and p["baseline_right"][b])
        comps.append({"baseline": b, "b": bb, "c": cc, "p": mcnemar_exact(bb, cc),
                      "baseline_boundary_fit_accuracy": sum(1 for p in pairs if p["baseline_right"][b]) / len(pairs)})
    for comp, a in zip(comps, holm([x["p"] for x in comps])):
        comp["p_holm"] = a
        comp["significant"] = a < alpha
    if all(x["significant"] and x["b"] > x["c"] for x in comps):
        st = "PASS"
    elif any(x["significant"] and x["c"] > x["b"] for x in comps):
        st = "FAIL"
    else:
        st = "MISS"
    return n2_1, {"status": st, "alpha": alpha, "comparisons": comps}


# ---------------------------------------------------------------- driver
def judge(n1, n2=None):
    """Judge parsed inputs. Returns the result dict without file metadata."""
    out = {"gate": GATE, "status": "ERROR", "n1": None, "n1_1": None, "n1_2": None, "disposition": "ERROR",
           "n2_1": "NOT_RUN", "n2_2": "NOT_RUN", "reasons": []}
    try:
        pre = lbprereg.load()
    except Exception as e:  # PreregMismatch or unreadable YAML
        out["reasons"].append(f"preregistration could not be verified: {e}")
        return out
    try:
        out["n1"], out["n1_1"], out["n1_2"], out["disposition"], r = judge_n1(n1, pre)
        out["reasons"].extend(r)
    except JudgeError as e:
        out["reasons"].append(f"N1 input: {e}")
    if n2 is not None:
        try:
            out["n2_1"], out["n2_2"] = judge_n2(n2, pre)
            if out["n2_1"]["status"] == "ERROR":
                out["reasons"].append("N2.1 ERROR: " + out["n2_1"]["reason"])
        except JudgeError as e:
            out["n2_1"] = {"status": "ERROR", "reason": str(e)}
            out["n2_2"] = {"status": "ERROR", "reason": str(e)}
            out["reasons"].append(f"N2 input: {e}")
    n2_err = n2 is not None and (out["n2_1"]["status"] == "ERROR" or out["n2_2"]["status"] == "ERROR")
    if out["reasons"] or out["disposition"] == "ERROR" or n2_err:
        out["status"] = "ERROR"
    else:
        out["status"] = out["disposition"]
    return out


def _strict(x):
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return "+inf" if x > 0 else "-inf"
        return x
    if isinstance(x, dict):
        return {k: _strict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_strict(v) for v in x]
    return x


def _load_json(path, which):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except Exception as e:
        return None, f"{which} file {path} could not be read as JSON: {e}"


def run_files(n1_path, n2_path=None, command=None):
    n1, err1 = _load_json(n1_path, "N1")
    n2, err2 = (None, None) if n2_path is None else _load_json(n2_path, "N2")
    if err1 or err2:
        res = {"gate": GATE, "status": "ERROR", "n1": None, "n1_1": None, "n1_2": None, "disposition": "ERROR",
               "n2_1": "NOT_RUN", "n2_2": "NOT_RUN", "reasons": [e for e in (err1, err2) if e]}
    else:
        res = judge(n1, n2)
    try:
        res["prereg_sha256"] = lbprereg.yaml_sha256()
    except Exception:
        res["prereg_sha256"] = None
    res["judge_sha256"] = _sha_file(__file__)
    res["input_sha256"] = {"n1": _sha_file(n1_path) if Path(n1_path).is_file() else None,
                           "n2": (_sha_file(n2_path) if Path(n2_path).is_file() else None) if n2_path else None}
    res["command"] = command
    res["utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return _strict(res)


def main(argv=None):
    ap = argparse.ArgumentParser(description="loopbudget G6 judge (frozen preregistration v1)")
    ap.add_argument("--n1", required=True)
    ap.add_argument("--n2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    command = shlex.join([sys.executable, *(sys.argv if argv is None else [__file__, *argv])])
    res = run_files(a.n1, a.n2, command)
    Path(a.out).write_text(json.dumps(res, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{GATE}: status {res['status']}" + ("" if not res["reasons"] else f"; reasons: {res['reasons']}"))
    return 0 if res["status"] in VERDICTS else 2


if __name__ == "__main__":
    sys.exit(main())
