"""Lifetime-aware byte accounting of the profiled training update (card S3-P; PREREGISTERED in force via lb.prereg:
measured_object, memory_knobs, conjuncts N2.1 and N2.2, code_freeze_s3).

predict(cfg, microbatch, checkpointing) -> predicted peak requested PyTorch CPU tensor bytes of the steady-state
update (lb.profiler's measured object) with a phase breakdown; predict_min(cfg) -> min over the profiler's knobs and
the argmin under lb.profiler.min_over_knobs's tie rule (lowest peak, then checkpointing off, then larger microbatch).
The prediction is arithmetic on shapes: lb.predictor_sim replays the update as signed allocation events from
documented rules (R1-R6 in its docstring; evidence in research/s3/predictor_dev/NOTES.md) and applies the
profiler's reconstruction (baseline + max prefix). No model, autograd, optimizer, profiler, FakeTensor/meta
execution or tracing runs at prediction time; param_shapes(cfg) is analytic. Fitted constants: none (the event
replay reproduces the calibration-grid traces; calibrate records the errors and an empty constant set).

Graded baselines (N2.2), bytes; fit at cap B iff value <= B x 1024:
  B1 param_count_with_state_multiplier: 4 x P x k, k = adam 4, sgd_momentum 3, sgd 2, adafactor 2 (knob-free).
  B2 persistent_weights_gradients_optimizer: exact parameter + gradient + optimizer-state bytes (knob-free).
  B3 static_four_component_sum: B2 + A_static(microbatch, checkpointing), min over the same knobs; A_static = one
     microbatch's inputs and targets plus every non-parameter storage autograd saves in that microbatch's forward,
     once per storage, no lifetimes, no transients (checkpointing: the checkpoint's saved block inputs; reversible:
     the saved stack output). Checkpoint RNG states are not autograd-saved tensors and are not counted.

CLIs (from the project root, src/ on PYTHONPATH):
  python -m lb.predictor calibrate --profiles DIR --out research/s3/v<version>/predictor_calibration.json
  python -m lb.predictor predict --calibration FILE --out research/s3/v<version>/predictions_registry.json
Exit codes: 0 written; 2 ERROR (any violation; nothing written).
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import statistics
import sys

from lb import prereg
from lb import predictor_sim as ps

NAME = "lifetime_aware_accounting"
BASELINES = ("param_count_with_state_multiplier", "persistent_weights_gradients_optimizer",
             "static_four_component_sum")
STATE_SLOTS = {"adam": 4, "sgd_momentum": 3, "sgd": 2, "adafactor": 2}
FITTED_CONSTANTS = {}                  # none: see module docstring
SRC_LB = pathlib.Path(__file__).resolve().parent
BOUNDARY_FRACTION = 0.20
WITHIN = 0.05


class PredictorError(RuntimeError):
    """A calibrate/predict input violates the card's rules; the CLI stops with exit 2 and writes nothing."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------- configurations
def parse_cfg(cfg, p=None):
    """A config_id string "task|structure|dN|TN|optimizer" or a registry dict -> normalized dict."""
    if isinstance(cfg, str):
        parts = cfg.split("|")
        if len(parts) != 5 or not parts[2].startswith("d") or not parts[3].startswith("T"):
            raise ValueError(f"malformed config_id {cfg!r}")
        cfg = {"task": parts[0], "structure": parts[1], "d": int(parts[2][1:]), "T": int(parts[3][1:]),
               "optimizer": parts[4]}
    out = {k: cfg[k] for k in ("task", "structure", "d", "T", "optimizer")}
    out["d"], out["T"] = int(out["d"]), int(out["T"])
    out["config_id"] = f"{out['task']}|{out['structure']}|d{out['d']}|T{out['T']}|{out['optimizer']}"
    if out["optimizer"] not in STATE_SLOTS:
        raise ValueError(f"unknown optimizer {out['optimizer']!r}")
    return out


def dims(task, p=None):
    tc = prereg.task_cfg(task, p)
    return tc["n_tokens"], tc["token_dim"], tc["classes"]


def param_shapes(cfg, p=None):
    """Ordered (name, shape) of the unique parameters, analytic (no model is built)."""
    c = parse_cfg(cfg)
    return ps.param_shapes(*dims(c["task"], p), c["structure"], c["d"], c["T"])


def n_params(cfg, p=None):
    return sum(ps.numel(s) for _, s in param_shapes(cfg, p))


def knobs(cfg, p=None):
    """The profiler's knob order: checkpointing off then on (non-reversible only), microbatches as in the YAML."""
    p = p or prereg.load()
    c = parse_cfg(cfg)
    cks = [False] if c["structure"] in ps.REVERSIBLE else [False, True]
    return [(int(m), ck) for ck in cks for m in p["memory_knobs"]["microbatch"]]


def _argmin(rows, key):
    best = min(rows, key=lambda r: (r[key], r["checkpointing"], -r["microbatch"]))
    return best


# ---------------------------------------------------------------------------------------------- the accounting
def predict(cfg, microbatch, checkpointing, p=None, constants=None):
    """Predicted peak bytes of the profiled update at one knob, with the phase breakdown."""
    p = p or prereg.load()
    c = parse_cfg(cfg)
    batch = int(p["memory_knobs"]["effective_batch"])
    if int(microbatch) not in [int(m) for m in p["memory_knobs"]["microbatch"]]:
        raise ValueError(f"microbatch {microbatch} not in the YAML knobs")
    sim, baseline, comp = ps.simulate_update(dims(c["task"], p), c["structure"], c["d"], c["T"], c["optimizer"],
                                             int(microbatch), bool(checkpointing), batch)
    r = ps.reconstruct(sim, baseline)
    add = sum((constants or FITTED_CONSTANTS).values())
    r["peak_bytes"] += add
    r.update(comp)
    r.update({"config_id": c["config_id"], "microbatch": int(microbatch), "checkpointing": bool(checkpointing),
              "constants_bytes": add})
    return r


def predict_min(cfg, p=None, constants=None):
    p = p or prereg.load()
    rows = [predict(cfg, m, ck, p, constants) for m, ck in knobs(cfg, p)]
    best = _argmin(rows, "peak_bytes")
    return {"config_id": parse_cfg(cfg)["config_id"], "per_knob": rows, "min_peak_bytes": best["peak_bytes"],
            "argmin_knob": {"microbatch": best["microbatch"], "checkpointing": best["checkpointing"]}}


# ---------------------------------------------------------------------------------------------- baselines
def param_count_with_state_multiplier(cfg, p=None):
    c = parse_cfg(cfg)
    return 4 * n_params(c, p) * STATE_SLOTS[c["optimizer"]]


def persistent_weights_gradients_optimizer(cfg, p=None):
    c = parse_cfg(cfg)
    shapes = [s for _, s in param_shapes(c, p)]
    W = sum(ps.numel(s) * ps.F32 for s in shapes)
    return 2 * W + ps.optimizer_state_bytes(c["optimizer"], shapes)


def a_static(cfg, microbatch, checkpointing, p=None):
    c = parse_cfg(cfg)
    return ps.static_saved_bytes(dims(c["task"], p), c["structure"], c["d"], c["T"], int(microbatch),
                                 bool(checkpointing))


def static_four_component_sum(cfg, microbatch, checkpointing, p=None):
    return persistent_weights_gradients_optimizer(cfg, p) + a_static(cfg, microbatch, checkpointing, p)


def static_four_component_sum_min(cfg, p=None):
    p = p or prereg.load()
    rows = [{"microbatch": m, "checkpointing": ck, "bytes": static_four_component_sum(cfg, m, ck, p)}
            for m, ck in knobs(cfg, p)]
    best = _argmin(rows, "bytes")
    return {"per_knob": rows, "min_bytes": best["bytes"],
            "argmin_knob": {"microbatch": best["microbatch"], "checkpointing": best["checkpointing"]}}


def caps_bytes(p=None):
    p = p or prereg.load()
    return [int(k) * 1024 for k in p["caps_kib"]]


def all_predictions(cfg, p=None, constants=None):
    """Every number the predict CLI writes for one configuration (no measurement is read)."""
    p = p or prereg.load()
    c = parse_cfg(cfg)
    acc = predict_min(c, p, constants)
    b3 = static_four_component_sum_min(c, p)
    vals = {NAME: acc["min_peak_bytes"], BASELINES[0]: param_count_with_state_multiplier(c, p),
            BASELINES[1]: persistent_weights_gradients_optimizer(c, p), BASELINES[2]: b3["min_bytes"]}
    caps = p["caps_kib"]
    return {"config_id": c["config_id"], "n_params": n_params(c, p),
            "accounting": {"min_peak_bytes": acc["min_peak_bytes"], "argmin_knob": acc["argmin_knob"],
                           "per_knob": [{k: r[k] for k in ("microbatch", "checkpointing", "peak_bytes", "peak_phase",
                                                           "baseline_bytes", "phase_peak")} for r in acc["per_knob"]]},
            "baselines": {BASELINES[0]: vals[BASELINES[0]], BASELINES[1]: vals[BASELINES[1]],
                          BASELINES[2]: {"min_bytes": b3["min_bytes"], "argmin_knob": b3["argmin_knob"],
                                         "per_knob": b3["per_knob"]}},
            "fit": {name: {str(k): bool(v <= int(k) * 1024) for k in caps} for name, v in vals.items()}}


# ---------------------------------------------------------------------------------------------- calibration
def _rel(pred, meas):
    return (pred - meas) / meas


def _dist(errs):
    if not errs:
        return None
    a = sorted(abs(e) for e in errs)
    q = lambda f: a[min(len(a) - 1, int(round(f * (len(a) - 1))))]   # noqa: E731
    return {"n": len(errs), "frac_within_5pct": sum(1 for e in errs if abs(e) <= WITHIN) / len(errs),
            "abs_median": statistics.median(a), "abs_p90": q(0.9), "abs_p99": q(0.99), "abs_max": a[-1],
            "signed_min": min(errs), "signed_max": max(errs), "exact_fraction": sum(1 for e in errs if e == 0) / len(errs)}


def _harness_check(profiles):
    from lb import freeze
    try:
        freeze.verify("harness")
        rec = freeze.load_record("harness")
    except freeze.CodeFreezeError as e:
        raise PredictorError(f"harness record: {e}")
    want = {pathlib.Path(k).name: h for k, h in rec["files"].items()}
    for path, prof in profiles:
        code = prof.get("code_sha256")
        if not isinstance(code, dict):
            raise PredictorError(f"{path}: no code_sha256")
        got = {k: v for k, v in code.items() if not k.startswith("predictor")}
        if got != want:
            diff = sorted(set(got.items()) ^ set(want.items()))
            raise PredictorError(f"{path}: code_sha256 differs from the harness record: {diff[:4]}")
    return freeze._sha256(freeze.record_path("harness"))


def load_profiles(root, p):
    from lb import registry
    root = pathlib.Path(root)
    files = sorted(root.rglob("profile.json"))
    if not files:
        raise PredictorError(f"no profile.json under {root}")
    yaml_sha = prereg.yaml_sha256()
    profiles = []
    for f in files:
        done = f.parent / "DONE.json"
        if not done.is_file() or json.loads(done.read_text()).get("profile_sha256") != _sha256(f):
            raise PredictorError(f"{f}: no DONE.json with this profile's sha256 (incomplete profiler output)")
        prof = json.loads(f.read_text())
        if prof.get("prereg_sha256") != yaml_sha:
            raise PredictorError(f"{f}: prereg_sha256 {prof.get('prereg_sha256')} != in force {yaml_sha}")
        if "measurements" not in prof or "min_peak_bytes" not in prof:
            raise PredictorError(f"{f}: not a min-over-knobs profile")
        profiles.append((f, prof))
    cal, reg = set(), set()
    for task in p["tasks"]:
        cal |= {c["config_id"] for c in registry.calibration_grid(task)}
        reg |= {c["config_id"] for c in registry.enumerate(task)}
    seen = {}
    for f, prof in profiles:
        cid = prof.get("config_id")
        if cid in reg:
            raise PredictorError(f"{f}: {cid} is a REGISTRY configuration; calibration takes the calibration grid only")
        if cid not in cal:
            raise PredictorError(f"{f}: {cid} is not a calibration-grid configuration")
        if cid in seen:
            raise PredictorError(f"{f}: duplicate profile for {cid} (also {seen[cid]})")
        seen[cid] = f
        want = set(knobs(cid, p))
        got = [(int(m["microbatch"]), bool(m["checkpointing"])) for m in prof["measurements"]]
        if len(got) != len(set(got)) or set(got) != want:
            raise PredictorError(f"{f}: knobs {sorted(got)} != {sorted(want)}")
        best = _argmin(prof["measurements"], "peak_bytes")
        if best["peak_bytes"] != prof["min_peak_bytes"]:
            raise PredictorError(f"{f}: min_peak_bytes inconsistent with its measurements")
    missing = sorted(cal - set(seen))
    if missing:
        raise PredictorError(f"{len(missing)} calibration configurations missing, e.g. {missing[:3]}")
    return profiles


def calibrate(profiles_dir, out):
    p = prereg.load()
    out = pathlib.Path(out)
    if out.exists():
        raise PredictorError(f"{out} exists")
    profiles = load_profiles(profiles_dir, p)
    harness_sha = _harness_check(profiles)
    caps = caps_bytes(p)
    per_knob_err, min_err, rows = [], [], []
    by_group = {}
    pairs = {NAME: [], **{b: [] for b in BASELINES}}
    argmin_agree = 0
    for f, prof in profiles:
        cid = prof["config_id"]
        c = parse_cfg(cid)
        pm = predict_min(c, p)
        pk = {(r["microbatch"], r["checkpointing"]): r["peak_bytes"] for r in pm["per_knob"]}
        knob_rows = []
        for m in prof["measurements"]:
            key = (int(m["microbatch"]), bool(m["checkpointing"]))
            e = _rel(pk[key], m["peak_bytes"])
            per_knob_err.append(e)
            by_group.setdefault(f"knob m{key[0]} ckpt{int(key[1])}", []).append(e)
            knob_rows.append({"microbatch": key[0], "checkpointing": key[1], "measured": m["peak_bytes"],
                              "predicted": pk[key], "rel_error": e})
        e = _rel(pm["min_peak_bytes"], prof["min_peak_bytes"])
        min_err.append(e)
        by_group.setdefault(f"min optimizer {c['optimizer']}", []).append(e)
        by_group.setdefault(f"min structure {c['structure']}", []).append(e)
        by_group.setdefault(f"min task {c['task']}", []).append(e)
        argmin_agree += pm["argmin_knob"] == prof["argmin_knob"]
        b3 = static_four_component_sum_min(c, p)["min_bytes"]
        vals = {NAME: pm["min_peak_bytes"], BASELINES[0]: param_count_with_state_multiplier(c, p),
                BASELINES[1]: persistent_weights_gradients_optimizer(c, p), BASELINES[2]: b3}
        for B in caps:
            if abs(prof["min_peak_bytes"] - B) <= BOUNDARY_FRACTION * B:
                truth = prof["min_peak_bytes"] <= B
                for name, v in vals.items():
                    pairs[name].append((cid, B, truth, v <= B))
        rows.append({"config_id": cid, "measured_min": prof["min_peak_bytes"], "predicted_min": pm["min_peak_bytes"],
                     "rel_error_min": e, "measured_argmin": prof["argmin_knob"], "predicted_argmin": pm["argmin_knob"],
                     "per_knob": knob_rows, "baselines": {b: vals[b] for b in BASELINES}})
    fit = {name: {"n_boundary_pairs": len(v),
                  "accuracy": (sum(1 for _, _, t, q in v if t == q) / len(v)) if v else None,
                  "errors": [{"config_id": cid, "cap_bytes": B, "measured_fits": t, "predicted_fits": q}
                             for cid, B, t, q in v if t != q]} for name, v in pairs.items()}
    rec = {"name": "predictor_calibration", "written_utc": utc_now(), "prereg_sha256": prereg.yaml_sha256(),
           "harness_record_sha256": harness_sha,
           "predictor_code_sha256": {f.name: _sha256(f) for f in sorted(SRC_LB.glob("predictor*.py"))},
           "profiles_dir": str(pathlib.Path(profiles_dir).resolve()), "n_profiles": len(profiles),
           "constants": dict(FITTED_CONSTANTS), "n_fitted_constants": len(FITTED_CONSTANTS),
           "boundary_rule": "|measured min-over-knobs peak - B| <= 0.20 B, B = cap_kib x 1024",
           "errors": {"per_knob": _dist(per_knob_err), "min_over_knobs": _dist(min_err),
                      "argmin_agreement": argmin_agree / len(profiles),
                      "groups": {k: _dist(v) for k, v in sorted(by_group.items())}},
           "fit_accuracy_calibration_boundary_pairs": fit, "configs": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1, allow_nan=False)
    return rec


# ---------------------------------------------------------------------------------------------- registry predictions
def _predictor_check(calibration):
    from lb import freeze
    try:
        records = freeze.verify("harness", "predictor")
        rec = freeze.load_record("predictor")
    except freeze.CodeFreezeError as e:
        raise PredictorError(f"code-freeze records: {e}")
    cal = pathlib.Path(calibration).resolve()
    key = freeze._key(cal)
    if key not in rec["files"]:
        raise PredictorError(f"{cal} is not recorded in the predictor code-freeze record")
    return records


def predict_registry(calibration, out):
    out = pathlib.Path(out)
    if out.exists():
        raise PredictorError(f"{out} exists")
    records = _predictor_check(calibration)
    p = prereg.load()
    cal = json.loads(pathlib.Path(calibration).read_text())
    if cal.get("prereg_sha256") != prereg.yaml_sha256():
        raise PredictorError("calibration prereg_sha256 differs from the YAML in force")
    code = {f.name: _sha256(f) for f in sorted(SRC_LB.glob("predictor*.py"))}
    if cal.get("predictor_code_sha256") != code:
        raise PredictorError("calibration was written by other predictor code")
    constants = cal.get("constants", {})
    from lb import registry
    configs = []
    for task in p["tasks"]:
        for c in registry.enumerate(task):
            r = all_predictions(c, p, constants)
            if r["n_params"] != c["n_params"]:
                raise PredictorError(f"{c['config_id']}: analytic n_params {r['n_params']} != registry {c['n_params']}")
            configs.append(r)
    if len(configs) != 3 * int(p["registry"]["configurations_per_task"]):
        raise PredictorError(f"{len(configs)} registry configurations")
    rec = {"name": "predictions_registry", "written_utc": utc_now(), "prereg_sha256": prereg.yaml_sha256(),
           "code_freeze": records, "calibration_sha256": _sha256(calibration), "constants": constants,
           "caps_kib": p["caps_kib"], "fit_rule": "predicted <= cap_kib x 1024", "n_configs": len(configs),
           "methods": [NAME] + list(BASELINES), "configs": configs}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as fh:
        json.dump(rec, fh, indent=1, allow_nan=False)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="lifetime-aware byte accounting: calibrate or predict")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("calibrate")
    a1.add_argument("--profiles", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("predict")
    a2.add_argument("--calibration", required=True)
    a2.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "calibrate":
            rec = calibrate(a.profiles, a.out)
            print(json.dumps({"written": a.out, "n_profiles": rec["n_profiles"],
                              "min_over_knobs": rec["errors"]["min_over_knobs"]}, indent=1))
        else:
            rec = predict_registry(a.calibration, a.out)
            print(json.dumps({"written": a.out, "n_configs": rec["n_configs"]}))
    except (PredictorError, prereg.PreregMismatch, FileExistsError, ValueError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
