"""Peak requested PyTorch CPU tensor bytes of one training update (the preregistration in force, loaded by lb.prereg:
measured_object, memory_knobs, code_freeze_s3). Port of the S0 method (research/s0/instrument_check/instrument_check.py: reconstruct, profiled_step):
live bytes = exact storage bytes alive before the step (parameters, gradients, optimizer state, deduplicated by storage)
plus the signed torch.profiler [memory] events of the step, sorted by start time; peak = baseline + max prefix (>= 0).

The step is lb.train.train_step (the same code as training): zero_grad, per-microbatch gather + forward + backward,
clip, optimizer step. Inputs are torch.randn of the task's shape for one effective batch (64); that buffer is a
loader buffer (excluded); each microbatch gather X[idx_chunk], y[idx_chunk] is allocated inside the step (included).
One warm-up update, then the SECOND update is profiled (steady state). One thread.
Fail-closed: measure_peak raises ProfileClosureError when the trace does not close (baseline + signed events !=
live bytes after the step); measure_model returns the closure error without raising (the planted-tensor test reads it).

CLI (from the project root, src/ on PYTHONPATH):
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m lb.profiler --config-id ID --out DIR [--microbatch M [--checkpointing]]
Without --microbatch: min over knobs. Writes DIR/profile.json, then DIR/DONE.json; refuses (exit 3) if DONE.json exists.
A registry-member configuration is measured only after lb.freeze.verify("harness", "predictor") passes (else exit 2
before any measurement); calibration-grid configurations are not guarded here (their consumers check code_sha256).
In profile.json every measurement carries trace_len and trace_sha256; the full signed trace is kept for the minimum only.
"""
import argparse
import gc
import hashlib
import json
import os
import pathlib
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from lb import freeze, prereg
from lb.models import REVERSIBLE
from lb.train import (RefuseOverwrite, _atomic_json, _sha256, build_from_config, code_hashes, make_optimizer,
                      parse_config_id, set_threads, train_step, utc_now)

_WARMED = False


class ProfileClosureError(RuntimeError):
    """The profiled step's trace does not close (closure_error != 0); the measurement is not used."""


def _warm_profiler():
    """First profiler session in a process (as at S0), so later sessions see only the step's own events."""
    global _WARMED
    if not _WARMED:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False):
            torch.empty(16).add_(1)
        _WARMED = True


def live_bytes(model, opt):
    """Exact bytes of parameters, gradients and optimizer state alive now (deduplicated by storage)."""
    ts = list(model.parameters()) + [q.grad for q in model.parameters() if q.grad is not None]
    for st in opt.state.values():
        ts += [v for v in st.values() if torch.is_tensor(v)]
    seen = {}
    for t in ts:
        seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


def component_bytes(model, opt):
    W = sum(q.untyped_storage().nbytes() for q in model.parameters())
    G = sum(q.grad.untyped_storage().nbytes() for q in model.parameters() if q.grad is not None)
    seen, O = set(), 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v) and v.untyped_storage().data_ptr() not in seen:
                seen.add(v.untyped_storage().data_ptr())
                O += v.untyped_storage().nbytes()
    return {"W_bytes": W, "G_bytes": G, "O_bytes": O}


def reconstruct(prof, baseline):
    """S0 reconstruction; phases may repeat (one ph_forward / ph_backward per microbatch)."""
    evs = list(prof.profiler.kineto_results.events())
    mem = sorted((e.start_ns(), i, e.nbytes()) for i, e in enumerate(evs) if e.name() == "[memory]")
    phases = sorted((e.start_ns(), e.end_ns(), e.name()) for e in evs if e.name().startswith("ph_"))
    cum, peak, peak_t = 0, 0, None
    prefix = []
    for t, _, nb in mem:
        cum += nb
        prefix.append((t, cum))
        if cum > peak:
            peak, peak_t = cum, t
    phase_peak = {}
    for a, b, name in phases:
        before = [c for t, c in prefix if t < a]
        inside = [c for t, c in prefix if a <= t <= b]
        v = baseline + max([before[-1] if before else 0] + inside)
        phase_peak[name] = max(phase_peak.get(name, v), v)
    peak_phase = "start (baseline)"
    if peak_t is not None:
        peak_phase = "between phases"
        for a, b, name in phases:
            if a <= peak_t <= b:
                peak_phase = name
    return {"baseline_bytes": baseline, "peak_bytes": baseline + peak, "peak_above_baseline": peak,
            "peak_phase": peak_phase, "phase_peak": phase_peak, "delta_end": cum,
            "n_alloc": sum(1 for _, _, nb in mem if nb > 0), "n_free": sum(1 for _, _, nb in mem if nb < 0),
            "alloc_trace": [nb for _, _, nb in mem]}


def _inputs(task_cfg, batch, seed=1):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(batch, task_cfg["n_tokens"], task_cfg["token_dim"], generator=g)
    y = torch.randint(0, task_cfg["classes"], (batch,), generator=g)
    return X, y, torch.arange(batch)


def measure_model(model, optimizer, task_cfg, microbatch, batch=64, lr=None):
    """Warm-up update, then profile the second update of an already built model (tests plant hooks on it)."""
    set_threads()
    _warm_profiler()
    p = prereg.load()
    lr = float(p["training"]["lr_grid"][optimizer][0]) if lr is None else lr
    opt = make_optimizer(optimizer, model.parameters(), lr)
    X, y, idx = _inputs(task_cfg, batch)
    model.train()
    # Warm-up update inside its own (discarded) profiler session, as at S0 (both steps profiled): the allocator
    # reporter only emits free events for blocks it saw allocated, so the gradients and optimizer state created here
    # must be allocated under the profiler for the steady-state step's zero_grad frees to appear in its trace.
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False):
        train_step(model, opt, X, y, idx, microbatch, lr)
    gc.collect()
    baseline = live_bytes(model, opt)
    gc.disable()
    try:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
            train_step(model, opt, X, y, idx, microbatch, lr)               # the profiled steady-state update
    finally:
        gc.enable()
    r = reconstruct(prof, baseline)
    r["post_live_bytes"] = live_bytes(model, opt)
    r["closure_error"] = baseline + r["delta_end"] - r["post_live_bytes"]
    r.update(component_bytes(model, opt))
    fb = task_cfg["n_tokens"] * task_cfg["token_dim"] * 4 + 8
    r["input_bytes"] = microbatch * fb
    r["input_note"] = "one microbatch of fp32 inputs and int64 targets (allocated inside the step)"
    r["microbatch"] = microbatch
    r["optimizer"] = optimizer
    r["lr"] = lr
    return r


def measure_peak(config_id, microbatch, checkpointing, seed=0):
    p = prereg.load()
    cfg = parse_config_id(config_id, p)
    if cfg["kind"] != "mixer":
        raise ValueError("the profiler measures mixer configurations only")
    batch = int(p["memory_knobs"]["effective_batch"])
    if microbatch not in p["memory_knobs"]["microbatch"] or batch % microbatch:
        raise ValueError(f"microbatch {microbatch} not in {p['memory_knobs']['microbatch']}")
    task_cfg = prereg.task_cfg(cfg["task"], p)
    set_threads()
    torch.manual_seed(seed)
    model = build_from_config(cfg, task_cfg, checkpointing)
    r = measure_model(model, cfg["optimizer"], task_cfg, microbatch, batch)
    if r["closure_error"] != 0:
        raise ProfileClosureError(f"{config_id} microbatch {microbatch} checkpointing {bool(checkpointing)}: "
                                  f"closure_error {r['closure_error']} bytes")
    r.update({"config_id": config_id, "config": cfg, "checkpointing": bool(checkpointing), "seed": seed,
              "n_params": int(sum(q.numel() for q in model.parameters())), "torch": torch.__version__,
              "threads": {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}})
    return r


def _trace_sha(trace):
    return hashlib.sha256(json.dumps(trace).encode()).hexdigest()


def min_over_knobs(config_id, seed=0):
    """Every (microbatch, checkpointing) knob of the YAML; checkpointing on only for non-reversible structures.
    Minimum = lowest peak; ties broken by checkpointing off, then the larger microbatch (the cheaper knob)."""
    p = prereg.load()
    cfg = parse_config_id(config_id, p)
    ckpts = [False] if cfg["structure"] in REVERSIBLE else [False, True]
    ms = []
    for ck in ckpts:
        for m in p["memory_knobs"]["microbatch"]:
            ms.append(measure_peak(config_id, int(m), ck, seed))
    best = min(ms, key=lambda r: (r["peak_bytes"], r["checkpointing"], -r["microbatch"]))
    return {"config_id": config_id, "measurements": ms, "min_peak_bytes": best["peak_bytes"],
            "argmin_knob": {"microbatch": best["microbatch"], "checkpointing": best["checkpointing"]}}


def _compact(r, keep_trace):
    out = {k: v for k, v in r.items() if k != "alloc_trace"}
    out["trace_len"] = len(r["alloc_trace"])
    out["trace_sha256"] = _trace_sha(r["alloc_trace"])
    if keep_trace:
        out["alloc_trace"] = r["alloc_trace"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="peak requested CPU tensor bytes of one training update")
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--microbatch", type=int, default=None)
    ap.add_argument("--checkpointing", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        print('REFUSING: run with CUDA_VISIBLE_DEVICES="" (CPU only)', file=sys.stderr)
        return 2
    out = pathlib.Path(a.out)
    if (out / "DONE.json").exists():
        print(f"REFUSED: {out / 'DONE.json'} exists", file=sys.stderr)
        return RefuseOverwrite.exit_code
    p = prereg.load()
    cfg = parse_config_id(a.config_id, p)
    code_freeze = None
    if cfg["registry_member"]:
        try:
            code_freeze = freeze.verify("harness", "predictor")
        except freeze.CodeFreezeError as e:
            print(f"ERROR: CodeFreezeError: {e}", file=sys.stderr)
            return 2
    start = utc_now()
    if a.microbatch is None:
        if a.checkpointing:
            ap.error("--checkpointing needs --microbatch (min over knobs covers both settings)")
        r = min_over_knobs(a.config_id)
        best = (r["argmin_knob"]["microbatch"], r["argmin_knob"]["checkpointing"])
        r["measurements"] = [_compact(m, (m["microbatch"], m["checkpointing"]) == best) for m in r["measurements"]]
    else:
        r = _compact(measure_peak(a.config_id, a.microbatch, a.checkpointing), True)
    r.update({"prereg_sha256": prereg.yaml_sha256(), "code_sha256": code_hashes(), "code_freeze": code_freeze,
              "argv": sys.argv,
              "start_utc": start, "end_utc": utc_now()})
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "profile.json", r)
    _atomic_json(out / "DONE.json", {"profile_sha256": _sha256(out / "profile.json"), "end_utc": r["end_utc"]})
    print(json.dumps({"config_id": a.config_id, "peak_bytes": r.get("min_peak_bytes", r.get("peak_bytes"))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
