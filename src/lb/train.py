"""Training loop of the frozen recipe (the preregistration in force, loaded by lb.prereg: training, memory_knobs, seeds,
failures, code_freeze_s3).

Run from the project root with src/ on PYTHONPATH (the launcher does this):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    python -m lb.train --config-id ID --lr LR --seed S --mode MODE --out DIR [--microbatch M] [--checkpointing] [--max-steps N]

config_id: a registry or calibration-grid ID "task|structure|dN|TN|optimizer" (also "task|pointer_shared|dN|TN|opt",
the sanity construct, wherever the matching shared ID is valid), or a classical bar "task|linear" (linear softmax on
flattened tokens, Adam) or "task|majority" (most frequent training class, no training).
Modes: dev, lrcal, floor, select, confirm, smoke. Only confirm reaches the test split, and only through
_confirm_only_test_split(), the single call site of load_task(..., "test", allow_test=True) in this package's harness.
Guards (check_guards, before any data or model work; code_freeze_s3): a registry-member configuration runs only in
  lrcal  (a training.reference_configs configuration, any registry optimizer, seed in seeds.selection, lr in
         lr_grid[optimizer], microbatch 64, no checkpointing),
  floor  (the untied reference with adam, seed in FLOOR_SEEDS = 101..105, lr in lr_grid[adam], microbatch 64, no
         checkpointing),
  select (seed in seeds.selection) and confirm (seed in seeds.confirmation + seeds.top_up);
dev and smoke refuse it. lrcal and floor verify the harness record; select and confirm verify the harness and
predictor records, for every configuration (lb.freeze.verify). The verified record hashes go to result.json
"code_freeze" (None when not verified).
Schedule (pinned, per optimizer step k, K planned steps, W = steps of the first epoch):
  k < W: lr * k / W (linear warm-up from 0); k >= W: lr * 0.5 * (1 + cos(pi * (k - W) / (K - W))) (cosine to 0 at K).
max_steps truncates training without compressing the schedule (smoke and tests only).
Exit codes: 0 done; 2 ERROR (guard violation or code-freeze mismatch, raised before any data is read); 3 refused
(DONE.json exists); other non-zero = execution error (the launcher reruns once).
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_
from torch.profiler import record_function

from lb import data as lbdata
from lb import freeze, prereg, registry
from lb.models import REVERSIBLE, STRUCTURES, build_model, build_pointer_shared

SRC_LB = pathlib.Path(__file__).resolve().parent
MODES = ("dev", "lrcal", "floor", "select", "confirm", "smoke")
FLOOR_SEEDS = (101, 102, 103, 104, 105)
BARS = ("linear", "majority")
EVAL_BATCH = 512


class RefuseOverwrite(RuntimeError):
    exit_code = 3


class GuardViolation(RuntimeError):
    """A configuration, mode, seed, lr or knob outside what code_freeze_s3 and the S3/S4 protocol allow."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def set_threads():
    """One intra-op and one inter-op thread; deterministic algorithms. Raises if one thread cannot be guaranteed."""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    torch.use_deterministic_algorithms(True)
    t = (torch.get_num_threads(), torch.get_num_interop_threads())
    if t != (1, 1):
        raise RuntimeError(f"threads (intra, inter) = {t}, expected (1, 1)")
    return {"intra_op": t[0], "inter_op": t[1]}


# ------------------------------------------------------------------------------------------------------------------
# Configuration IDs
# ------------------------------------------------------------------------------------------------------------------
def parse_config_id(config_id, p=None):
    p = p or prereg.load()
    parts = config_id.split("|")
    task = parts[0]
    if task not in p["tasks"]:
        raise ValueError(f"unknown task in {config_id!r}")
    if len(parts) == 2 and parts[1] in BARS:
        return {"config_id": config_id, "kind": "bar", "task": task, "structure": parts[1], "d": None, "T": None,
                "optimizer": "adam" if parts[1] == "linear" else None, "tied": None,
                "registry_member": False, "calibration_member": False}
    if len(parts) != 5 or not parts[2].startswith("d") or not parts[3].startswith("T"):
        raise ValueError(f"malformed config_id {config_id!r}")
    s, d, T, opt = parts[1], int(parts[2][1:]), int(parts[3][1:]), parts[4]
    if s not in STRUCTURES + ("pointer_shared",):
        raise ValueError(f"unknown structure in {config_id!r}")
    check = config_id if s != "pointer_shared" else "|".join([task, "shared", parts[2], parts[3], opt])
    reg_ids = {c["config_id"] for c in registry.enumerate(task)}
    cal_ids = {c["config_id"] for c in registry.calibration_grid(task)}
    if check not in reg_ids and check not in cal_ids:
        raise ValueError(f"{config_id!r} is neither a registry nor a calibration-grid configuration")
    return {"config_id": config_id, "kind": "mixer", "task": task, "structure": s, "d": d, "T": T, "optimizer": opt,
            "tied": s in ("shared", "two_groups", "reversible_shared", "pointer_shared"),
            "registry_member": check in reg_ids, "calibration_member": check in cal_ids}


def reference_configs(p=None):
    """training.reference_configs parsed: {"untied": ("untied", d, T), "tied": ("shared", d, T)}."""
    p = p or prereg.load()
    out = {}
    for cls, text in p["training"]["reference_configs"].items():
        m = re.fullmatch(r"\s*(\w+), T = (\d+), d = (\d+)\s*", text)
        if not m:
            raise prereg.PreregMismatch(f"reference_configs.{cls}: cannot parse {text!r}")
        out[cls] = (m.group(1), int(m.group(3)), int(m.group(2)))
    if set(out) != {"untied", "tied"}:
        raise prereg.PreregMismatch(f"reference_configs classes {sorted(out)} != ['tied', 'untied']")
    return out


def reference_config_id(task, cls, optimizer, p=None):
    s, d, T = reference_configs(p)[cls]
    return f"{task}|{s}|d{d}|T{T}|{optimizer}"


def _lr_in_grid(lr, grid):
    return any(float(lr) == float(v) for v in grid)


def check_guards(cfg, mode, lr, seed, microbatch, checkpointing, p):
    """Raise GuardViolation (mode rules) or freeze.CodeFreezeError (records) before any data or model work.
    Returns the verified record hashes {name: sha256} or None (modes that verify nothing)."""
    if mode not in MODES:
        raise GuardViolation(f"mode {mode!r} not in {MODES}")
    seeds = p["seeds"]
    if cfg["registry_member"]:
        refs = reference_configs(p)
        key = (cfg["structure"], cfg["d"], cfg["T"])
        where = f"{cfg['config_id']} (registry member) in mode {mode!r}"
        if mode in ("dev", "smoke"):
            raise GuardViolation(f"{where}: registry configurations run only in lrcal, floor, select or confirm")
        if mode == "lrcal":
            if key not in refs.values():
                raise GuardViolation(f"{where}: lrcal trains only the reference configurations {sorted(refs.values())}")
            if cfg["optimizer"] not in p["registry"]["optimizers"]:
                raise GuardViolation(f"{where}: optimizer {cfg['optimizer']!r} not a registry optimizer")
            if seed not in seeds["selection"]:
                raise GuardViolation(f"{where}: seed {seed} not in seeds.selection {seeds['selection']}")
            if not _lr_in_grid(lr, p["training"]["lr_grid"][cfg["optimizer"]]):
                raise GuardViolation(f"{where}: lr {lr} not in lr_grid[{cfg['optimizer']}]")
        elif mode == "floor":
            if key != refs["untied"] or cfg["optimizer"] != "adam":
                raise GuardViolation(f"{where}: floor trains only the untied reference {refs['untied']} with adam")
            if seed not in FLOOR_SEEDS:
                raise GuardViolation(f"{where}: seed {seed} not in FLOOR_SEEDS {FLOOR_SEEDS}")
            if not _lr_in_grid(lr, p["training"]["lr_grid"]["adam"]):
                raise GuardViolation(f"{where}: lr {lr} not in lr_grid[adam]")
        elif mode == "select":
            if seed not in seeds["selection"]:
                raise GuardViolation(f"{where}: seed {seed} not in seeds.selection {seeds['selection']}")
        elif mode == "confirm":
            allowed = list(seeds["confirmation"]) + list(seeds["top_up"])
            if seed not in allowed:
                raise GuardViolation(f"{where}: seed {seed} not in seeds.confirmation + seeds.top_up {allowed}")
        if mode in ("lrcal", "floor") and (microbatch != 64 or checkpointing):
            raise GuardViolation(f"{where}: microbatch 64 without checkpointing only")
    if mode in ("lrcal", "floor"):
        return freeze.verify("harness")
    if mode in ("select", "confirm"):
        return freeze.verify("harness", "predictor")
    return None


class LinearBar(nn.Module):
    """Classical bar: linear softmax on flattened tokens."""

    def __init__(self, task_cfg):
        super().__init__()
        self.fc = nn.Linear(task_cfg["n_tokens"] * task_cfg["token_dim"], task_cfg["classes"])
        self.n_params = sum(q.numel() for q in self.parameters())

    def forward(self, x):
        return self.fc(x.flatten(1))


def build_from_config(cfg, task_cfg, checkpointing=False):
    """Model for a parsed mixer or linear-bar config (call torch.manual_seed first)."""
    if cfg["kind"] == "bar":
        if checkpointing:
            raise ValueError("checkpointing does not apply to a classical bar")
        if cfg["structure"] != "linear":
            raise ValueError("the majority bar has no model")
        return LinearBar(task_cfg)
    if checkpointing and cfg["structure"] in REVERSIBLE:
        raise ValueError("checkpointing is not combined with reversible structures (memory_knobs.checkpointing_note)")
    if cfg["structure"] == "pointer_shared":
        m = build_pointer_shared(task_cfg, cfg["d"], cfg["T"])
    else:
        m = build_model(task_cfg, cfg["structure"], cfg["d"], cfg["T"])
    return m.set_checkpointing(checkpointing)


def make_optimizer(name, params, lr):
    """YAML registry.optimizer_settings; foreach=False, fused off, weight decay 0."""
    params = list(params)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False, fused=False)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=False, weight_decay=0, foreach=False, fused=False)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=0, foreach=False, fused=False)
    if name == "adafactor":
        return torch.optim.Adafactor(params, lr=lr, foreach=False)
    raise ValueError(f"unknown optimizer {name!r}")


# ------------------------------------------------------------------------------------------------------------------
# Schedule, one optimizer step, evaluation
# ------------------------------------------------------------------------------------------------------------------
def lr_at(k, lr, warmup_steps, total_steps):
    if k < warmup_steps:
        return lr * k / warmup_steps
    if total_steps <= warmup_steps:
        return lr
    return lr * 0.5 * (1.0 + math.cos(math.pi * (k - warmup_steps) / (total_steps - warmup_steps)))


def _chunks(idx, microbatch):
    return idx.split(microbatch)


def train_step(model, opt, X, y, idx, microbatch, lr, max_norm=1.0):
    """One optimizer update on the effective batch X[idx]: gradients accumulated over microbatch chunks with
    loss = sum of per-example cross-entropy / b, so the summed gradient is the effective-batch mean gradient.
    Each chunk is gathered from X when used (only one microbatch of inputs is alive). Returns the batch mean loss."""
    b = idx.numel()
    for g in opt.param_groups:
        g["lr"] = lr
    with record_function("ph_zero_grad"):
        opt.zero_grad(set_to_none=True)
    total = 0.0
    for ic in _chunks(idx, microbatch):
        with record_function("ph_forward"):
            xc = X[ic]
            yc = y[ic]
            loss = F_.cross_entropy(model(xc), yc, reduction="sum") / b
        with record_function("ph_backward"):
            loss.backward()
            total += loss.item()
            del xc, yc, loss
    with record_function("ph_clip"):
        torch.nn.utils.clip_grad_norm_([q for g in opt.param_groups for q in g["params"]], max_norm, foreach=False)
    with record_function("ph_step"):
        opt.step()
    return total


@torch.no_grad()
def accuracy(model, X, y):
    """Percent correct; an example whose logits contain a non-finite value counts as wrong."""
    was = model.training
    model.eval()
    correct, nonfinite = 0, 0
    for s in range(0, X.shape[0], EVAL_BATCH):
        logits = model(X[s:s + EVAL_BATCH])
        fin = torch.isfinite(logits).all(dim=1)
        correct += int(((logits.argmax(dim=1) == y[s:s + EVAL_BATCH]) & fin).sum())
        nonfinite += int((~fin).sum())
    model.train(was)
    return 100.0 * correct / X.shape[0], nonfinite


class _GlobalOnlyTracker:
    """Stands in for FlopCounterMode's ModuleTracker, whose hooks fail inside the reversible stack's no-grad forward;
    every FLOP is attributed to "Global" (the only total used here)."""
    parents = {"Global"}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def flops_per_example(model, X, y, n=64):
    """Forward + backward FLOPs per example (torch.utils.flop_counter registry and dispatch) on one batch, on a deep
    copy of the model."""
    from torch.utils.flop_counter import FlopCounterMode
    m = copy.deepcopy(model)
    xb, yb = X[:n], y[:n]
    fc = FlopCounterMode(display=False)
    fc.mod_tracker = _GlobalOnlyTracker()
    with fc:
        F_.cross_entropy(m(xb), yb).backward()
    return fc.get_total_flops() / xb.shape[0]


# ------------------------------------------------------------------------------------------------------------------
# Data access: development splits and the confirm-only test split
# ------------------------------------------------------------------------------------------------------------------
def _load_dev(task):
    Xtr, ytr = lbdata.load_task(task, "train")
    Xva, yva = lbdata.load_task(task, "val")
    return Xtr, ytr, Xva, yva


def _confirm_only_test_split(task, mode):
    """The only route to the test split. Raises unless mode == "confirm"."""
    if mode != "confirm":
        raise lbdata.TestSplitAccess(f"mode {mode!r} may not read the test split")
    return lbdata.load_task(task, "test", allow_test=True)


# ------------------------------------------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------------------------------------------
def _num(x):
    return float(x) if x is not None and math.isfinite(x) else None


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def code_hashes():
    return {f.name: _sha256(f) for f in sorted(SRC_LB.glob("*.py"))}


def _atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fit(model, opt, X, y, epochs, seed, microbatch, lr, max_steps=None, eval_fn=None, batch=64):
    """Train with the frozen schedule; returns the per-epoch record. eval_fn(model) -> (val_acc, nonfinite)."""
    N = X.shape[0]
    spe = math.ceil(N / batch)
    total = epochs * spe
    g = torch.Generator().manual_seed(seed)
    step, first_nonfinite, completed = 0, None, 0
    losses, vals, truncated = [], [], False
    model.train()
    for ep in range(epochs):
        if max_steps is not None and step >= max_steps:
            truncated = True
            break
        perm = torch.randperm(N, generator=g)
        loss_sum, seen = 0.0, 0
        for s in range(0, N, batch):
            if max_steps is not None and step >= max_steps:
                truncated = True
                break
            idx = perm[s:s + batch]
            lval = train_step(model, opt, X, y, idx, microbatch, lr_at(step, lr, spe, total))
            if not math.isfinite(lval) and first_nonfinite is None:
                first_nonfinite = step
            loss_sum += lval * idx.numel()
            seen += idx.numel()
            step += 1
        completed += seen == N
        losses.append(_num(loss_sum / seen) if seen else None)
        vals.append(eval_fn(model)[0] if eval_fn else None)
        if truncated:
            break
    return {"steps": step, "steps_planned": total, "steps_per_epoch": spe, "warmup_steps": spe,
            "epochs_completed": int(completed), "truncated": truncated,
            "train_loss_per_epoch": losses, "val_acc_per_epoch": vals, "first_nonfinite_step": first_nonfinite}


def train_run(config_id, lr, seed, mode, out_dir, microbatch=64, checkpointing=False, max_steps=None):
    out = pathlib.Path(out_dir)
    if (out / "DONE.json").exists():
        raise RefuseOverwrite(f"{out / 'DONE.json'} exists; refusing to overwrite")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not in {MODES}")
    start_utc, t0 = utc_now(), time.perf_counter()
    threads = set_threads()
    p = prereg.load()
    batch = int(p["memory_knobs"]["effective_batch"])
    if microbatch not in p["memory_knobs"]["microbatch"] or batch % microbatch:
        raise ValueError(f"microbatch {microbatch} not in {p['memory_knobs']['microbatch']}")
    cfg = parse_config_id(config_id, p)
    code_freeze = check_guards(cfg, mode, lr, seed, microbatch, checkpointing, p)   # before any data or model work
    task = cfg["task"]
    task_cfg = prereg.task_cfg(task, p)
    epochs = int(p["training"]["epochs"][task])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    Xtr, ytr, Xva, yva = _load_dev(task)
    test_acc, test_nonfinite = None, None
    if cfg["structure"] == "majority":
        cls = int(torch.bincount(ytr, minlength=task_cfg["classes"]).argmax())
        val_acc = 100.0 * float((yva == cls).sum()) / yva.numel()
        rec = {"steps": 0, "steps_planned": 0, "steps_per_epoch": 0, "warmup_steps": 0, "epochs_completed": 0,
               "truncated": False, "train_loss_per_epoch": [], "val_acc_per_epoch": [], "first_nonfinite_step": None}
        val_nonfinite, n_params, flops = 0, 0, 0.0
        if mode == "confirm":
            Xte, yte = _confirm_only_test_split(task, mode)
            test_acc, test_nonfinite = 100.0 * float((yte == cls).sum()) / yte.numel(), 0
        extra = {"majority_class": cls}
    else:
        model = build_from_config(cfg, task_cfg, checkpointing)
        n_params = int(sum(q.numel() for q in model.parameters()))
        opt = make_optimizer(cfg["optimizer"], model.parameters(), lr)
        flops = flops_per_example(model, Xtr, ytr)
        rec = fit(model, opt, Xtr, ytr, epochs, seed, microbatch, lr, max_steps,
                  eval_fn=lambda m: accuracy(m, Xva, yva), batch=batch)
        val_acc, val_nonfinite = accuracy(model, Xva, yva)
        if mode == "confirm":
            Xte, yte = _confirm_only_test_split(task, mode)
            test_acc, test_nonfinite = accuracy(model, Xte, yte)
        extra = {}
    status = "diverged" if rec["first_nonfinite_step"] is not None else "ok"
    result = {
        "config": cfg, "config_id": config_id, "task": task, "lr": lr, "seed": seed, "mode": mode,
        "microbatch": microbatch, "checkpointing": bool(checkpointing), "effective_batch": batch,
        "epochs": epochs, "max_steps": max_steps, **rec,
        "final_val_acc": val_acc, "final_val_nonfinite_examples": val_nonfinite,
        "test_acc": test_acc, "test_nonfinite_examples": test_nonfinite, "status": status,
        "n_params": n_params, "train_flops_per_example": flops,
        "flops_note": "forward + backward on one batch of 64 training examples, torch.utils.flop_counter; optimizer step excluded",
        "schedule": "k < W: lr*k/W; else lr*0.5*(1+cos(pi*(k-W)/(K-W))); W = steps per epoch, K = epochs*W",
        "prereg_sha256": prereg.yaml_sha256(), "code_sha256": code_hashes(), "code_freeze": code_freeze,
        "torch": torch.__version__,
        "threads": threads, "argv": sys.argv, "start_utc": start_utc, "end_utc": utc_now(),
        "wall_seconds": time.perf_counter() - t0, **extra,
    }
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json(out / "result.json", result)
    _atomic_json(out / "DONE.json", {"result_sha256": _sha256(out / "result.json"), "status": status,
                                     "end_utc": result["end_utc"]})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--microbatch", type=int, default=64)
    ap.add_argument("--checkpointing", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        print('REFUSING: run with CUDA_VISIBLE_DEVICES="" (CPU only)', file=sys.stderr)
        return 2
    try:
        r = train_run(a.config_id, a.lr, a.seed, a.mode, a.out, a.microbatch, a.checkpointing, a.max_steps)
    except RefuseOverwrite as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return RefuseOverwrite.exit_code
    except (GuardViolation, freeze.CodeFreezeError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({k: r[k] for k in ("config_id", "mode", "status", "steps", "final_val_acc", "test_acc",
                                        "wall_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
