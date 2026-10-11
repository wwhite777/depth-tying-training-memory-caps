"""S3 tests for src/lb/train.py and src/lb/launch.py (cards S3-T and S3-T2).

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_train.py -v
Data: synthetic tensors of each task's shape through an instrumented stand-in for lb.data.load_task (records every
(task, split, allow_test) request); the real test split is never read. The launcher test runs subprocesses on the real
HAR train/val splits (skipped if research/s3/data_manifest.json lacks HAR) and never requests the test split.
Configurations: calibration grid (widths 40, 56) and the classical bars only; no registry configuration is trained.
Registry IDs appear only in guard tests, which raise before any data or model work.
Microbatch-accumulation equivalence follows the preregistration in force (loaded by lb.prereg)
equivalence_tolerances:
  relative_error = per tensor max|a - b| / max|reference|, then the max over tensors; in GRADIENT comparisons a tensor
  whose exact gradient is identically zero by construction (tok2.bias in a non-reversible structure) is measured
  against the model's largest per-tensor max|reference|;
  gates in float64 (model and inputs cast to double, through the unchanged, dtype-agnostic train.train_step):
  accumulated gradient (the gradient the optimizer step receives) at microbatch 8 and 1 vs 64 for every optimizer's
  state; parameters after three updates for every optimizer (adafactor: zero-by-construction tensors excluded and
  reported); the same comparisons in fp32 (also against a float64 reference from the same fp32 state, and the
  reordering control) are reported, not gated. Code-freeze records (lb.freeze) are written only into temporary,
  patched RECORD_DIRs.
Every check has an in-file negative control showing that it can fail.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from lb import data as lbdata  # noqa: E402
from lb import freeze, launch, prereg, train  # noqa: E402

P = prereg.load()
TOL = P["equivalence_tolerances"]
OPTS = list(P["registry"]["optimizers"])
REPORT = {}
PY = sys.executable
NONREVERSIBLE = ("shared", "two_groups", "untied", "wide_single", "pointer_shared")
ZERO_GRAD_RATIO = 1e-6          # analysis check: zero-by-construction gradients <= 1e-6 x the model's largest max|g|


class FakeData:
    """Instrumented load_task: synthetic tensors of the task's shape; records every request."""

    N = {"train": 160, "val": 70, "test": 50}

    def __init__(self):
        self.calls = []

    def __call__(self, task, split, *, allow_test=False, **kw):
        self.calls.append((task, split, allow_test))
        if split == "test" and allow_test is not True:
            raise lbdata.TestSplitAccess("test without allow_test")
        c = prereg.task_cfg(task, P)
        g = torch.Generator().manual_seed({"train": 11, "val": 12, "test": 13}[split])
        n = self.N[split]
        return (torch.randn(n, c["n_tokens"], c["token_dim"], generator=g),
                torch.randint(0, c["classes"], (n,), generator=g))

    def test_requests(self):
        return [c for c in self.calls if c[1] == "test"]


@contextlib.contextmanager
def temp_records(*names):
    """Code-freeze records of the current tree in a temporary, patched RECORD_DIR (never the real research/s3/).
    The predictor record holds a temporary stand-in calibration artifact (no src/lb/predictor*.py exists yet)."""
    with tempfile.TemporaryDirectory() as d, mock.patch.object(freeze, "RECORD_DIR", Path(d) / "records"):
        art = Path(d) / "calibration_artifact.json"
        art.write_text("{}\n")
        for n in names:
            freeze.write_record(n, (art,) if n == "predictor" else ())
        yield Path(d)


# ------------------------------------------------------------------------------------------------------------------
# Relative error (equivalence_tolerances.relative_error) and the accumulation fixtures
# ------------------------------------------------------------------------------------------------------------------
F64 = torch.float64


def zero_by_construction(structure, name):
    return structure in NONREVERSIBLE and name.endswith("tok2.bias")


def rel_error(ref, other, structure=None, gradient=False, exclude=()):
    """equivalence_tolerances.relative_error. ref, other: {name: tensor}; returns (max over tensors, per tensor).
    gradient=True: zero-by-construction tensors are measured against the largest per-tensor max|ref| of the model.
    exclude: tensor names left out of the comparison (Adafactor's zero-by-construction parameters, v3)."""
    top = max(float(t.abs().max()) for t in ref.values())
    per = {}
    for n, a in ref.items():
        if n in exclude:
            continue
        den = top if gradient and zero_by_construction(structure, n) else float(a.abs().max())
        per[n] = float((a - other[n]).abs().max()) / max(den, 1e-300)
    return max(per.values()), per


def _setup(config_id, lr, seed=0, n=160, dtype=torch.float32):
    """Model, optimizer and synthetic data of the task's shape (n examples), seeded; dtype float64 casts the model
    (after the fp32 initialisation) and the inputs to double, so both precisions start from the same values."""
    cfg = train.parse_config_id(config_id, P)
    tc = prereg.task_cfg(cfg["task"], P)
    torch.manual_seed(seed)
    model = train.build_from_config(cfg, tc).to(dtype)
    opt = train.make_optimizer(cfg["optimizer"], model.parameters(), lr)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, tc["n_tokens"], tc["token_dim"], generator=g).to(dtype)
    y = torch.randint(0, tc["classes"], (n,), generator=g)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return cfg, model, opt, X, y, perm


def three_steps(config_id, microbatch, lr, reverse=False, dtype=torch.float32):
    """3 optimizer updates (64, 64, 32 examples) through train.train_step from the same seed and data order; returns
    {name: parameter}. reverse=True: the reordering control (each effective batch's index tensor reversed)."""
    _, model, opt, X, y, perm = _setup(config_id, lr, dtype=dtype)
    for s in range(0, perm.numel(), 64):
        idx = perm[s:s + 64]
        train.train_step(model, opt, X, y, idx.flip(0) if reverse else idx, microbatch, lr)
    out = {n: q.detach().clone() for n, q in model.named_parameters()}
    if any(t.dtype != dtype for t in out.values()):
        raise AssertionError("train.train_step changed the parameter dtype")
    return out


class StepState:
    """Two updates with train.train_step at microbatch 64 (largest grid lr) in `dtype`; then, from deep copies of model
    and optimizer, the gradient the optimizer step receives: train.train_step with lr 0.0, then p.grad (kept after the
    step; clipped). Batches: the 32-example remainder (perm[128:160]) and a full 64-example batch (perm[0:64])."""

    def __init__(self, config_id, lr, dtype=torch.float32):
        self.dtype = dtype
        self.cfg, self.model, self.opt, self.X, self.y, perm = _setup(config_id, lr, dtype=dtype)
        for s in (0, 64):
            train.train_step(self.model, self.opt, self.X, self.y, perm[s:s + 64], 64, lr)
        self.batches = {"remainder32": perm[128:160], "full64": perm[0:64]}

    def grad(self, batch, microbatch, reverse=False, double=False):
        """reverse: the reordering control (index tensor reversed). double: float64 reference FROM THE SAME STATE
        (the deep copy of this fp32 state cast to float64, inputs too; the gradient does not read optimizer state, so
        the copy's step uses a fresh lr-0 SGD over the cast parameters)."""
        m, o = copy.deepcopy((self.model, self.opt))
        if o.param_groups[0]["params"][0] is not next(m.parameters()):
            raise AssertionError("deep copy broke the optimizer-parameter link")
        X, idx = self.X, self.batches[batch]
        if double:
            m, X = m.to(F64), X.to(F64)
            o = torch.optim.SGD(m.parameters(), lr=0.0, foreach=False)
        before = {n: q.detach().clone() for n, q in m.named_parameters()}
        train.train_step(m, o, X, self.y, idx.flip(0) if reverse else idx, microbatch, 0.0)
        for n, q in m.named_parameters():                      # an lr-0 step leaves the parameters unchanged
            if not torch.equal(before[n], q.detach()):
                raise AssertionError(f"lr-0 step changed {n}")
        want = F64 if double else self.dtype
        out = {n: q.grad.detach().clone() for n, q in m.named_parameters()}
        if any(t.dtype != want for t in out.values()):
            raise AssertionError("train.train_step produced gradients of another dtype")
        return out


class _FirstChunkScaled:
    """Stand-in for train.F_: the first cross_entropy call of a step (the first microbatch chunk) is scaled."""

    def __init__(self, scale):
        self.scale, self.n = scale, 0

    def __getattr__(self, k):
        return getattr(torch.nn.functional, k)

    def cross_entropy(self, *a, **kw):
        self.n += 1
        loss = torch.nn.functional.cross_entropy(*a, **kw)
        return loss * self.scale if self.n == 1 else loss


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def _maxes(out, keys):
    return {k: max(r[k] for r in out.values()) for k in keys}


class TestAccumulation(unittest.TestCase):
    """v3 microbatch-accumulation equivalence: float64 gates (gradient: every optimizer's state; parameters: every
    optimizer), fp32 reported. Configurations from the calibration grid of all three tasks, depth 8 included."""
    BASE = ("HAR|shared|d40|T2|{}", "HAR|two_groups|d40|T4|{}", "HAR|untied|d40|T2|{}", "HAR|wide_single|d40|T1|{}",
            "HAR|reversible_shared|d40|T2|{}", "HAR|reversible_untied|d40|T2|{}", "KWS|untied|d40|T2|{}",
            "CIFAR10|shared|d40|T2|{}")
    GRAD_CONFIGS = BASE + ("KWS|two_groups|d56|T8|{}", "CIFAR10|untied|d56|T8|{}",
                           "CIFAR10|reversible_untied|d56|T8|{}")
    PARAM_CONFIGS = BASE
    CTRL = "HAR|untied|d40|T2|sgd"

    def test_gradient_gate_float64(self):
        """Gate (microbatch_accumulation_gradient_rel, float64): microbatch 8 and 1 vs 64 from the same state (two
        float64 updates at the largest grid lr of each optimizer), the 32-example remainder and a full batch. Includes
        the analysis check: zero-by-construction reference gradients <= 1e-6 x the model's largest max|g|."""
        tol = float(TOL["microbatch_accumulation_gradient_rel"])
        self.assertIn("float64", TOL["precision"])
        out, worst, ratios = {}, {}, {}
        for opt in OPTS:
            lr = float(max(P["training"]["lr_grid"][opt]))
            for tmpl in self.GRAD_CONFIGS:
                cid = tmpl.format(opt)
                st = StepState(cid, lr, dtype=F64)
                s = st.cfg["structure"]
                for batch in st.batches:
                    ref = st.grad(batch, 64)
                    top = max(float(t.abs().max()) for t in ref.values())
                    zero = [n for n in ref if zero_by_construction(s, n)]
                    self.assertEqual(bool(zero), s in NONREVERSIBLE, cid)
                    for n in zero:
                        r = float(ref[n].abs().max()) / top
                        ratios[f"{cid} {batch} {n}"] = r
                        with self.subTest(analysis=cid, batch=batch, tensor=n):
                            self.assertLessEqual(r, ZERO_GRAD_RATIO)
                    for m in (8, 1):
                        e, per = rel_error(ref, st.grad(batch, m), s, gradient=True)
                        key = f"{cid} {batch} m={m}"
                        out[key], worst[key] = e, max(per, key=per.get)
                        with self.subTest(config=cid, batch=batch, microbatch=m):
                            self.assertLessEqual(e, tol, f"{key}: worst tensor {worst[key]}")
        REPORT["f64_gradient_gate_max_per_optimizer"] = {o: max(v for k, v in out.items() if f"|{o} " in k)
                                                         for o in OPTS}
        REPORT["f64_gradient_gate_max_per_structure"] = {
            t.format("").rstrip("|"): {o: max(v for k, v in out.items() if k.startswith(t.format(o) + " "))
                                       for o in OPTS} for t in self.GRAD_CONFIGS}
        REPORT["f64_gradient_gate_n_comparisons"] = len(out)
        REPORT["f64_tok2_bias_ratio_max"] = max(ratios.values())
        REPORT["f64_tok2_bias_ratio_n_tensors"] = len(ratios)

    def test_parameter_gate_float64(self):
        """Gate (microbatch_accumulation_parameter_rel, float64): parameters after three updates (64, 64, 32) at the
        largest grid lr, microbatch 8 and 1 vs 64, every optimizer; for adafactor the zero-by-construction tensors
        (tok2.bias outside reversible couplings) are excluded and their values reported."""
        tol = float(TOL["microbatch_accumulation_parameter_rel"])
        self.assertEqual(sorted(TOL["microbatch_accumulation_parameter_optimizers"]), sorted(OPTS))
        out, worst, excluded = {}, {}, {}
        for opt in OPTS:
            lr = float(max(P["training"]["lr_grid"][opt]))
            for tmpl in self.PARAM_CONFIGS:
                cid = tmpl.format(opt)
                s = train.parse_config_id(cid, P)["structure"]
                ref = three_steps(cid, 64, lr, dtype=F64)
                self.assertGreater(rel_error(ref, three_steps(cid, 64, 0.0, dtype=F64))[0], 100 * tol, cid)  # moved
                exc = [n for n in ref if zero_by_construction(s, n)] if opt == "adafactor" else []
                for m in (8, 1):
                    other = three_steps(cid, m, lr, dtype=F64)
                    e, per = rel_error(ref, other, exclude=exc)
                    key = f"{cid} m={m}"
                    out[key], worst[key] = e, max(per, key=per.get)
                    if exc:
                        excluded[key] = {n: rel_error({n: ref[n]}, {n: other[n]})[0] for n in exc}
                    with self.subTest(config=cid, microbatch=m):
                        self.assertLessEqual(e, tol, f"{key}: worst tensor {worst[key]}")
        REPORT["f64_parameter_gate_max_per_optimizer"] = {o: max(v for k, v in out.items() if f"|{o} m=" in k)
                                                          for o in OPTS}
        REPORT["f64_parameter_gate_max_per_structure"] = {
            t.format("").rstrip("|"): {o: max(v for k, v in out.items() if k.startswith(t.format(o) + " "))
                                       for o in OPTS} for t in self.PARAM_CONFIGS}
        REPORT["f64_adafactor_excluded_tensor_rel_err"] = excluded
        REPORT["f64_adafactor_excluded_max"] = max(v for r in excluded.values() for v in r.values())

    def test_fp32_reported(self):
        """Reported, not gated (v3 precision, fp32_parameter_difference). Gradients from the fp32 state after two fp32
        updates: microbatch 8 and 1 vs 64; microbatch 64, 8, 1 and the reordering control vs a float64 reference
        computed from the same fp32 state; the reordering control vs 64. Parameters after three fp32 updates, every
        optimizer: microbatch 8 and 1 vs 64 next to the reordering control. Only finiteness is asserted."""
        grads, params = {}, {}
        for opt in OPTS:
            lr = float(max(P["training"]["lr_grid"][opt]))
            for tmpl in self.GRAD_CONFIGS:
                cid = tmpl.format(opt)
                st = StepState(cid, lr)
                s = st.cfg["structure"]
                for batch in st.batches:
                    g = {m: st.grad(batch, m) for m in (64, 8, 1)}
                    rev = st.grad(batch, 64, reverse=True)
                    f64 = st.grad(batch, 64, double=True)
                    row = {f"m{m}_vs_m64": rel_error(g[64], g[m], s, gradient=True)[0] for m in (8, 1)}
                    row["reorder_vs_m64"] = rel_error(g[64], rev, s, gradient=True)[0]
                    for lab, t in (("m64", g[64]), ("m8", g[8]), ("m1", g[1]), ("reorder", rev)):
                        row[f"{lab}_vs_same_state_f64"] = rel_error(
                            f64, {n: v.to(F64) for n, v in t.items()}, s, gradient=True)[0]
                    grads[f"{cid} {batch}"] = row
            for tmpl in self.PARAM_CONFIGS:
                cid = tmpl.format(opt)
                ref = three_steps(cid, 64, lr)
                row = {f"m{m}_vs_m64": rel_error(ref, three_steps(cid, m, lr))[0] for m in (8, 1)}
                row["reorder_vs_m64"] = rel_error(ref, three_steps(cid, 64, lr, reverse=True))[0]
                params[cid] = row
        for tab in (grads, params):
            for k, row in tab.items():
                for c, v in row.items():
                    self.assertTrue(math.isfinite(v), f"{k} {c}")
        gk = list(next(iter(grads.values())))
        REPORT["fp32_gradient_reported"] = grads
        REPORT["fp32_gradient_reported_max"] = _maxes(grads, gk)
        REPORT["fp32_gradient_reported_max_per_task"] = {
            t: _maxes({k: v for k, v in grads.items() if k.startswith(t + "|")}, gk) for t in P["tasks"]}
        REPORT["fp32_parameter_reported"] = params
        REPORT["fp32_parameter_reported_max_per_optimizer"] = {
            o: _maxes({k: v for k, v in params.items() if k.endswith("|" + o)}, ["m8_vs_m64", "m1_vs_m64",
                                                                                   "reorder_vs_m64"]) for o in OPTS}
        # negative control of the finiteness check: a non-finite parameter is detected
        bad = {n: v.clone() for n, v in ref.items()}
        next(iter(bad.values())).view(-1)[0] = float("nan")
        self.assertFalse(math.isfinite(rel_error(ref, bad)[0]))

    def test_analysis_check_negative_control(self):
        """The zero-gradient analysis check fails where tok2.bias does carry a gradient (inside a reversible
        coupling), so it is not vacuous and the zero-by-construction rule must not be applied there."""
        st = StepState("HAR|reversible_shared|d40|T2|sgd", 0.3, dtype=F64)
        ref = st.grad("full64", 64)
        top = max(float(t.abs().max()) for t in ref.values())
        r = max(float(v.abs().max()) / top for n, v in ref.items() if n.endswith("tok2.bias"))
        REPORT["negative_control_reversible_tok2_bias_ratio"] = r
        self.assertGreater(r, ZERO_GRAD_RATIO)

    def _ctrl_grads(self, patch=None, batch="full64"):
        st = StepState(self.CTRL, 0.3, dtype=F64)
        ref = st.grad(batch, 64)
        with (patch if patch is not None else contextlib.nullcontext()):
            g8 = st.grad(batch, 8)
        return st, ref, g8

    def test_negative_control_a_dropped_chunk(self):
        """Planted defect (a): the last microbatch chunk is dropped; both float64 gates fail."""
        tol_g = float(TOL["microbatch_accumulation_gradient_rel"])
        tol_p = float(TOL["microbatch_accumulation_parameter_rel"])
        drop = mock.patch.object(train, "_chunks", side_effect=lambda idx, m: idx.split(m)[:-1] or idx.split(m))
        res = {}
        for batch in ("remainder32", "full64"):
            st, ref, g8 = self._ctrl_grads(drop, batch)
            res[f"f64 gradient {batch}"] = rel_error(ref, g8, st.cfg["structure"], gradient=True)[0]
            self.assertGreater(res[f"f64 gradient {batch}"], tol_g)
        ref = three_steps(self.CTRL, 64, 0.3, dtype=F64)
        with drop:
            res["f64 parameters"] = rel_error(ref, three_steps(self.CTRL, 8, 0.3, dtype=F64))[0]
        self.assertGreater(res["f64 parameters"], tol_p)
        REPORT["negative_control_a_dropped_chunk"] = res

    def test_negative_control_b_wrong_scale(self):
        """Planted defect (b): the first chunk's loss is scaled by 1.01; the float64 gradient gate fails."""
        st, ref, g8 = self._ctrl_grads(mock.patch.object(train, "F_", _FirstChunkScaled(1.01)))
        e = rel_error(ref, g8, st.cfg["structure"], gradient=True)[0]
        REPORT["negative_control_b_first_chunk_x1.01"] = e
        self.assertGreater(e, float(TOL["microbatch_accumulation_gradient_rel"]))

    def test_negative_control_c_planted_tok2_bias_error(self):
        """Planted defect (c): 1e-3 x the model's largest max|g| added to one tok2.bias gradient of the float64
        microbatch-8 result; the zero-by-construction rule does not hide it (the clean comparison passes)."""
        tol = float(TOL["microbatch_accumulation_gradient_rel"])
        st, ref, g8 = self._ctrl_grads()
        s = st.cfg["structure"]
        clean = rel_error(ref, g8, s, gradient=True)[0]
        self.assertLessEqual(clean, tol)
        top = max(float(t.abs().max()) for t in ref.values())
        name = next(n for n in g8 if zero_by_construction(s, n))
        g8[name] = g8[name].clone()
        g8[name][0] += 1e-3 * top
        e, per = rel_error(ref, g8, s, gradient=True)
        REPORT["negative_control_c_planted_tok2_bias"] = {"tensor": name, "clean": clean, "planted": e}
        self.assertGreater(e, tol)
        self.assertEqual(max(per, key=per.get), name)


class TestSchedule(unittest.TestCase):
    def test_schedule_values(self):
        lr, W, K = 0.01, 92, 92 * 30
        self.assertEqual(train.lr_at(0, lr, W, K), 0.0)
        self.assertAlmostEqual(train.lr_at(W - 1, lr, W, K), lr * (W - 1) / W, places=15)
        self.assertEqual(train.lr_at(W, lr, W, K), lr)                       # end of warm-up
        last = train.lr_at(K - 1, lr, W, K)
        self.assertAlmostEqual(last, lr * 0.5 * (1 + math.cos(math.pi * (K - 1 - W) / (K - W))), places=18)
        self.assertGreater(last, 0.0)
        self.assertLess(last, 1e-7)
        dec = [train.lr_at(k, lr, W, K) for k in range(W, K)]
        self.assertTrue(all(a > b for a, b in zip(dec, dec[1:])))
        REPORT["schedule"] = {"lr": lr, "W": W, "K": K, "step0": 0.0, "end_of_warmup(step W)": lr, "last": last}

    def test_fit_uses_schedule_per_step(self):
        """fit() passes lr_at(k) to every optimizer step; the warm-up length is one epoch (ceil(N / 64) steps)."""
        seen = []
        real = train.train_step

        def rec(model, opt, X, y, idx, m, lr):
            seen.append((lr, idx.numel()))
            return real(model, opt, X, y, idx, m, lr)

        cfg = train.parse_config_id("HAR|shared|d40|T2|adam", P)
        torch.manual_seed(0)
        model = train.build_from_config(cfg, prereg.task_cfg("HAR", P))
        opt = train.make_optimizer("adam", model.parameters(), 0.01)
        X, y = FakeData()("HAR", "train")
        with mock.patch.object(train, "train_step", side_effect=rec):
            r = train.fit(model, opt, X, y, epochs=2, seed=0, microbatch=64, lr=0.01)
        W, K = 3, 6
        self.assertEqual([b for _, b in seen], [64, 64, 32] * 2)                 # last partial batch kept
        self.assertEqual([lr for lr, _ in seen], [train.lr_at(k, 0.01, W, K) for k in range(K)])
        self.assertEqual((r["steps"], r["steps_planned"], r["warmup_steps"], r["epochs_completed"]), (6, 6, 3, 2))
        # negative control: the other common warm-up convention ((k + 1) / W) is distinguishable
        self.assertNotEqual([lr for lr, _ in seen][:W], [0.01 * (k + 1) / W for k in range(W)])


class TestTestSplitAccess(unittest.TestCase):
    CID = "HAR|shared|d40|T2|adam"

    def _run(self, mode, fake, d):
        return train.train_run(self.CID, 0.001, 0, mode, Path(d) / mode, microbatch=64, max_steps=2)

    def test_dev_runs_never_load_test(self):
        verified = {"dev": None, "smoke": None, "lrcal": ["harness"], "floor": ["harness"],
                    "select": ["harness", "predictor"], "confirm": ["harness", "predictor"]}
        with tempfile.TemporaryDirectory() as d, temp_records("harness", "predictor") as rd:
            recs = {n: hashlib.sha256((rd / "records" / f"CODE_FREEZE_{n}.json").read_bytes()).hexdigest()
                    for n in ("harness", "predictor")}
            for mode in ("dev", "lrcal", "floor", "select", "smoke"):
                fake = FakeData()
                with mock.patch.object(lbdata, "load_task", fake):
                    r = self._run(mode, fake, d)
                self.assertEqual(fake.test_requests(), [], mode)
                self.assertEqual(sorted(c[1] for c in fake.calls), ["train", "val"], mode)
                self.assertIsNone(r["test_acc"])
                want = None if verified[mode] is None else {n: recs[n] for n in verified[mode]}
                self.assertEqual(r["code_freeze"], want, mode)
            fake = FakeData()
            with mock.patch.object(lbdata, "load_task", fake):
                r = self._run("confirm", fake, d)
            self.assertEqual(fake.test_requests(), [("HAR", "test", True)])
            self.assertIsNotNone(r["test_acc"])
            self.assertEqual(r["code_freeze"], recs)
        REPORT["test_access"] = ("dev/lrcal/floor/select/smoke: no test request; confirm: one request "
                                 "('HAR', 'test', True); code_freeze recorded per mode")

    def test_confirm_only_route_is_structural(self):
        self.assertEqual(set(train.MODES), {"dev", "lrcal", "floor", "select", "confirm", "smoke"})
        for mode in ("dev", "lrcal", "floor", "select", "smoke"):
            with self.assertRaises(lbdata.TestSplitAccess):
                train._confirm_only_test_split("HAR", mode)
        with self.assertRaises(lbdata.TestSplitAccess):          # the data card's own gate
            lbdata.load_task("HAR", "test")
        tree = ast.parse((ROOT / "src/lb/train.py").read_text())
        hits = []
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            for c in [n for n in ast.walk(fn) if isinstance(n, ast.Call)]:
                name = getattr(c.func, "attr", getattr(c.func, "id", None))
                if name == "load_task" and any(isinstance(a, ast.Constant) and a.value == "test" for a in c.args):
                    hits.append(fn.name)
                if name == "_confirm_only_test_split":
                    hits.append(("caller", fn.name))
        self.assertEqual([h for h in hits if not isinstance(h, tuple)], ["_confirm_only_test_split"])
        self.assertEqual({h[1] for h in hits if isinstance(h, tuple)}, {"train_run"})

    def test_negative_control_planted_leak_detected(self):
        """Planted defect: the development loader also reads the test split; the recorder catches it."""
        real = train._load_dev

        def leaky(task):
            lbdata.load_task(task, "test", allow_test=True)
            return real(task)

        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(lbdata, "load_task", fake), \
                mock.patch.object(train, "_load_dev", side_effect=leaky):
            self._run("dev", fake, d)
        self.assertNotEqual(fake.test_requests(), [])


class TestDivergenceAndOutputs(unittest.TestCase):
    def test_diverged_run_kept(self):
        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(lbdata, "load_task", fake):
            r = train.train_run("HAR|shared|d40|T2|sgd", 1e30, 0, "dev", Path(d) / "div", max_steps=6)
            self.assertTrue((Path(d) / "div" / "DONE.json").exists())
            disk = json.loads((Path(d) / "div" / "result.json").read_text())
        self.assertEqual(r["status"], "diverged")
        self.assertEqual(disk["status"], "diverged")
        self.assertIsNotNone(r["first_nonfinite_step"])
        self.assertTrue(math.isfinite(r["final_val_acc"]))
        self.assertEqual(r["steps"], 6)                                   # training continued after divergence
        REPORT["diverged"] = {k: r[k] for k in ("status", "first_nonfinite_step", "steps", "final_val_acc",
                                                "final_val_nonfinite_examples", "train_loss_per_epoch")}

    def test_nonfinite_logits_count_wrong(self):
        logits = torch.tensor([[float("nan"), 0.0], [1.0, 0.0], [0.0, float("inf")], [0.0, 1.0]])

        class Fixed(nn.Module):
            def forward(self, x):
                return logits[: x.shape[0]]

        y = torch.tensor([0, 0, 1, 1])
        acc, nonfinite = train.accuracy(Fixed(), torch.zeros(4, 1), y)
        self.assertEqual((acc, nonfinite), (50.0, 2))
        # negative control: plain argmax would score the non-finite rows as correct here (NaN and inf are argmax)
        self.assertEqual(float((logits.argmax(1) == y).float().mean()) * 100, 100.0)

    def test_refuse_overwrite(self):
        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(lbdata, "load_task", fake):
            out = Path(d) / "maj"
            train.train_run("HAR|majority", 0.0, 0, "dev", out)
            h = hashlib.sha256((out / "result.json").read_bytes()).hexdigest()
            with self.assertRaises(train.RefuseOverwrite):
                train.train_run("HAR|majority", 0.0, 0, "dev", out)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
            cp = subprocess.run([PY, "-m", "lb.train", "--config-id", "HAR|majority", "--lr", "0", "--seed", "0",
                                 "--mode", "dev", "--out", str(out)], cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp.returncode, 3, cp.stderr.decode()[-2000:])
            self.assertEqual(hashlib.sha256((out / "result.json").read_bytes()).hexdigest(), h)
            # negative control: a fresh directory is not refused
            train.train_run("HAR|majority", 0.0, 0, "dev", Path(d) / "fresh")

    def test_bars_run(self):
        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(lbdata, "load_task", fake):
            maj = train.train_run("HAR|majority", 0.0, 0, "dev", Path(d) / "maj")
            lin = train.train_run("HAR|linear", 0.003, 0, "dev", Path(d) / "lin", max_steps=3)
            for k in ("result.json", "DONE.json"):
                self.assertTrue((Path(d) / "lin" / k).exists())
        Xtr, ytr = fake("HAR", "train")
        Xva, yva = fake("HAR", "val")
        cls = int(torch.bincount(ytr, minlength=6).argmax())
        self.assertEqual(maj["majority_class"], cls)
        self.assertEqual(maj["final_val_acc"], 100.0 * float((yva == cls).sum()) / yva.numel())
        self.assertEqual((maj["steps"], maj["n_params"]), (0, 0))
        self.assertEqual(lin["n_params"], 16 * 72 * 6 + 6)
        self.assertEqual((lin["steps"], lin["status"], lin["config"]["optimizer"]), (3, "ok", "adam"))
        self.assertGreater(lin["train_flops_per_example"], 0)
        REPORT["bars"] = {"majority_val": maj["final_val_acc"], "linear_val_3_steps": lin["final_val_acc"]}

    def test_result_fields(self):
        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(lbdata, "load_task", fake):
            r = train.train_run("HAR|reversible_shared|d40|T2|adafactor", 0.01, 0, "dev", Path(d) / "r",
                                microbatch=16, max_steps=4)
            done = json.loads((Path(d) / "r" / "DONE.json").read_text())
            rh = hashlib.sha256((Path(d) / "r" / "result.json").read_bytes()).hexdigest()
            with self.assertRaises(ValueError):                 # reversible + checkpointing is not a knob setting
                train.train_run("HAR|reversible_shared|d40|T2|sgd", 0.01, 0, "dev", Path(d) / "x", checkpointing=True)
            self.assertFalse((Path(d) / "x").exists())
        self.assertEqual(done["result_sha256"], rh)
        for k in ("lr", "seed", "mode", "microbatch", "checkpointing", "epochs", "steps", "train_loss_per_epoch",
                  "val_acc_per_epoch", "final_val_acc", "test_acc", "status", "wall_seconds", "n_params",
                  "train_flops_per_example", "prereg_sha256", "code_sha256", "torch", "threads", "start_utc",
                  "end_utc"):
            self.assertIn(k, r)
        self.assertEqual(r["prereg_sha256"], prereg.yaml_sha256())
        self.assertIn("train.py", r["code_sha256"])
        self.assertIn("code_freeze", r)
        self.assertIsNone(r["code_freeze"])                     # dev mode verifies no record
        self.assertEqual(r["threads"], {"intra_op": 1, "inter_op": 1})
        self.assertEqual(r["epochs"], P["training"]["epochs"]["HAR"])
        with self.assertRaises(ValueError):                     # an ID outside registry and calibration grid
            train.parse_config_id("HAR|shared|d36|T2|adam", P)


class TestGuards(unittest.TestCase):
    """code_freeze_s3 guards in train_run (check_guards). Registry IDs are used only where the guard raises before
    any data or model work; allowed registry cases are checked with check_guards alone (verify patched), never run."""

    UNTIED_REF = "HAR|untied|d32|T4|{}"
    TIED_REF = "HAR|shared|d32|T4|{}"

    def _refused(self, config_id, lr, seed, mode, exc, records=("harness", "predictor"), **kw):
        """train_run must raise exc before any data request and before creating the out dir."""
        fake = FakeData()
        with tempfile.TemporaryDirectory() as d, temp_records(*records), \
                mock.patch.object(lbdata, "load_task", fake), \
                mock.patch.object(train, "build_from_config", side_effect=AssertionError("model built")):
            with self.assertRaises(exc) as cm:
                train.train_run(config_id, lr, seed, mode, Path(d) / "out", **kw)
            self.assertFalse((Path(d) / "out").exists())
        self.assertEqual(fake.calls, [])
        return str(cm.exception)

    def test_floor_rule_states_seeds(self):
        self.assertIn("seeds 101-105", P["floor"]["rule"])
        self.assertEqual(train.FLOOR_SEEDS, (101, 102, 103, 104, 105))
        self.assertEqual(train.reference_configs(P), {"untied": ("untied", 32, 4), "tied": ("shared", 32, 4)})
        self.assertNotIn("seeds 101-105", P["floor"]["rule"].replace("101-105", "101-106"))   # negative control
        # the seed sets of selection, confirmation and top-up are disjoint from the floor seeds
        others = set(P["seeds"]["selection"]) | set(P["seeds"]["confirmation"]) | set(P["seeds"]["top_up"])
        self.assertFalse(others & set(train.FLOOR_SEEDS))

    def test_allowed_cases_pass_and_verify_calls(self):
        """Every allowed registry case passes; lrcal/floor verify the harness record, select/confirm both records,
        for registry, calibration and bar configurations alike; dev/smoke verify nothing."""
        calls = []

        def fake_verify(*names):
            calls.append(names)
            return {n: "sha-" + n for n in names}

        n_ok = 0
        with mock.patch.object(freeze, "verify", side_effect=fake_verify):
            for opt in OPTS:
                for tmpl in (self.UNTIED_REF, self.TIED_REF):
                    cfg = train.parse_config_id(tmpl.format(opt), P)
                    self.assertTrue(cfg["registry_member"])
                    for lr in P["training"]["lr_grid"][opt]:
                        calls.clear()
                        r = train.check_guards(cfg, "lrcal", float(lr), 0, 64, False, P)
                        self.assertEqual((r, calls), ({"harness": "sha-harness"}, [("harness",)]))
                        n_ok += 1
                    for mode, seeds in (("select", P["seeds"]["selection"]),
                                        ("confirm", P["seeds"]["confirmation"] + P["seeds"]["top_up"])):
                        for seed in seeds:
                            calls.clear()
                            train.check_guards(cfg, mode, 0.001, seed, 64, False, P)
                            self.assertEqual(calls, [("harness", "predictor")])
                            n_ok += 1
            cfg = train.parse_config_id(self.UNTIED_REF.format("adam"), P)
            for seed in train.FLOOR_SEEDS:
                for lr in P["training"]["lr_grid"]["adam"]:
                    calls.clear()
                    train.check_guards(cfg, "floor", float(lr), seed, 64, False, P)
                    self.assertEqual(calls, [("harness",)])
                    n_ok += 1
            for cid in ("HAR|majority", "HAR|linear", "HAR|shared|d40|T2|adam"):
                cfg = train.parse_config_id(cid, P)
                self.assertFalse(cfg["registry_member"])
                for mode, want in (("dev", []), ("smoke", []), ("lrcal", [("harness",)]), ("floor", [("harness",)]),
                                   ("select", [("harness", "predictor")]), ("confirm", [("harness", "predictor")])):
                    calls.clear()
                    train.check_guards(cfg, mode, 0.001, 0, 64, False, P)
                    self.assertEqual(calls, want, f"{cid} {mode}")
        REPORT["guards_allowed_registry_cases"] = n_ok

    def test_negative_controls_mode_guards(self):
        res = {}
        res["lrcal non-reference"] = self._refused("HAR|untied|d32|T2|adam", 0.001, 0, "lrcal", train.GuardViolation)
        res["lrcal lr off grid"] = self._refused(self.UNTIED_REF.format("adam"), 0.002, 0, "lrcal",
                                                 train.GuardViolation)
        res["lrcal seed 1"] = self._refused(self.TIED_REF.format("sgd"), 0.1, 1, "lrcal", train.GuardViolation)
        res["lrcal microbatch 8"] = self._refused(self.TIED_REF.format("sgd"), 0.1, 0, "lrcal", train.GuardViolation,
                                                  microbatch=8)
        res["lrcal pointer_shared"] = self._refused("HAR|pointer_shared|d32|T4|adam", 0.001, 0, "lrcal",
                                                    train.GuardViolation)
        res["floor seed 7"] = self._refused(self.UNTIED_REF.format("adam"), 0.001, 7, "floor", train.GuardViolation)
        res["floor tied reference"] = self._refused(self.TIED_REF.format("adam"), 0.001, 101, "floor",
                                                    train.GuardViolation)
        res["floor sgd"] = self._refused(self.UNTIED_REF.format("sgd"), 0.1, 101, "floor", train.GuardViolation)
        res["floor checkpointing"] = self._refused(self.UNTIED_REF.format("adam"), 0.001, 101, "floor",
                                                   train.GuardViolation, checkpointing=True)
        res["select without predictor record (calibration)"] = self._refused(
            "HAR|shared|d40|T2|adam", 0.001, 0, "select", freeze.CodeFreezeError, records=("harness",))
        res["select without predictor record (registry)"] = self._refused(
            self.TIED_REF.format("adam"), 0.001, 0, "select", freeze.CodeFreezeError, records=("harness",))
        res["select seed 1"] = self._refused(self.TIED_REF.format("adam"), 0.001, 1, "select", train.GuardViolation)
        res["confirm bar without records"] = self._refused("HAR|majority", 0.0, 1, "confirm", freeze.CodeFreezeError,
                                                           records=())
        res["lrcal without harness record"] = self._refused(self.UNTIED_REF.format("adam"), 0.001, 0, "lrcal",
                                                            freeze.CodeFreezeError, records=())
        res["dev registry"] = self._refused(self.TIED_REF.format("adam"), 0.001, 0, "dev", train.GuardViolation)
        res["smoke registry"] = self._refused(self.UNTIED_REF.format("adam"), 0.001, 0, "smoke", train.GuardViolation)
        res["confirm seed 0"] = self._refused(self.UNTIED_REF.format("adam"), 0.001, 0, "confirm",
                                              train.GuardViolation)
        REPORT["guard_negative_controls"] = res

    def test_cli_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
            out = Path(d) / "x"
            cp = subprocess.run([PY, "-m", "lb.train", "--config-id", "HAR|shared|d32|T4|adam", "--lr", "0.001",
                                 "--seed", "0", "--mode", "dev", "--out", str(out)], cwd=ROOT, env=env,
                                capture_output=True)
            self.assertEqual(cp.returncode, 2, cp.stderr.decode()[-2000:])
            self.assertIn("GuardViolation", cp.stderr.decode())
            self.assertFalse(out.exists())
            # the CLI accepts --mode select; a missing record is an ERROR (exit 2) before any data request
            err, fake = io.StringIO(), FakeData()
            with temp_records(), mock.patch.object(lbdata, "load_task", fake), contextlib.redirect_stderr(err):
                rc = train.main(["--config-id", "HAR|majority", "--lr", "0", "--seed", "0", "--mode", "select",
                                 "--out", str(out)])
            self.assertEqual((rc, fake.calls, out.exists()), (2, [], False))
            self.assertIn("CodeFreezeError", err.getvalue())
            REPORT["guard_cli"] = {"dev_registry_exit": cp.returncode, "stderr_tail": cp.stderr.decode()[-160:],
                                   "select_without_records_exit": rc}
        # negative control: an unknown mode is rejected by the parser itself
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train.main(["--config-id", "HAR|majority", "--lr", "0", "--seed", "0", "--mode", "bogus", "--out", "x"])


HAR_DATA = "HAR" in json.loads((ROOT / "research/s3/data_manifest.json").read_text()).get("tasks", {}) \
    if (ROOT / "research/s3/data_manifest.json").exists() else False


@unittest.skipUnless(HAR_DATA, "HAR data not prepared")
class TestLauncher(unittest.TestCase):
    def test_skip_rerun_once_resume(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            specs = d / "runs.jsonl"
            good = ["--config-id", "HAR|majority", "--lr", "0", "--seed", "0", "--mode", "smoke"]
            bad = ["--config-id", "HAR|shared|d36|T2|adam", "--lr", "0.001", "--seed", "0", "--mode", "smoke"]
            specs.write_text("\n".join(json.dumps(s) for s in [
                {"run_id": "good", "args": good, "out": str(d / "good")},
                {"run_id": "bad", "args": bad, "out": str(d / "bad")}]) + "\n")
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            cmd = [PY, "-m", "lb.launch", "--specs", str(specs), "--jobs", "2", "--log", str(d / "launch.log")]
            cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp.returncode, 1, cp.stderr.decode()[-2000:])
            self.assertTrue((d / "good" / "DONE.json").exists())
            self.assertFalse((d / "bad" / "DONE.json").exists())
            self.assertEqual(sorted(p.name for p in (d / "bad").glob("launch_attempt*.log")),
                             ["launch_attempt1.log", "launch_attempt2.log"])
            ev = [json.loads(x) for x in (d / "launch.log").read_text().splitlines()]
            self.assertEqual([e["run_id"] for e in ev if e["event"] == "rerun"], ["bad"])
            r = json.loads((d / "good" / "result.json").read_text())
            self.assertEqual(r["threads"], {"intra_op": 1, "inter_op": 1})
            self.assertIsNone(r["test_acc"])
            # resume: nothing is rerun
            cp2 = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp2.returncode, 1)
            ev2 = [json.loads(x) for x in (d / "launch.log").read_text().splitlines()][len(ev):]
            self.assertEqual({e["run_id"]: e["reason"] for e in ev2 if e["event"] == "skip"},
                             {"good": "DONE.json exists", "bad": "two counted attempts without DONE.json"})
            self.assertFalse((d / "bad" / "launch_attempt3.log").exists())
            # both finished attempts are counted (exit records present)
            self.assertEqual(launch.attempt_state(d / "bad")[0], 2)
            cp3 = subprocess.run(cmd[:-4] + ["--jobs", "9", "--log", str(d / "x.log")], cwd=ROOT, env=env,
                                 capture_output=True)
            self.assertEqual(cp3.returncode, 2)                     # more than 8 concurrent processes refused
            REPORT["launcher"] = {"first_exit": cp.returncode, "resume_exit": cp2.returncode,
                                  "jobs9_exit": cp3.returncode, "events": [e["event"] for e in ev]}


class TestLauncherRefusal(unittest.TestCase):
    """Exit 2 (guard violation, argparse usage error) is a refusal: recorded, never rerun, launcher exit 1. Negative
    control: an exit-1 crash in the same launch is still rerun once. No data is read and nothing is trained."""

    def test_refusal_not_rerun_crash_rerun_once(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            specs = d / "runs.jsonl"
            guard = ["--config-id", "HAR|shared|d32|T4|adam", "--lr", "0.001", "--seed", "0", "--mode", "dev"]
            usage = ["--config-id", "HAR|majority", "--lr", "0", "--mode", "dev"]          # --seed missing
            crash = ["--config-id", "HAR|shared|d36|T2|adam", "--lr", "0.001", "--seed", "0", "--mode", "smoke"]
            specs.write_text("\n".join(json.dumps(s) for s in [
                {"run_id": "guard", "args": guard, "out": str(d / "guard")},
                {"run_id": "usage", "args": usage, "out": str(d / "usage")},
                {"run_id": "crash", "args": crash, "out": str(d / "crash")}]) + "\n")
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            cmd = [PY, "-m", "lb.launch", "--specs", str(specs), "--jobs", "3", "--log", str(d / "launch.log")]
            cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp.returncode, 1, cp.stderr.decode()[-2000:])
            ev = [json.loads(x) for x in (d / "launch.log").read_text().splitlines()]
            end = [e for e in ev if e["event"] == "launcher_end"][-1]
            self.assertEqual(end["summary"], {"guard": "refused", "usage": "refused", "crash": "failed"})
            for rid in ("guard", "usage"):
                self.assertEqual(sorted(p.name for p in (d / rid).glob("launch_attempt*.log")), ["launch_attempt1.log"])
                rec = json.loads((d / rid / "launch_attempt1.json").read_text())
                self.assertEqual((rec["status"], rec["exit"]), ("refused", 2))
                self.assertEqual(launch.attempt_state(d / rid)[0], 0)          # a refusal is not a counted attempt
            self.assertIn("GuardViolation", (d / "guard" / "launch_attempt1.log").read_text())
            # negative control: the exit-1 crash is an execution error and is rerun exactly once
            self.assertEqual(sorted(p.name for p in (d / "crash").glob("launch_attempt*.log")),
                             ["launch_attempt1.log", "launch_attempt2.log"])
            self.assertEqual(json.loads((d / "crash" / "launch_attempt1.json").read_text())["exit"], 1)
            self.assertEqual([e["run_id"] for e in ev if e["event"] == "rerun"], ["crash"])
            self.assertEqual(sorted(e["run_id"] for e in ev if e["event"] == "refused"), ["guard", "usage"])
            # resume: the refused specs are skipped, never rerun
            cp2 = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp2.returncode, 1)
            ev2 = [json.loads(x) for x in (d / "launch.log").read_text().splitlines()][len(ev):]
            skips = {e["run_id"]: e["reason"] for e in ev2 if e["event"] == "skip"}
            self.assertEqual(skips["guard"], "refused (exit 2) in attempt 1")
            self.assertEqual(skips["usage"], "refused (exit 2) in attempt 1")
            self.assertEqual(skips["crash"], "two counted attempts without DONE.json")
            self.assertFalse((d / "guard" / "launch_attempt2.log").exists())
            REPORT["launcher_refusal"] = {"exit": cp.returncode, "summary": end["summary"], "resume_exit": cp2.returncode,
                                          "resume_skips": skips}


if __name__ == "__main__":
    unittest.main()
