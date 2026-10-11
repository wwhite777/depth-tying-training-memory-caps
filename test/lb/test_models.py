"""S3 equivalence and enumeration tests for src/lb/models.py and src/lb/registry.py (card S3-M).

Run (CPU, single thread, deterministic):
  CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_models.py -v
Tolerances come from the preregistration in force (lb.prereg) equivalence_tolerances (read through lb.prereg.load()).
Relative error of a tensor pair = max|a - b| / max|b| (b = reference); the max is taken over all compared tensors.
Shapes: registry widths 24 and 32 as before, plus every calibration-grid shape (widths 40 and 56, depth 8 included) of
all three tasks (equivalence_tolerances.test_configurations), in fp32 at the frozen tolerances.
Every check has an in-file negative control showing it can fail.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F_

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lb import prereg, registry  # noqa: E402
from lb import models  # noqa: E402
from lb.models import Coupling, build_model, build_pointer_shared  # noqa: E402

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

P = prereg.load()
TOL = P["equivalence_tolerances"]
TASKS = list(P["tasks"])
CFGS = {t: prereg.task_cfg(t, P) for t in TASKS}
B = 4
REPORT = {}


def _cal_shapes(task, structures):
    """Sorted unique (structure, d, T) of the calibration grid (registry.calibration_grid) for the given structures."""
    return sorted({(c["structure"], c["d"], c["T"]) for c in registry.calibration_grid(task)
                   if c["structure"] in structures})


def _shapes(task, structures, base):
    """base (structure, d, T) shapes at registry widths, then the calibration-grid shapes of the task."""
    return list(base) + _cal_shapes(task, structures)


def _batch(cfg, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, cfg["n_tokens"], cfg["token_dim"], generator=g)
    y = torch.randint(0, cfg["classes"], (B,), generator=g)
    return x, y


def _seeded(fn, *a):
    torch.manual_seed(1234)
    return fn(*a)


def _rel(a, b):
    den = b.abs().max().item()
    return (a - b).abs().max().item() / (den if den > 0 else 1.0)


def _max_rel(xs, ys):
    return max(_rel(a, b) for a, b in zip(xs, ys))


def _bitwise(xs, ys):
    xs, ys = list(xs), list(ys)
    return len(xs) == len(ys) and all(torch.equal(a, b) for a, b in zip(xs, ys))


OPTS = {
    "adam": lambda ps: torch.optim.Adam(ps, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, foreach=False, fused=False),
    "sgd_momentum": lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9, nesterov=False, foreach=False, fused=False),
    "sgd": lambda ps: torch.optim.SGD(ps, lr=0.1, foreach=False, fused=False),
    "adafactor": lambda ps: torch.optim.Adafactor(ps, lr=1e-2, foreach=False),
}


class PointerSharedEqualsLooped(unittest.TestCase):
    """pointer_shared_equals_looped: tolerance 'bitwise equal' (outputs, gradients, one update)."""

    def _pair(self, cfg, d, T):
        a = _seeded(build_model, cfg, "shared", d, T)
        b = _seeded(build_pointer_shared, cfg, d, T)
        return a, b

    def test_pointer_shared_equals_looped(self):
        self.assertIn("bitwise", TOL["pointer_shared_equality"])
        n = 0
        cal = 0
        for task, cfg in CFGS.items():
            base = [("shared", d, T) for d in (24, 32) for T in (2, 4)]
            for _, d, T in _shapes(task, ("shared",), base):
                cal += d in (40, 56)
                x, y = _batch(cfg, 7)
                a, b = self._pair(cfg, d, T)
                self.assertEqual(len(b.blocks), T)
                self.assertTrue(all(blk is b.blocks[0] for blk in b.blocks))
                self.assertEqual(a.n_params, b.n_params)
                self.assertTrue(_bitwise(a.parameters(), b.parameters()), "initial weights")
                oa, ob = a(x), b(x)
                self.assertTrue(torch.equal(oa, ob), f"{task} d{d} T{T} outputs")
                F_.cross_entropy(oa, y).backward()
                F_.cross_entropy(ob, y).backward()
                self.assertTrue(_bitwise([p.grad for p in a.parameters()], [p.grad for p in b.parameters()]),
                                f"{task} d{d} T{T} gradients")
                for name, mk in OPTS.items():
                    a, b = self._pair(cfg, d, T)
                    for m in (a, b):
                        opt = mk(list(m.parameters()))
                        opt.zero_grad()
                        F_.cross_entropy(m(x), y).backward()
                        opt.step()
                    self.assertFalse(_bitwise(a.parameters(), _seeded(build_model, cfg, "shared", d, T).parameters()),
                                     f"{name}: the step changed nothing")
                    self.assertTrue(_bitwise(a.parameters(), b.parameters()), f"{task} d{d} T{T} {name} update")
                    n += 1
        REPORT["pointer_shared_cases"] = n
        REPORT["pointer_shared_calibration_shapes"] = cal

    def test_negative_control_one_ulp(self):
        cfg = CFGS[TASKS[0]]
        a, b = self._pair(cfg, 24, 2)
        x, _ = _batch(cfg, 7)
        with torch.no_grad():
            w = b.blocks[0].ch1.weight
            w[0, 0] = torch.nextafter(w[0, 0], torch.tensor(float("inf")))
        self.assertFalse(torch.equal(a(x), b(x)), "a one-ulp weight change must break bitwise equality")


class ExactCheckpointEquivalence(unittest.TestCase):
    """exact_checkpoint_equivalence: identical loss, gradients within exact_checkpoint_gradient_rel."""

    CASES = [("shared", (2, 4, 8)), ("two_groups", (4, 8)), ("untied", (2, 4, 8)), ("wide_single", (1,))]

    def _run(self, cfg, s, d, T, ckpt, x, y):
        m = _seeded(build_model, cfg, s, d, T).set_checkpointing(ckpt)
        calls = [0]   # pre-hook: non-reentrant checkpoint stops recomputation early, so a post-hook would not fire
        for blk in m.blocks:
            blk.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        loss = F_.cross_entropy(m(x), y)
        loss.backward()
        return loss.detach(), [p.grad for p in m.parameters()], calls[0]

    def test_exact_checkpoint_equivalence(self):
        tol = float(TOL["exact_checkpoint_gradient_rel"])
        worst = 0.0
        n = 0
        for task, cfg in CFGS.items():
            x, y = _batch(cfg, 11)
            base = [(s, d, T) for s, Ts in self.CASES for d in (24, 32) for T in Ts]
            for s, d, T in _shapes(task, [c[0] for c in self.CASES], base):
                n += 1
                l0, g0, c0 = self._run(cfg, s, d, T, False, x, y)
                l1, g1, c1 = self._run(cfg, s, d, T, True, x, y)
                self.assertEqual(c0, T)
                self.assertEqual(c1, 2 * T, "checkpointing must recompute every block once in backward")
                self.assertTrue(torch.equal(l0, l1), f"{task} {s} d{d} T{T} loss")
                r = _max_rel(g1, g0)
                worst = max(worst, r)
                self.assertLessEqual(r, tol, f"{task} {s} d{d} T{T} grad rel {r}")
        REPORT["checkpoint_max_rel"] = worst
        REPORT["checkpoint_cases"] = n

    def test_reversible_ignores_checkpointing(self):
        cfg = CFGS[TASKS[0]]
        x, y = _batch(cfg, 11)
        m = _seeded(build_model, cfg, "reversible_untied", 24, 2).set_checkpointing(True)
        with mock.patch.object(models, "checkpoint", side_effect=AssertionError("checkpoint called")):
            F_.cross_entropy(m(x), y).backward()

    def test_negative_control(self):
        cfg = CFGS[TASKS[0]]
        x, y = _batch(cfg, 11)
        _, g0, _ = self._run(cfg, "untied", 24, 2, False, x, y)
        g1 = [g * (1 + 1e-5) for g in g0]
        self.assertGreater(_max_rel(g1, g0), float(TOL["exact_checkpoint_gradient_rel"]))


def _reversible_grads(m, x, y, custom):
    m.custom_backward = custom
    m.zero_grad(set_to_none=True)
    xr = x.clone().requires_grad_(True)
    F_.cross_entropy(m(xr), y).backward()
    return [xr.grad] + [p.grad.clone() for p in m.parameters()]


def _inversion_error(m, x):
    with torch.no_grad():
        z = m.embed(x)
        seq = [m.blocks[i] for i in m.schedule]
        true = []
        for c in seq:
            true.append(z)
            z = c(z)
        err = 0.0
        for c, t in zip(reversed(seq), reversed(true)):
            z = c.inverse(z)
            err = max(err, (z - t).abs().max().item())
    return err


def _broken_inverse(self, yv):
    y1, y2 = yv.chunk(2, dim=-1)
    x2 = y2                      # G term dropped on purpose
    x1 = y1 - self.f(x2)
    return torch.cat([x1, x2], dim=-1)


class ReversibleGradientAndInversion(unittest.TestCase):
    """reversible_gradient_and_inversion_check: rel grad <= reversible_gradient_rel; inversion abs <= reversible_inversion_abs."""

    def _check(self, task, s, d, T):
        cfg = CFGS[task]
        x, y = _batch(cfg, 13)
        m = _seeded(build_model, cfg, s, d, T)
        ref = _reversible_grads(m, x, y, custom=False)
        got = _reversible_grads(m, x, y, custom=True)
        return _max_rel(got, ref), _inversion_error(m, x)

    def test_reversible_gradient_and_inversion_check(self):
        gt, it = float(TOL["reversible_gradient_rel"]), float(TOL["reversible_inversion_abs"])
        worst_g = worst_i = 0.0
        n = 0
        for task in TASKS:
            base = [(s, d, T) for s in ("reversible_untied", "reversible_shared") for d in (24, 32) for T in (2, 4, 8)]
            for s, d, T in _shapes(task, ("reversible_untied", "reversible_shared"), base):
                n += 1
                g, i = self._check(task, s, d, T)
                worst_g, worst_i = max(worst_g, g), max(worst_i, i)
                self.assertLessEqual(g, gt, f"{task} {s} d{d} T{T} grad rel {g}")
                self.assertLessEqual(i, it, f"{task} {s} d{d} T{T} inversion abs {i}")
        REPORT["reversible_max_grad_rel"] = worst_g
        REPORT["reversible_cases"] = n
        REPORT["reversible_max_inversion_abs"] = worst_i

    def test_broken_inverse_fails(self):
        gt, it = float(TOL["reversible_gradient_rel"]), float(TOL["reversible_inversion_abs"])
        with mock.patch.object(Coupling, "inverse", _broken_inverse):
            for s in ("reversible_untied", "reversible_shared"):
                g, i = self._check(TASKS[0], s, 24, 4)
                self.assertGreater(g, gt, f"{s}: broken inverse passed the gradient check")
                self.assertGreater(i, it, f"{s}: broken inverse passed the inversion check")
                REPORT[f"broken_inverse_{s}"] = (g, i)

    def test_activations_not_stored_per_coupling(self):
        cfg = CFGS[TASKS[0]]
        x, y = _batch(cfg, 13)
        counts = {}
        for custom in (True, False):
            for T in (2, 8):
                m = _seeded(build_model, cfg, "reversible_untied", 24, T)
                m.custom_backward = custom
                n = [0]

                def pack(t):
                    n[0] += 1
                    return t
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    out = m(x)
                F_.cross_entropy(out, y).backward()
                counts[(custom, T)] = n[0]
        self.assertEqual(counts[(True, 2)], counts[(True, 8)], counts)
        self.assertGreater(counts[(False, 8)], counts[(False, 2)], counts)
        REPORT["saved_tensor_counts"] = counts


def _block_params(n, d):
    return 4 * d + 2 * n * n + 2 * n + 4 * d * d + 3 * d


class StructureCounts(unittest.TestCase):
    def test_structure_counts(self):
        for task, cfg in CFGS.items():
            n, k, c = cfg["n_tokens"], cfg["token_dim"], cfg["classes"]
            for d in (24, 32, 48):
                outer = k * d + d + 2 * d + d * c + c
                blk, cpl = _block_params(n, d), 2 * _block_params(n, d // 2)
                for s, T, want in [("shared", 4, blk), ("shared", 8, blk), ("two_groups", 4, 2 * blk),
                                   ("two_groups", 8, 2 * blk), ("untied", 2, 2 * blk), ("untied", 4, 4 * blk),
                                   ("untied", 8, 8 * blk), ("wide_single", 1, blk), ("reversible_shared", 4, cpl),
                                   ("reversible_shared", 8, cpl), ("reversible_untied", 4, 4 * cpl),
                                   ("reversible_untied", 8, 8 * cpl)]:
                    m = build_model(cfg, s, d, T)
                    self.assertEqual(m.n_params, outer + want, f"{task} {s} d{d} T{T}")
                    self.assertEqual(m.tied, s in P["registry"]["tied_class"])
                    self.assertEqual((m.structure, m.d, m.T), (s, d, T))
                    if d == 32 and T in (4, 1):
                        REPORT.setdefault("n_params_d32_T4", {})[f"{task}|{s}|T{T}"] = m.n_params
        self.assertNotEqual(build_model(CFGS[TASKS[0]], "untied", 24, 4).n_params, build_model(CFGS[TASKS[0]], "shared", 24, 4).n_params)


class Registry(unittest.TestCase):
    def test_registry(self):
        reg = P["registry"]
        ids = set()
        for task in TASKS:
            r = registry.enumerate(task)
            self.assertEqual(len(r), int(reg["configurations_per_task"]))
            self.assertEqual(len(r), 248)
            self.assertEqual(sum(c["tied"] for c in r), 128)
            self.assertEqual(sum(not c["tied"] for c in r), 120)
            self.assertEqual(len({c["config_id"] for c in r}), 248)
            self.assertEqual(list(r[0]), ["config_id", "task", "structure", "d", "T", "optimizer", "tied", "n_params"])
            for c in r[::17]:
                self.assertEqual(c["n_params"], build_model(CFGS[task], c["structure"], c["d"], c["T"]).n_params)
            cal = registry.calibration_grid(task)
            self.assertEqual({c["d"] for c in cal}, {40, 56})
            self.assertFalse({c["config_id"] for c in cal} & {c["config_id"] for c in r})
            REPORT.setdefault("calibration_grid_size", {})[task] = len(cal)
            ids |= {c["config_id"] for c in r}
        self.assertEqual(len(ids), 248 * len(TASKS))

    def test_hash_mismatch_raises(self):
        with mock.patch.object(prereg, "verify", side_effect=prereg.PreregMismatch("simulated")):
            with self.assertRaises(prereg.PreregMismatch):
                registry.enumerate(TASKS[0])
            with self.assertRaises(prereg.PreregMismatch):
                registry.calibration_grid(TASKS[0])


def tearDownModule():
    print("\nREPORT", {k: REPORT[k] for k in sorted(REPORT)}, file=sys.stderr)


if __name__ == "__main__":
    unittest.main()
