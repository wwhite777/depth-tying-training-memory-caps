"""Synthetic tests of research/s9/transfer_analysis.py (and the counts of make_transfer_specs.py).
No real run is read except the copied selection.json, judge_final and probe timing; nothing is written under result/.
Every fake tree lives in a temporary directory (set TMPDIR to choose where).

  TMPDIR=<scratch> ~/envs/jeongwoncheol_loopbudget_docs/bin/python -m unittest research/s9/test_transfer_analysis.py -v

TRANSFER_ANALYSIS_SCRIPT=<path> runs the tests against another copy of the analysis script (planted-failure checks).
"""
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = pathlib.Path(os.environ.get("TRANSFER_ANALYSIS_SCRIPT", HERE / "transfer_analysis.py"))
COPY = ("research/s4/selection.json", "research/s4/judge_final_20261011T054347Z.json",
        "research/s9/transfer_probe/timing_20261014T070439Z.json")
PRECOND_E = ("KWS|shared|d32|T2|adafactor", "CIFAR10|shared|d24|T2|adafactor")
N_EPOCHS = 5


def stem(cid):
    return cid.replace("|", "__")


def name(cid, lr, s):
    return f"{stem(cid)}__lr{repr(float(lr))}__s{s}"


def configs():
    sel = json.loads((ROOT / COPY[0]).read_text())
    cfgs = {}
    for c in sel["cells"]:
        if c["cap_kib"] != "unbounded":
            for cls in ("tied", "untied"):
                cfgs[c[cls]["config_id"]] = c[cls]
    return cfgs


CFGS = configs()


def write_run(d, cid, seed, mb, ckpt, lr, test_acc, losses, wall=100.0, status="ok"):
    d.mkdir(parents=True)
    r = {"config_id": cid, "seed": seed, "microbatch": mb, "checkpointing": ckpt, "lr": lr, "mode": "confirm",
         "status": status, "test_acc": float(test_acc), "final_val_acc": float(test_acc) + 0.5,
         "train_loss_per_epoch": [float(x) for x in losses], "wall_seconds": float(wall)}
    (d / "result.json").write_text(json.dumps(r))
    (d / "DONE.json").write_text(json.dumps({"status": status}))


class Tree:
    """A fake project tree. base[cid] is the archived test_acc of seed 1; seeds 2-12 add a fixed spread.
    delta(cid, seed) gives the feasible-minus-archived shift; loss_change(cid, seed) optionally perturbs the
    feasible loss curve as (epoch index, factor)."""

    def __init__(self, design_seeds, delta, base=None, loss_change=None, env_all=False, skip=(), env_shift=None):
        self.tmp = tempfile.mkdtemp(prefix="lb_s9_test_")
        self.root = pathlib.Path(self.tmp) / "root"
        for rel in COPY:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / rel, self.root / rel)
        spread = np.linspace(-1.5, 1.5, 12)
        for i, (cid, e) in enumerate(sorted(CFGS.items())):
            b = (base or {}).get(cid, 70.0 + 0.7 * i)
            ckpt = bool(e["argmin_knob"]["checkpointing"])
            for s in range(1, 13):
                acc = b + spread[s - 1]
                losses = [2.0 / (k + 1) + 0.001 * s + 0.0001 * i for k in range(N_EPOCHS)]
                write_run(self.root / "result/s4/confirm" / name(cid, e["lr"], s), cid, s, 64, False, e["lr"],
                          acc, losses, wall=50.0)
                env = env_all and s in design_seeds or (s == 1 and cid in PRECOND_E)
                if env:
                    sh = (env_shift or {}).get((cid, s), 0.0)
                    write_run(self.root / "result/s9/transfer_env" / name(cid, e["lr"], s), cid, s, 64, False,
                              e["lr"], acc + sh, losses, wall=55.0)
                if s in design_seeds and (cid, s) not in skip:
                    fl = list(losses)
                    if loss_change and loss_change(cid, s):
                        k, f = loss_change(cid, s)
                        for j in range(k, N_EPOCHS):
                            fl[j] = fl[j] * f
                    fname = name(cid, e["lr"], s) + "__m1" + ("_ckpt" if ckpt else "")
                    write_run(self.root / "result/s9/transfer" / fname, cid, s, 1, ckpt, e["lr"],
                              acc + delta(cid, s), fl, wall=1000.0)

    def run(self, design, *extra):
        out = pathlib.Path(self.tmp) / f"out_{len(list(pathlib.Path(self.tmp).glob('out_*')))}.json"
        p = subprocess.run([sys.executable, str(SCRIPT), "--design", design, "--out", str(out),
                            "--root", str(self.root), *extra], capture_output=True, text=True)
        res = json.loads(out.read_text()) if out.exists() else None
        return p.returncode, res, p.stdout + p.stderr

    def close(self):
        shutil.rmtree(self.tmp)


class TransferAnalysisTests(unittest.TestCase):
    def tree(self, *a, **k):
        t = Tree(*a, **k)
        self.addCleanup(t.close)
        return t

    def test_a_zero_delta_supported(self):
        t = self.tree((1, 2, 3), lambda c, s: 0.0)
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["R1"]["n"], 22)
        self.assertEqual(res["R1"]["df"], 21)
        self.assertEqual(res["R1"]["verdict"], "SUPPORTED")
        self.assertEqual(res["precondition_E"]["result"], "PASS")
        self.assertEqual(res["R2"]["mean_ratio"], 0.0)
        self.assertEqual(res["R3"]["n_sign_changes"], 0)
        self.assertEqual(res["R4"]["n_identical_curves"], 66)
        self.assertEqual(res["n_runs_used"], 66)
        self.assertIn("scipy", res)
        self.assertTrue(any(k.startswith("result/s9/transfer/") for k in res["inputs_sha256"]))

    def test_a_seed1_design_supported(self):
        t = self.tree((1,), lambda c, s: 0.0)
        rc, res, log = t.run("seed1")
        self.assertEqual(rc, 0, log)
        self.assertEqual((res["R1"]["verdict"], res["n_runs_used"]), ("SUPPORTED", 22))

    def test_b_shift_contradicted(self):
        rng = np.random.default_rng(11)
        noise = {(c, s): float(rng.normal(0, 0.05)) for c in CFGS for s in (1, 2, 3)}
        t = self.tree((1, 2, 3), lambda c, s: 1.0 + noise[(c, s)],
                      loss_change=lambda c, s: (2, 1.01))
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["R1"]["verdict"], "CONTRADICTED")
        self.assertAlmostEqual(res["R1"]["mean"], 1.0, delta=0.05)
        self.assertGreater(res["R1"]["ci95"][0], 0)
        self.assertEqual(res["R4"]["first_diff_epoch_counts"], {"3": 66})
        self.assertAlmostEqual(res["R4"]["max_abs_final_rel_diff"], 0.01, places=9)

    def test_c_large_noise_unresolved(self):
        rng = np.random.default_rng(5)
        raw = {(c, s): float(rng.normal(0, 3.0)) for c in sorted(CFGS) for s in (1, 2, 3)}
        m = float(np.mean(list(raw.values())))
        t = self.tree((1, 2, 3), lambda c, s: raw[(c, s)] - m)
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        self.assertAlmostEqual(res["R1"]["mean"], 0.0, places=9)
        self.assertGreater(res["R1"]["ci90"][1], 0.5)
        self.assertEqual(res["R1"]["verdict"], "UNRESOLVED")

    def test_d_missing_run_refused(self):
        miss = ("CIFAR10|shared|d64|T8|adafactor", 2)
        t = self.tree((1, 2, 3), lambda c, s: 0.0, skip=(miss,))
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 2, log)
        self.assertIsNone(res)
        rc, res, log = t.run("seeds123", "--partial")
        self.assertEqual(rc, 0, log)
        self.assertFalse(res["complete"])
        self.assertIsNone(res["R1"]["verdict"])
        self.assertEqual(res["R1"]["verdict_on_available"], "SUPPORTED")
        self.assertEqual(res["n_runs_used"], 65)
        self.assertEqual([(x["config_id"], x["seed"], x["feasible"]) for x in res["excluded"]],
                         [(miss[0], miss[1], "missing")])

    def test_d_failed_status_refused(self):
        t = self.tree((1,), lambda c, s: 0.0)
        cid = "HAR|untied|d24|T2|adafactor"
        p = t.root / "result/s9/transfer" / (name(cid, 0.1, 1) + "__m1_ckpt") / "result.json"
        r = json.loads(p.read_text())
        r["status"] = "diverged"
        p.write_text(json.dumps(r))
        rc, res, log = t.run("seed1")
        self.assertEqual(rc, 2, log)

    def test_e_precondition_mismatch(self):
        cid = "CIFAR10|shared|d24|T2|adafactor"
        t = self.tree((1,), lambda c, s: 0.0, env_shift={(cid, 1): 1e-9})
        rc, res, log = t.run("seed1")
        self.assertEqual(rc, 2, log)
        self.assertIn("precondition E", log)
        rc, res, log = t.run("seed1", "--partial")
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["precondition_E"]["result"], "FAIL")
        st = {k: v["status"] for k, v in res["precondition_E"]["runs"].items()}
        self.assertEqual(st[name(cid, 0.01, 1)], "fail")
        self.assertEqual(st[name("KWS|shared|d32|T2|adafactor", 0.1, 1)], "pass")
        self.assertIsNone(res["R1"]["verdict"])

    def test_e_precondition_mismatch_env_comparator(self):
        cid = "CIFAR10|shared|d24|T2|adafactor"
        t = self.tree((1,), lambda c, s: 0.0, env_all=True, env_shift={(cid, 1): 1e-9})
        rc, res, log = t.run("seed1", "--comparator", "env")
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["precondition_E"]["result"], "FAIL")
        self.assertEqual(res["comparator"], "env")
        self.assertEqual(res["R1"]["verdict"], "SUPPORTED")
        row = [r for r in res["runs"] if r["config_id"] == cid][0]
        self.assertAlmostEqual(row["delta_pp"], -1e-9, delta=1e-12)
        self.assertEqual(res["cost"]["n_env_reruns"], 22)

    def test_e_env_comparator_missing_refused(self):
        t = self.tree((1,), lambda c, s: 0.0)                   # only the two precondition reruns exist
        rc, res, log = t.run("seed1", "--comparator", "env")
        self.assertEqual(rc, 2, log)

    def test_f_r3_sign_changes(self):
        tied, d24, d64 = ("HAR|reversible_shared|d32|T4|adafactor", "HAR|untied|d24|T2|adafactor",
                          "HAR|reversible_untied|d64|T2|adafactor")
        base = {tied: 90.0, d24: 89.0, d64: 95.0}
        t = self.tree((1, 2, 3), lambda c, s: 2.0 if c == d24 else 0.0, base=base)
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        r3 = res["R3"]
        self.assertEqual(r3["n_cells"], 21)
        self.assertEqual(r3["n_sign_changes"], 2)
        self.assertEqual(r3["sign_change_cells"], [["HAR", 256], ["HAR", 323]])
        cell = [c for c in r3["cells"] if (c["task"], c["cap_kib"]) == ("HAR", 256)][0]
        self.assertAlmostEqual(cell["D_comparator_pp"], 1.0, places=9)
        self.assertAlmostEqual(cell["D_feasible_pp"], -1.0, places=9)
        self.assertAlmostEqual(cell["diff_pp"], -2.0, places=9)
        self.assertEqual(cell["seeds"], [1, 2, 3])
        other = [c for c in r3["cells"] if (c["task"], c["cap_kib"]) == ("HAR", 512)][0]
        self.assertFalse(other["sign_change"])
        self.assertAlmostEqual(other["diff_pp"], 0.0, places=9)

    def test_refuses_overwrite(self):
        t = self.tree((1,), lambda c, s: 0.0)
        out = pathlib.Path(t.tmp) / "exists.json"
        out.write_text("{}")
        p = subprocess.run([sys.executable, str(SCRIPT), "--design", "seed1", "--out", str(out),
                            "--root", str(t.root)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertEqual(out.read_text(), "{}")


class SpecCountTests(unittest.TestCase):
    def test_spec_counts(self):
        spec = importlib.util.spec_from_file_location("mts", HERE / "make_transfer_specs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for design, n, n_ck in (("seed1", 22, 18), ("seeds123", 66, 54), ("precondE", 2, 0)):
            specs = [s for s, _ in m.build(design)]
            self.assertEqual(len(specs), n)
            self.assertEqual(sum("--checkpointing" in s["args"] for s in specs), n_ck)
            est = [e for _, e in m.build(design)]
            self.assertEqual(est, sorted(est, reverse=True))


if __name__ == "__main__":
    unittest.main()
