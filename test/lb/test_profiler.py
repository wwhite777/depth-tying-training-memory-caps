"""S3 tests for src/lb/profiler.py (card S3-T): the S0 method ported to the harness's training step.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_profiler.py -v
Configurations: calibration grid only (widths 40, 56; HAR shapes; pointer-shared traces on all three tasks), plus the pointer-shared sanity construct; no
registry configuration is profiled (the registry guard of the CLI is tested with the measurement functions patched
out and code-freeze records only in a temporary, patched RECORD_DIR). Every check has an in-file negative control
showing it can fail.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from lb import freeze, prereg, profiler, train  # noqa: E402
from lb.models import build_model  # noqa: E402

P = prereg.load()
HAR = prereg.task_cfg("HAR", P)
KNOWN_BYTES = 262_144
PLANT_BYTES = 65_536
REPORT = {}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def _alloc_profile(nbytes):
    profiler.set_threads()
    profiler._warm_profiler()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=False) as prof:
        with record_function("ph_known"):
            t = torch.empty(nbytes // 4, dtype=torch.float32)
            del t
    return profiler.reconstruct(prof, 0)


class TestInstrument(unittest.TestCase):
    def test_known_size_proof(self):
        r = _alloc_profile(KNOWN_BYTES)
        self.assertEqual((r["peak_above_baseline"], r["delta_end"], r["peak_phase"]), (KNOWN_BYTES, 0, "ph_known"))
        # negative control: a different size reconstructs to that size, so the equality above discriminates
        r2 = _alloc_profile(KNOWN_BYTES + 64)
        self.assertNotEqual(r2["peak_above_baseline"], KNOWN_BYTES)
        REPORT["known_size"] = {"requested": KNOWN_BYTES, "reconstructed_peak": r["peak_above_baseline"],
                                "end_delta": r["delta_end"], "control_262208": r2["peak_above_baseline"]}

    def test_planted_tensor_detected(self):
        """A 65,536-byte tensor created in the profiled step's forward and kept alive through the step."""
        res = {}
        for label in ("clean", "planted"):
            torch.manual_seed(0)
            model = build_model(HAR, "untied", 40, 2)
            calls = {"n": 0}

            def hook(mod, inp, out):
                calls["n"] += 1
                if label == "planted" and calls["n"] == 2:      # call 1 = warm-up update; call 2 = profiled update
                    mod._planted = torch.zeros(PLANT_BYTES // 4)

            model.embed.register_forward_hook(hook)
            r = profiler.measure_model(model, "sgd", HAR, 64)
            res[label] = {k: r[k] for k in ("peak_bytes", "peak_phase", "closure_error", "baseline_bytes")}
        c, p = res["clean"], res["planted"]
        self.assertEqual(c["closure_error"], 0)                      # clean step closes exactly
        self.assertEqual(p["closure_error"], PLANT_BYTES)            # negative control: the planted defect is flagged
        self.assertGreaterEqual(p["peak_bytes"] - c["peak_bytes"], PLANT_BYTES)
        REPORT["planted"] = {**res, "peak_rise": p["peak_bytes"] - c["peak_bytes"]}

    def test_closure_zero_all_optimizers_and_knobs(self):
        out = {}
        for opt in P["registry"]["optimizers"]:
            for m, ck in ((64, False), (8, True), (1, False)):
                r = profiler.measure_peak(f"HAR|shared|d40|T2|{opt}", m, ck)
                out[f"{opt} m={m} ckpt={ck}"] = (r["closure_error"], r["peak_bytes"], r["peak_phase"])
                self.assertEqual(r["closure_error"], 0)
                self.assertEqual(r["baseline_bytes"], r["W_bytes"] + r["G_bytes"] + r["O_bytes"])
        REPORT["closure"] = out


class TestStructures(unittest.TestCase):
    PAIRS = (("HAR|shared|d40|T4|adam", "HAR|pointer_shared|d40|T4|adam", 64),
             ("HAR|shared|d56|T2|sgd_momentum", "HAR|pointer_shared|d56|T2|sgd_momentum", 8),
             ("KWS|shared|d40|T8|adafactor", "KWS|pointer_shared|d40|T8|adafactor", 16),
             ("CIFAR10|shared|d56|T8|sgd", "CIFAR10|pointer_shared|d56|T8|sgd", 64))

    def test_pointer_shared_trace_equals_shared(self):
        out = {}
        for a, b, m in self.PAIRS:
            ra, rb = profiler.measure_peak(a, m, False), profiler.measure_peak(b, m, False)
            self.assertEqual(ra["peak_bytes"], rb["peak_bytes"])
            self.assertEqual(ra["alloc_trace"], rb["alloc_trace"])
            out[f"{a} vs pointer m={m}"] = {"peak": ra["peak_bytes"], "pointer_peak": rb["peak_bytes"],
                                            "trace_len": len(ra["alloc_trace"]), "identical": True}
        # negative control: shared vs untied at the same d and T are distinguishable
        ru = profiler.measure_peak("HAR|untied|d40|T4|adam", 64, False)
        rs = profiler.measure_peak("HAR|shared|d40|T4|adam", 64, False)
        self.assertNotEqual(ru["alloc_trace"], rs["alloc_trace"])
        out["control shared vs untied d40 T4 adam"] = {"shared": rs["peak_bytes"], "untied": ru["peak_bytes"]}
        REPORT["pointer_vs_shared"] = out

    def test_determinism(self):
        r1 = profiler.measure_peak("HAR|two_groups|d40|T4|adafactor", 16, False)
        r2 = profiler.measure_peak("HAR|two_groups|d40|T4|adafactor", 16, False)
        self.assertEqual(r1["peak_bytes"], r2["peak_bytes"])
        self.assertEqual(r1["alloc_trace"], r2["alloc_trace"])
        # negative control: another knob gives another peak
        self.assertNotEqual(profiler.measure_peak("HAR|two_groups|d40|T4|adafactor", 8, False)["peak_bytes"],
                            r1["peak_bytes"])
        REPORT["determinism"] = {"peak": r1["peak_bytes"], "repeat": r2["peak_bytes"]}

    def test_checkpointing_lowers_peak(self):
        out = {}
        for cid in ("HAR|untied|d40|T4|sgd", "HAR|shared|d56|T8|adam"):
            off = profiler.measure_peak(cid, 64, False)["peak_bytes"]
            on = profiler.measure_peak(cid, 64, True)["peak_bytes"]
            self.assertLessEqual(on, off, cid)
            self.assertFalse(off <= on and off != on)            # negative control: the reverse order is false
            out[cid] = {"off": off, "on": on}
        REPORT["checkpointing"] = out

    def test_reversible_peak_flat_in_T(self):
        d, m = 40, 64
        pk = {}
        for s in ("reversible_shared", "shared", "reversible_untied", "untied"):
            for T in (2, 8):
                r = profiler.measure_peak(f"HAR|{s}|d{d}|T{T}|sgd", m, False)
                pk[(s, T)] = r
        g = {s: pk[(s, 8)]["peak_bytes"] - pk[(s, 2)]["peak_bytes"] for s in ("reversible_shared", "shared",
                                                                              "reversible_untied", "untied")}
        nonpersist = {s: (pk[(s, 8)]["peak_bytes"] - pk[(s, 8)]["baseline_bytes"])
                      - (pk[(s, 2)]["peak_bytes"] - pk[(s, 2)]["baseline_bytes"]) for s in g}
        self.assertLess(g["reversible_shared"], g["shared"])
        self.assertLess(g["reversible_untied"], g["untied"])
        self.assertLess(nonpersist["reversible_untied"], nonpersist["untied"])
        self.assertFalse(g["shared"] < g["reversible_shared"])    # negative control: the reverse order is false
        stored_output = m * HAR["n_tokens"] * d * 4
        REPORT["reversible_T"] = {
            "peaks": {f"{s} T{T}": pk[(s, T)]["peak_bytes"] for (s, T) in pk},
            "baselines_WGO": {f"{s} T{T}": pk[(s, T)]["baseline_bytes"] for (s, T) in pk},
            "growth_T2_to_T8": g, "growth_above_persistent_W_G_O": nonpersist,
            "stored_output_bytes": stored_output}


class TestKnobsAndCLI(unittest.TestCase):
    def test_min_over_knobs(self):
        r = profiler.min_over_knobs("HAR|reversible_shared|d40|T2|sgd")
        self.assertEqual(len(r["measurements"]), len(P["memory_knobs"]["microbatch"]))   # no checkpointing
        self.assertEqual(r["min_peak_bytes"], min(x["peak_bytes"] for x in r["measurements"]))
        r2 = profiler.min_over_knobs("HAR|untied|d40|T2|sgd")
        self.assertEqual(len(r2["measurements"]), 2 * len(P["memory_knobs"]["microbatch"]))
        self.assertEqual(r2["min_peak_bytes"], min(x["peak_bytes"] for x in r2["measurements"]))
        REPORT["min_over_knobs"] = {"reversible_shared d40 T2 sgd": [r["min_peak_bytes"], r["argmin_knob"]],
                                    "untied d40 T2 sgd": [r2["min_peak_bytes"], r2["argmin_knob"]],
                                    "untied all": [(x["microbatch"], x["checkpointing"], x["peak_bytes"])
                                                   for x in r2["measurements"]]}

    def test_cli_writes_and_refuses(self):
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONPATH=str(ROOT / "src"))
            cmd = [sys.executable, "-m", "lb.profiler", "--config-id", "HAR|shared|d40|T2|sgd", "--microbatch", "64",
                   "--out", d]
            cp = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp.returncode, 0, cp.stderr.decode()[-2000:])
            j = json.loads((Path(d) / "profile.json").read_text())
            self.assertTrue((Path(d) / "DONE.json").exists())
            inproc = profiler.measure_peak("HAR|shared|d40|T2|sgd", 64, False)
            self.assertEqual(j["peak_bytes"], inproc["peak_bytes"])      # subprocess == in-process measurement
            self.assertEqual(j["threads"], {"intra_op": 1, "inter_op": 1})
            self.assertIsNone(j["code_freeze"])                         # calibration grid: not guarded at run time
            cp2 = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True)
            self.assertEqual(cp2.returncode, 3)
            REPORT["cli"] = {"peak_subprocess": j["peak_bytes"], "peak_in_process": inproc["peak_bytes"],
                             "refuse_exit": cp2.returncode}



class TestFailClosedAndGuard(unittest.TestCase):
    CID = "HAR|untied|d40|T2|sgd"

    def test_measure_peak_fails_closed_on_planted_leak(self):
        """A model returned through a patched build_from_config keeps a 65,536-byte tensor alive from the profiled
        step's forward: measure_peak raises ProfileClosureError (measure_model still returns the value)."""
        real = profiler.build_from_config

        def build(plant):
            def f(cfg, task_cfg, checkpointing=False):
                m = real(cfg, task_cfg, checkpointing)
                calls = {"n": 0}

                def hook(mod, inp, out):
                    calls["n"] += 1
                    if plant and calls["n"] == 2:      # call 1 = warm-up update; call 2 = profiled update
                        mod._planted = torch.zeros(PLANT_BYTES // 4)

                m.embed.register_forward_hook(hook)
                return m
            return f

        self.assertTrue(issubclass(profiler.ProfileClosureError, RuntimeError))
        with mock.patch.object(profiler, "build_from_config", side_effect=build(True)):
            with self.assertRaises(profiler.ProfileClosureError) as cm:
                profiler.measure_peak(self.CID, 64, False)
        self.assertIn(str(PLANT_BYTES), str(cm.exception))
        # negative control: the same patched path without the plant closes and does not raise
        with mock.patch.object(profiler, "build_from_config", side_effect=build(False)):
            r = profiler.measure_peak(self.CID, 64, False)
        self.assertEqual(r["closure_error"], 0)
        REPORT["fail_closed"] = {"planted": str(cm.exception), "clean_closure_error": r["closure_error"]}

    def _main(self, argv):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = profiler.main(argv)
        return rc, err.getvalue()

    def test_cli_registry_guard(self):
        """A registry-member configuration is measured only after verify("harness", "predictor"); the measurement
        functions are patched out, so no registry configuration is profiled."""
        reg = "HAR|shared|d32|T4|adam"
        fake = {"peak_bytes": 1, "alloc_trace": [], "closure_error": 0}
        res = {}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(freeze, "RECORD_DIR", Path(d) / "records"), \
                mock.patch.object(profiler, "measure_peak", side_effect=AssertionError("measured")), \
                mock.patch.object(profiler, "min_over_knobs", side_effect=AssertionError("measured")):
            rc, err = self._main(["--config-id", reg, "--microbatch", "64", "--out", f"{d}/a"])
            res["no_records"] = rc
            self.assertEqual(rc, 2)
            self.assertIn("CodeFreezeError", err)
            self.assertFalse(Path(d, "a").exists())
            freeze.write_record("harness")
            res["harness_only"], _ = self._main(["--config-id", reg, "--out", f"{d}/b"])
            self.assertEqual(res["harness_only"], 2)
            calls = []

            def spy(*names):
                calls.append(names)
                raise freeze.CodeFreezeError("spy")

            with mock.patch.object(freeze, "verify", side_effect=spy):
                self._main(["--config-id", reg, "--out", f"{d}/c"])
            self.assertEqual(calls, [("harness", "predictor")])
        # positive control with the measurement patched to a stub: verified records pass and are recorded; a
        # calibration configuration does not call verify
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(profiler, "measure_peak", return_value=dict(fake)), \
                mock.patch.object(freeze, "verify", return_value={"harness": "h", "predictor": "p"}) as v:
            rc, err = self._main(["--config-id", reg, "--microbatch", "64", "--out", f"{d}/r"])
            self.assertEqual(rc, 0, err)
            self.assertEqual(json.loads(Path(d, "r", "profile.json").read_text())["code_freeze"],
                             {"harness": "h", "predictor": "p"})
            v.reset_mock()
            rc, err = self._main(["--config-id", "HAR|shared|d40|T2|adam", "--microbatch", "64", "--out", f"{d}/k"])
            self.assertEqual(rc, 0, err)
            v.assert_not_called()
            self.assertIsNone(json.loads(Path(d, "k", "profile.json").read_text())["code_freeze"])
        REPORT["registry_guard"] = res


if __name__ == "__main__":
    unittest.main()
