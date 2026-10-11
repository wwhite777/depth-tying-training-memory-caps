"""S3-P tests for src/lb/predictor.py and src/lb/predictor_sim.py (the lifetime-aware accounting, its baselines and
CLIs).

Run (CPU, one thread, one process):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest test/lb/test_predictor.py -v
Configurations measured in process: calibration grid only (widths 40, 56); no registry configuration is profiled,
trained or compared with a prediction. CLI tests use temporary record directories and paths (lb.freeze.RECORD_DIR is
patched); nothing is written under research/s3/. Every check has an in-file negative control showing it can fail.
"""
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from lb import freeze, models, predictor, predictor_sim, prereg, profiler, registry, train  # noqa: E402

P = prereg.load()
REPORT = {}
CAL = {c["config_id"]: c for task in P["tasks"] for c in registry.calibration_grid(task)}
REG = {c["config_id"] for task in P["tasks"] for c in registry.enumerate(task)}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def _measure(cid, m, ck):
    if cid not in CAL or cid in REG:
        raise AssertionError(f"{cid}: only calibration-grid configurations are measured")
    return profiler.measure_peak(cid, m, ck)


def _closed_form_a_static(task, structure, d, T, b, ckpt):
    """Hand count of B3's A_static (bytes): inputs, targets and the storages autograd saves (no transients)."""
    tc = prereg.task_cfg(task, P)
    n, k, C = tc["n_tokens"], tc["token_dim"], tc["classes"]
    bnd, bn = b * n * d, b * n
    inputs = 4 * b * n * k + 8 * b                       # X[idx] float32 + y[idx] int64
    head = 4 * (bnd + 2 * bn + b * d + b * C + 1) + 8   # head LN input, mean, rstd; fc mat1; log-probs; total_weight; wrapped b
    if structure in predictor_sim.REVERSIBLE:
        return inputs + head                             # the saved stack output is the head LN input (one storage)
    if ckpt:
        return inputs + head + 4 * T * bnd               # T checkpointed block inputs (embed out .. block T-1 out)
    per_block = 4 * (10 * bnd + 4 * bn)                  # z, clone, tok1 out, gelu, u, ln2 out (6 bnd) + 2 x 2bnd; 4 x mean/rstd
    return inputs + head + T * per_block                 # per block: its input z; the head LN saves block T's output


class TestShapesAndPurity(unittest.TestCase):
    def test_param_shapes_equal_model_all_calibration(self):
        n, mism = 0, []
        for cid, c in CAL.items():
            tc = prereg.task_cfg(c["task"], P)
            with torch.device("meta"):
                m = models.build_model(tc, c["structure"], c["d"], c["T"])
            got = [tuple(q.shape) for q in m.parameters()]
            want = [tuple(s) for _, s in predictor.param_shapes(c)]
            n += 1
            if got != want or predictor.n_params(c) != c["n_params"]:
                mism.append(cid)
        self.assertEqual(n, 360)
        self.assertEqual(mism, [])
        # negative control: the comparison distinguishes a different width and a different depth
        c = CAL["HAR|untied|d40|T2|sgd"]
        with torch.device("meta"):
            m = models.build_model(prereg.task_cfg("HAR", P), "untied", 40, 2)
        got = [tuple(q.shape) for q in m.parameters()]
        self.assertNotEqual(got, [tuple(s) for _, s in predictor.param_shapes(dict(c, d=56))])
        self.assertNotEqual(got, [tuple(s) for _, s in predictor.param_shapes(dict(c, T=4))])
        REPORT["param_shapes"] = {"configs_checked": n, "mismatches": mism}

    def test_purity_no_model_autograd_optimizer_profiler(self):
        boom = mock.Mock(side_effect=AssertionError("forbidden at prediction time"))
        patches = [mock.patch.object(models.MixerNet, "forward", boom),
                   mock.patch.object(torch.autograd, "backward", boom),
                   mock.patch.object(torch.optim.Optimizer, "step", boom),
                   mock.patch.object(torch.optim.Adam, "step", boom),
                   mock.patch.object(torch.optim.SGD, "step", boom),
                   mock.patch.object(torch.optim.Adafactor, "step", boom),
                   mock.patch.object(torch.profiler, "profile", boom),
                   mock.patch.object(profiler, "profile", boom)]
        for pt in patches:
            pt.start()
        try:
            r1 = predictor.predict("HAR|untied|d40|T2|adam", 8, True)
            r2 = predictor.predict_min("KWS|reversible_shared|d56|T2|adafactor")
            # negative control: under the same patches the measured object cannot be produced
            with self.assertRaises(AssertionError):
                profiler.measure_peak("HAR|untied|d40|T2|adam", 8, True)
        finally:
            for pt in patches:
                pt.stop()
        self.assertEqual(boom.call_count, 1)                 # only the control touched a patched function
        # a fresh interpreter: the accounting never imports torch
        code = ("import sys; from lb import predictor; r = predictor.predict_min('CIFAR10|shared|d40|T2|sgd'); "
                "print(int('torch' in sys.modules), r['min_peak_bytes'])")
        ctrl = "import sys; from lb import registry; print(int('torch' in sys.modules), 0)"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(ROOT / "src"))
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
        out_c = subprocess.run([sys.executable, "-c", ctrl], cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.split()[0], "0")
        self.assertEqual(out_c.stdout.split()[0], "1")      # negative control: importing the harness loads torch
        REPORT["purity"] = {"predict_peak": r1["peak_bytes"], "predict_min": r2["min_peak_bytes"],
                            "torch_imported_by_predictor": out.stdout.split()[0],
                            "control_torch_imported_by_registry": out_c.stdout.split()[0]}

    def test_knob_order_and_tie_rule(self):
        self.assertEqual(len(predictor.knobs("HAR|reversible_untied|d40|T2|sgd")), 7)
        ks = predictor.knobs("HAR|untied|d40|T2|sgd")
        self.assertEqual(ks[:7], [(m, False) for m in P["memory_knobs"]["microbatch"]])
        self.assertEqual(ks[7:], [(m, True) for m in P["memory_knobs"]["microbatch"]])
        rows = [{"microbatch": 1, "checkpointing": True, "peak_bytes": 5}, {"microbatch": 2, "checkpointing": True,
                "peak_bytes": 5}, {"microbatch": 1, "checkpointing": False, "peak_bytes": 5}]
        best = predictor._argmin(rows, "peak_bytes")
        self.assertEqual((best["microbatch"], best["checkpointing"]), (1, False))
        # negative control: without the tie rule the first row would win
        self.assertNotEqual(min(rows, key=lambda r: r["peak_bytes"]), best)


class TestBaselines(unittest.TestCase):
    def test_b1_hand_checked(self):
        # HAR untied d40 T2: embed 40*72+40; block 2*40 + 2*(16*16+16) + 2*40 + (80*40+80) + (40*80+40); head 80 + 6*40+6
        p_har = (40 * 72 + 40) + 2 * (80 + 2 * 272 + 80 + 3280 + 3240) + (80 + 246)
        # KWS reversible_shared d40 T2: one coupling of two d/2 = 20 blocks; n = 49, token_dim 10, 12 classes
        blk20 = 40 + 2 * (49 * 49 + 49) + 40 + (40 * 20 + 40) + (20 * 40 + 20)
        p_kws = (40 * 10 + 40) + 2 * blk20 + (80 + 12 * 40 + 12)
        self.assertEqual(predictor.param_count_with_state_multiplier("HAR|untied|d40|T2|adam"), 4 * p_har * 4)
        self.assertEqual(predictor.param_count_with_state_multiplier("KWS|reversible_shared|d40|T2|sgd"), 4 * p_kws * 2)
        self.assertEqual(predictor.param_count_with_state_multiplier("KWS|reversible_shared|d40|T2|adafactor"),
                         4 * p_kws * 2)
        # negative control: the multiplier depends on the optimizer
        self.assertNotEqual(predictor.param_count_with_state_multiplier("HAR|untied|d40|T2|sgd_momentum"),
                            4 * p_har * 4)
        REPORT["B1"] = {"HAR untied d40 T2 params": p_har, "KWS reversible_shared d40 T2 params": p_kws}

    def test_b3_hand_checked(self):
        cases = [("HAR", "untied", 40, 2, "adam", 64, False), ("HAR", "shared", 40, 2, "sgd", 8, True),
                 ("KWS", "reversible_untied", 40, 2, "sgd_momentum", 4, False)]
        out = {}
        for task, s, d, T, opt, m, ck in cases:
            cid = f"{task}|{s}|d{d}|T{T}|{opt}"
            a = predictor.a_static(cid, m, ck)
            want = _closed_form_a_static(task, s, d, T, m, ck)
            self.assertEqual(a, want, cid)
            self.assertEqual(predictor.static_four_component_sum(cid, m, ck),
                             predictor.persistent_weights_gradients_optimizer(cid) + want)
            out[f"{cid} m{m} ckpt{int(ck)}"] = a
        # negative controls: checkpointing and the microbatch change A_static; the closed form is not trivially equal
        self.assertNotEqual(predictor.a_static("HAR|shared|d40|T2|sgd", 8, False),
                            predictor.a_static("HAR|shared|d40|T2|sgd", 8, True))
        self.assertNotEqual(predictor.a_static("HAR|untied|d40|T2|adam", 32, False),
                            _closed_form_a_static("HAR", "untied", 40, 2, 64, False))
        REPORT["B3_A_static"] = out

    def test_b2_equals_profiler_wgo(self):
        out = {}
        for cid in ("HAR|untied|d40|T2|adam", "KWS|reversible_shared|d56|T2|adafactor", "CIFAR10|two_groups|d40|T4|sgd_momentum"):
            r = _measure(cid, 64, False)
            wgo = r["W_bytes"] + r["G_bytes"] + r["O_bytes"]
            b2 = predictor.persistent_weights_gradients_optimizer(cid)
            self.assertEqual(b2, wgo, cid)
            self.assertEqual(b2, r["baseline_bytes"], cid)
            out[cid] = {"B2": b2, "measured_W+G+O": wgo}
        # negative control: B2 of another optimizer differs from the measured adam state
        self.assertNotEqual(predictor.persistent_weights_gradients_optimizer("HAR|untied|d40|T2|sgd"),
                            out["HAR|untied|d40|T2|adam"]["measured_W+G+O"])
        REPORT["B2"] = out


def _synthetic_profiles(dirpath, drop=None, add=None, bad_code=None, bad_prereg=None, no_done=None):
    """Min-over-knobs profile.json files for all 360 calibration configurations (synthetic peaks; the CLI checks)."""
    code = train.code_hashes()
    ids = sorted(CAL) + ([add] if add else [])
    for i, cid in enumerate(ids):
        if cid == drop:
            continue
        ms = [{"microbatch": m, "checkpointing": ck, "peak_bytes": 300_000 + 1000 * i + 37 * j}
              for j, (m, ck) in enumerate(predictor.knobs(cid))]
        best = predictor._argmin(ms, "peak_bytes")
        prof = {"config_id": cid, "measurements": ms, "min_peak_bytes": best["peak_bytes"],
                "argmin_knob": {"microbatch": best["microbatch"], "checkpointing": best["checkpointing"]},
                "prereg_sha256": prereg.yaml_sha256() if cid != bad_prereg else "0" * 64,
                "code_sha256": dict(code, **({"train.py": "0" * 64} if cid == bad_code else {}))}
        d = pathlib.Path(dirpath) / cid.replace("|", "__")
        d.mkdir(parents=True)
        (d / "profile.json").write_text(json.dumps(prof))
        if cid != no_done:
            (d / "DONE.json").write_text(json.dumps({"profile_sha256": predictor._sha256(d / "profile.json")}))


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.rec_dir = self.root / "records"
        self.patch = mock.patch.object(freeze, "RECORD_DIR", self.rec_dir)
        self.patch.start()
        self.assertTrue(str(freeze.record_path("harness")).startswith(self.tmp.name))
        freeze.write_record("harness")

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def _calibrate(self, prof_dir):
        out = self.root / "cal.json"
        rc = predictor.main(["calibrate", "--profiles", str(prof_dir), "--out", str(out)])
        return rc, out

    def test_calibrate_fails_closed(self):
        res = {}
        for label, kw in (("wrong_code_hash", {"bad_code": "HAR|untied|d40|T2|sgd"}),
                          ("missing_configuration", {"drop": "KWS|shared|d56|T8|adam"}),
                          ("registry_configuration_in_input", {"add": "HAR|untied|d32|T4|adam"}),
                          ("wrong_prereg_hash", {"bad_prereg": "CIFAR10|untied|d40|T2|sgd"}),
                          ("profile_without_done", {"no_done": "HAR|shared|d40|T2|sgd"})):
            pdir = self.root / label
            _synthetic_profiles(pdir, **kw)
            rc, out = self._calibrate(pdir)
            self.assertEqual(rc, 2, label)
            self.assertFalse(out.exists(), label)
            res[label] = rc
        # negative control (the gate can pass): the complete valid set calibrates and writes once
        pdir = self.root / "valid"
        _synthetic_profiles(pdir)
        rc, out = self._calibrate(pdir)
        self.assertEqual(rc, 0)
        rec = json.loads(out.read_text())
        self.assertEqual((rec["n_profiles"], rec["n_fitted_constants"]), (360, 0))
        self.assertEqual(rec["errors"]["per_knob"]["n"], sum(len(predictor.knobs(c)) for c in CAL))
        rc2, _ = self._calibrate(pdir)                      # open("x"): an existing output is never overwritten
        self.assertEqual(rc2, 2)
        res["valid_set_exit"] = rc
        res["rewrite_exit"] = rc2
        REPORT["calibrate_cli"] = res

    def test_predict_fails_closed_without_predictor_record(self):
        cal = self.root / "cal_stub.json"
        cal.write_text(json.dumps({"prereg_sha256": prereg.yaml_sha256(), "constants": {}}))
        out = self.root / "pred.json"
        rc = predictor.main(["predict", "--calibration", str(cal), "--out", str(out)])
        self.assertEqual(rc, 2)
        self.assertFalse(out.exists())
        # a predictor record that omits the calibration file is refused as well
        freeze.write_record("predictor")
        with self.assertRaises(predictor.PredictorError):
            predictor._predictor_check(cal)
        # negative control (the gate can pass): records with the calibration file listed verify
        freeze.record_path("predictor").unlink()
        freeze.write_record("predictor", [cal])
        recs = predictor._predictor_check(cal)
        self.assertEqual(sorted(recs), ["harness", "predictor"])
        REPORT["predict_cli"] = {"exit_without_predictor_record": rc, "gate_passes_with_records": sorted(recs)}


class TestAccuracySubset(unittest.TestCase):
    CASES = [("HAR|untied|d40|T2|adam", 64, False), ("HAR|shared|d56|T4|adafactor", 1, True),
             ("HAR|two_groups|d40|T4|sgd_momentum", 8, True), ("KWS|reversible_shared|d40|T4|adam", 1, False),
             ("KWS|wide_single|d56|T1|sgd", 2, False), ("KWS|untied|d56|T2|adafactor", 16, True),
             ("CIFAR10|reversible_untied|d56|T2|sgd_momentum", 4, False), ("CIFAR10|shared|d40|T8|adam", 32, True),
             ("CIFAR10|untied|d40|T4|sgd", 1, True)]

    def test_accuracy_in_process(self):
        errs, wrong, out = [], [], {}
        for cid, m, ck in self.CASES:
            r = _measure(cid, m, ck)
            pr = predictor.predict(cid, m, ck)
            e = (pr["peak_bytes"] - r["peak_bytes"]) / r["peak_bytes"]
            errs.append(e)
            # negative control: B2 (no activations, no transients) as a predictor of the same peak
            wrong.append((predictor.persistent_weights_gradients_optimizer(cid) - r["peak_bytes"]) / r["peak_bytes"])
            out[f"{cid} m{m} ckpt{int(ck)}"] = {"measured": r["peak_bytes"], "predicted": pr["peak_bytes"],
                                                "rel_error": e, "measured_phase": r["peak_phase"],
                                                "predicted_phase": pr["peak_phase"]}
        med = statistics.median(abs(e) for e in errs)
        med_wrong = statistics.median(abs(e) for e in wrong)
        self.assertLessEqual(med, 0.005)
        self.assertGreater(med_wrong, 0.005)               # the same check fails for the B2 control
        REPORT["accuracy_subset"] = {"cases": out, "median_abs_rel_error": med, "max_abs_rel_error":
                                     max(abs(e) for e in errs), "control_B2_median_abs_rel_error": med_wrong}


if __name__ == "__main__":
    unittest.main()
