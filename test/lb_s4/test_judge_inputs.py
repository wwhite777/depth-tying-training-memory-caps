"""Tests for src/lb_s4/judge_inputs.py (card S3-A) and the end-to-end synthetic chain:
fixtures -> select_cells -> specs4 confirm / top_up -> synthetic confirm runs -> judge_inputs n1 and n2 ->
`python src/lb/judge.py --n1 --n2 --out` (subprocess). Freeze records in a patched temporary RECORD_DIR; the judge
reads only the real frozen YAML. Nothing is trained or profiled.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m unittest test/lb_s4/test_judge_inputs.py -v
End-to-end cases (designed dispositions): GO (tied +10 points, noise +-0.5), KILL (tied -5 points), MISS_TOP_UP_REQUIRED
(tied + floor exactly, noise +-2, seeds 1-10), INCONCLUSIVE (the same plus top-up seeds 11-12). Every ERROR case is a
negative control of the happy path (exit 2, no output file).
"""
import contextlib
import io
import json
import os
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s4fix as F  # noqa: E402
from lb import freeze  # noqa: E402
from lb_s4 import judge_inputs, select_cells, specs4  # noqa: E402

REPORT = {}
FLOORS = {"HAR": 1.0, "KWS": 1.5, "CIFAR10": 2.0}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def quiet(fn, argv):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rc = fn(argv)
    return rc, err.getvalue()


def noise(cid, seed, half):
    return random.Random(f"noise|{cid}|{seed}").uniform(-half, half)


def acc_fn(gain, half):
    """test accuracy = 60 + gain_task (tied class only) + per-(config, seed) noise in [-half, half]."""
    def f(cid, seed):
        e = F.REG[cid]
        g = gain(e["task"]) if e["class"] == "tied" else 0.0
        return 60.0 + g + noise(cid, seed, half)
    return f


CASES = {
    "GO": (acc_fn(lambda t: 10.0, 0.5), False),
    "KILL": (acc_fn(lambda t: -5.0, 0.5), False),
    "MISS_TOP_UP_REQUIRED": (acc_fn(lambda t: FLOORS[t], 2.0), False),
    "INCONCLUSIVE": (acc_fn(lambda t: FLOORS[t], 2.0), True),
}


def spec_runs(specs):
    for s in specs:
        a = dict(zip(s["args"][::2], s["args"][1::2]))
        yield a["--config-id"], float(a["--lr"]), int(a["--seed"])


def write_confirm_from_specs(root, env, specs, test_acc):
    for cid, lr, seed in spec_runs(specs):
        F.dump(F.confirm_path(root, cid, seed), F.run_doc(cid, env, "confirm", seed, lr, 50.0,
                                                          test_acc=test_acc(cid, seed)))


def run_judge(n1, n2, out):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cp = subprocess.run([sys.executable, str(F.ROOT / "src" / "lb" / "judge.py"), "--n1", str(n1), "--n2", str(n2),
                         "--out", str(out)], cwd=F.ROOT, env=env, capture_output=True, text=True, timeout=300)
    return cp.returncode, json.loads(Path(out).read_text()), cp.stdout + cp.stderr


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = F.Env()
        t = cls.env.tmp
        cls.lr = F.write_lr_selected(t / "lr_selected.json", cls.env)
        cls.profiles, cls.runs = t / "profiles", t / "runs"
        F.write_profiles(cls.profiles, cls.env)
        F.write_select_runs(cls.runs, cls.env)
        cls.sel = t / "selection.json"
        rc, err = quiet(select_cells.main, ["--profiles", str(cls.profiles), "--runs", str(cls.runs),
                                            "--lr", str(cls.lr), "--out", str(cls.sel)])
        assert rc == 0, err
        cls.floors = F.write_floors(t / "floor_values.json", cls.env, FLOORS)
        cls.preds = F.write_predictions(t / "predictions_registry.json", cls.env, F.default_peak)
        cls.conf_specs = specs4.confirm_specs(cls.sel, F.P)
        cls.top_specs = specs4.top_up_specs(cls.sel, F.P)
        cls.selected = {cid: lr for cid, lr, _ in spec_runs(cls.conf_specs)}
        cls.confirm = t / "confirm_base"                      # GO data, shared by the error cases
        write_confirm_from_specs(cls.confirm, cls.env, cls.conf_specs, CASES["GO"][0])

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.out = self.env.tmp / "out" / (self.id().split(".")[-1] + ".json")

    def n1(self, confirm=None, sel=None, floors=None, out=None):
        return quiet(judge_inputs.main, ["n1", "--selection", str(sel or self.sel), "--confirm", str(confirm or self.confirm),
                                         "--floors", str(floors or self.floors), "--out", str(out or self.out)])

    def n2(self, preds=None, out=None):
        return quiet(judge_inputs.main, ["n2", "--predictions", str(preds or self.preds), "--profiles",
                                         str(self.profiles), "--out", str(out or self.out)])


class TestEndToEnd(Base):
    def _case(self, name):
        test_acc, top = CASES[name]
        root = self.env.tmp / f"confirm_{name}"
        write_confirm_from_specs(root, self.env, self.conf_specs, test_acc)
        if top:
            write_confirm_from_specs(root, self.env, self.top_specs, test_acc)
        n1p, n2p, jp = (self.env.tmp / f"{name}_{k}.json" for k in ("n1", "n2", "judge"))
        rc, err = self.n1(confirm=root, out=n1p)
        self.assertEqual(rc, 0, err)
        rc, err = self.n2(out=n2p)
        self.assertEqual(rc, 0, err)
        rc, res, log = run_judge(n1p, n2p, jp)
        n1 = json.loads(n1p.read_text())
        return rc, res, log, n1

    def _check(self, name, n1_1):
        rc, res, log, n1 = self._case(name)
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["disposition"], name, log)
        self.assertEqual(res["status"], name, log)
        self.assertEqual(res["n1_1"]["status"], n1_1)
        seeds = F.CONF + (F.TOP if CASES[name][1] else [])
        self.assertEqual(res["n1"]["seed_set"], seeds)
        self.assertEqual(n1["_provenance"]["seed_set"], seeds)
        self.assertEqual(res["n2_1"]["status"], "PASS")
        self.assertEqual(res["n2_2"]["status"], "PASS")
        # cell structure: 21 finite + 3 unbounded; the infeasible cells; the unbounded cells outside the Holm family
        self.assertEqual(len(n1["cells"]), 24)
        cells = {(c["task"], c["cap_kib"]): c for c in n1["cells"]}
        har = cells[("HAR", 256)]
        self.assertIsNone(har["untied_config"])
        self.assertEqual(har["untied_test_acc"], [])
        self.assertEqual(len(har["tied_test_acc"]), len(seeds))
        cif = cells[("CIFAR10", 256)]
        self.assertEqual((cif["tied_config"], cif["untied_config"], cif["tied_test_acc"]), (None, None, []))
        jc = {(c["task"], c["cap_kib"]): c for c in res["n1"]["cells"]}
        for t in F.TASKS:
            u = jc[(t, F.CONTROL)]
            self.assertTrue(u["feasible"])
            self.assertFalse(u["in_holm_family"])
        self.assertEqual(res["n1"]["holm_family_size"], 19)       # 21 finite cells minus HAR 256 and CIFAR10 256
        # paired by seed: tied minus untied of a cell equals the fixture's per-seed difference
        c = cells[("KWS", 645)]
        f = CASES[name][0]
        for i, s in enumerate(c["seeds"]):
            self.assertAlmostEqual(c["tied_test_acc"][i] - c["untied_test_acc"][i],
                                   f(c["tied_config"], s) - f(c["untied_config"], s), places=9)
        REPORT.setdefault("end_to_end", {})[name] = {
            "expected": name, "status": res["status"], "disposition": res["disposition"],
            "n1_1": res["n1_1"]["status"], "seed_set": res["n1"]["seed_set"], "n2_1": res["n2_1"]["status"],
            "n2_2": res["n2_2"]["status"], "holm_family_size": res["n1"]["holm_family_size"],
            "max_upper_bound_minus_floor": max(x["upper_bound_95"] - x["floor"] for x in res["n1"]["cells"]
                                               if x["in_holm_family"]),
            "min_p_plus_holm": min(x["p_plus_holm"] for x in res["n1"]["cells"] if x["in_holm_family"])}

    def test_go(self):
        self._check("GO", "PASS")

    def test_kill(self):
        self._check("KILL", "FAIL")

    def test_miss_top_up_required(self):
        self._check("MISS_TOP_UP_REQUIRED", "MISS")

    def test_inconclusive_after_top_up(self):
        self._check("INCONCLUSIVE", "MISS")

    def test_n2_negative_control(self):
        """Accounting 10 percent above the measurement: N2.1 FAIL through the same chain."""
        q = F.write_predictions(self.env.tmp / "pred_off.json", self.env, lambda c: int(F.default_peak(c) * 1.10))
        n1p, n2p, jp = (self.env.tmp / f"neg_{k}.json" for k in ("n1", "n2", "judge"))
        self.assertEqual(self.n1(out=n1p)[0], 0)
        self.assertEqual(self.n2(preds=q, out=n2p)[0], 0)
        rc, res, log = run_judge(n1p, n2p, jp)
        self.assertEqual(rc, 0, log)
        self.assertEqual(res["n2_1"]["status"], "FAIL")
        self.assertEqual(res["n2_1"]["n_within_5pct"], 0)
        REPORT["n2_negative_control"] = {"n2_1": res["n2_1"]["status"], "share_within_5pct": res["n2_1"]["share_within_5pct"],
                                         "status": res["status"]}

    def test_n2_without_code_freeze_and_measured_values(self):
        d = json.loads(self.preds.read_text())
        del d["code_freeze"]
        q = F.dump(self.env.tmp / "pred_nocf.json", d)
        rc, err = self.n2(preds=q)
        self.assertEqual(rc, 0, err)
        n2 = json.loads(self.out.read_text())
        self.assertEqual(len(n2["configs"]), 744)
        self.assertEqual(n2["caps_kib"], F.CAPS)
        for c in n2["configs"]:
            self.assertEqual(c["measured_min_peak_bytes"], F.default_peak(c["config_id"]))
            self.assertEqual(c["baselines"], {"param_count_with_state_multiplier": 0,
                                              "persistent_weights_gradients_optimizer": 10 ** 9,
                                              "static_four_component_sum": 0})


class TestN1Errors(Base):
    CID = None

    def setUp(self):
        super().setUp()
        self.cid = sorted(self.selected)[0]
        self.path = F.confirm_path(self.confirm, self.cid, F.CONF[0])

    def _error(self, label, contains, *paths, mutate=None, **kw):
        with F.restored(*paths):
            if mutate:
                mutate()
            rc, err = self.n1(**kw)
        self.assertEqual(rc, 2, f"{label}: {err}")
        self.assertFalse(self.out.exists(), label)
        self.assertIn(contains, err, label)
        REPORT.setdefault("n1_errors", {})[label] = err.strip()[-200:]

    def _edit(self, label, contains, **kw):
        self._error(label, contains, self.path, mutate=lambda: F.edit_json(self.path, **kw))

    def test_baseline_ok(self):
        rc, err = self.n1()
        self.assertEqual(rc, 0, err)                              # the shared fixture itself is valid
        d = json.loads(self.out.read_text())
        self.assertEqual(d["_provenance"]["n_confirm_runs"], len(self.selected) * len(F.CONF))
        self.assertEqual(d["prereg_sha256"], F.YAML)
        self.assertEqual(d["floors"], FLOORS)
        REPORT["n1_selected_configs"] = len(self.selected)

    def test_missing(self):
        self._error("missing confirm run", "missing", self.path, mutate=self.path.unlink)

    def test_duplicate(self):
        q = self.confirm / "dup" / "result.json"
        self._error("duplicate confirm run", "duplicate", q, mutate=lambda: F.dump(q, json.loads(self.path.read_text())))
        shutil.rmtree(q.parent, ignore_errors=True)

    def test_extra_config(self):
        other = next(c for c in F.REG if c not in self.selected)
        q = F.confirm_path(self.confirm, other, 1)
        self._error("confirm run of an unselected configuration", "unexpected", q,
                    mutate=lambda: F.dump(q, F.run_doc(other, self.env, "confirm", 1, F.lr_of(other), 50.0, test_acc=50.0)))
        shutil.rmtree(q.parent, ignore_errors=True)

    def _top_paths(self, cids, seeds):
        return [F.confirm_path(self.confirm, c, s) for c in cids for s in seeds]

    def _write_top(self, cids, seeds):
        for c in cids:
            for s in seeds:
                F.dump(F.confirm_path(self.confirm, c, s),
                       F.run_doc(c, self.env, "confirm", s, self.selected[c], 50.0, test_acc=55.0))

    def test_partial_top_up_one_config(self):
        cids = sorted(self.selected)[:1]
        paths = self._top_paths(cids, F.TOP)
        self._error("partial top-up (one configuration)", "partial top-up", *paths,
                    mutate=lambda: self._write_top(cids, F.TOP))
        for p in paths:
            shutil.rmtree(p.parent, ignore_errors=True)

    def test_partial_top_up_one_seed(self):
        cids = sorted(self.selected)
        paths = self._top_paths(cids, F.TOP[:1])
        self._error("partial top-up (seed 11 only, every configuration)", "partial top-up", *paths,
                    mutate=lambda: self._write_top(cids, F.TOP[:1]))
        for p in paths:
            shutil.rmtree(p.parent, ignore_errors=True)

    def test_full_top_up_control(self):
        """Negative control of the partial top-up: every configuration with both top-up seeds is accepted."""
        cids = sorted(self.selected)
        paths = self._top_paths(cids, F.TOP)
        with F.restored(*paths):
            self._write_top(cids, F.TOP)
            rc, err = self.n1()
        for p in paths:
            shutil.rmtree(p.parent, ignore_errors=True)
        self.assertEqual(rc, 0, err)
        self.assertEqual(json.loads(self.out.read_text())["cells"][0]["seeds"], F.CONF + F.TOP)

    def test_wrong_seed(self):
        self._edit("seed outside confirmation + top_up", "seed", seed=13)

    def test_mode(self):
        self._edit("mode select", "mode", mode="select")

    def test_status(self):
        self._edit("status error", "status", status="error")

    def test_test_acc_null(self):
        self._edit("null test_acc", "test_acc", test_acc=None)

    def test_lr(self):
        self._edit("lr not the select-run lr", "select-run lr", lr=self.selected[self.cid] * 3)

    def test_microbatch(self):
        self._edit("microbatch 8", "microbatch", microbatch=8)

    def test_truncated(self):
        self._edit("truncated confirm run", "truncated", truncated=True)

    def test_prereg(self):
        self._edit("run prereg hash", "prereg_sha256", prereg_sha256="0" * 64)

    def test_code(self):
        self._edit("run code hash", "code_sha256", code_sha256=dict(self.env.code, **{"judge.py": "0" * 64}))

    def test_code_freeze(self):
        self._edit("run code_freeze", "code_freeze", code_freeze={"harness": self.env.records["harness"],
                                                                  "predictor": "1" * 64})

    def test_floors_prereg(self):
        q = F.write_floors(self.env.tmp / "fl_bad.json", self.env, FLOORS, prereg_sha256="0" * 64)
        self._error("floors prereg hash", "prereg_sha256", floors=q)

    def test_floors_missing_task(self):
        q = F.write_floors(self.env.tmp / "fl_miss.json", self.env, {"HAR": 1.0, "KWS": 1.0})
        self._error("floors missing a task", "floor_values tasks", floors=q)

    def test_floors_below_one(self):
        q = F.write_floors(self.env.tmp / "fl_low.json", self.env, dict(FLOORS, KWS=0.5))
        self._error("floor below 1.0", ">= 1.0", floors=q)

    def test_floors_code_freeze(self):
        q = F.write_floors(self.env.tmp / "fl_cf.json", self.env, FLOORS, code_freeze={"harness": "2" * 64})
        self._error("floors code_freeze", "code_freeze", floors=q)

    def test_selection_prereg(self):
        d = json.loads(self.sel.read_text())
        d["prereg_sha256"] = "0" * 64
        q = F.dump(self.env.tmp / "sel_bad.json", d)
        self._error("selection prereg hash", "prereg_sha256", sel=q)

    def test_selection_code_freeze(self):
        d = json.loads(self.sel.read_text())
        d["code_freeze"] = {"harness": self.env.records["harness"], "predictor": "3" * 64}
        q = F.dump(self.env.tmp / "sel_cf.json", d)
        self._error("selection code_freeze", "code_freeze", sel=q)

    def test_selection_missing_cell(self):
        d = json.loads(self.sel.read_text())
        d["cells"] = [c for c in d["cells"] if not (c["task"] == "KWS" and c["cap_kib"] == F.CONTROL)]
        q = F.dump(self.env.tmp / "sel_cell.json", d)
        self._error("selection without the KWS unbounded cell", "not exactly", sel=q)

    def test_out_exists(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text("keep")
        rc, err = self.n1()
        self.assertEqual(rc, 2)
        self.assertEqual(self.out.read_text(), "keep")
        self.out.unlink()


class TestN2Errors(Base):
    def _error(self, label, contains, doc=None, paths=(), mutate=None):
        q = None
        if doc is not None:
            q = F.dump(self.env.tmp / f"pred_{label.replace(' ', '_')}.json", doc)
        with F.restored(*paths):
            if mutate:
                mutate()
            rc, err = self.n2(preds=q)
        self.assertEqual(rc, 2, f"{label}: {err}")
        self.assertFalse(self.out.exists(), label)
        self.assertIn(contains, err, label)
        REPORT.setdefault("n2_errors", {})[label] = err.strip()[-200:]

    def doc(self):
        return json.loads(self.preds.read_text())

    def test_baseline_ok(self):
        rc, err = self.n2()
        self.assertEqual(rc, 0, err)

    def test_prereg(self):
        self._error("predictions prereg hash", "prereg_sha256", dict(self.doc(), prereg_sha256="0" * 64))

    def test_code_freeze(self):
        self._error("predictions code_freeze", "code_freeze",
                    dict(self.doc(), code_freeze={"harness": self.env.records["harness"], "predictor": "4" * 64}))

    def test_code_freeze_names(self):
        self._error("predictions code_freeze without predictor", "code_freeze names",
                    dict(self.doc(), code_freeze={"harness": self.env.records["harness"]}))

    def test_caps(self):
        self._error("predictions caps", "caps_kib", dict(self.doc(), caps_kib=F.CAPS[:-1]))

    def test_missing_config(self):
        d = self.doc()
        d["configs"] = d["configs"][1:]
        self._error("prediction missing", "missing 1", d)

    def test_duplicate_config(self):
        d = self.doc()
        d["configs"].append(d["configs"][0])
        self._error("prediction duplicate", "duplicate", d)

    def test_extra_config(self):
        d = self.doc()
        d["configs"].append(dict(d["configs"][0], config_id="HAR|untied|d40|T2|adam"))
        self._error("prediction extra", "extra 1", d)

    def test_bad_baseline(self):
        d = self.doc()
        d["configs"][3]["baselines"]["static_four_component_sum"] = {"argmin_knob": {}}
        self._error("baseline without min_bytes", "static_four_component_sum", d)

    def test_missing_baseline(self):
        d = self.doc()
        del d["configs"][3]["baselines"]["persistent_weights_gradients_optimizer"]
        self._error("baseline missing", "baselines", d)

    def test_negative_prediction(self):
        d = self.doc()
        d["configs"][5]["accounting"]["min_peak_bytes"] = -1
        self._error("negative prediction", "accounting.min_peak_bytes", d)

    def test_missing_profile(self):
        p = self.profiles / specs4.name_of(sorted(F.REG)[0]) / "profile.json"
        self._error("profile missing", "missing", paths=(p,), mutate=p.unlink)

    def test_missing_predictor_record(self):
        rec = freeze.record_path("predictor")
        self._error("predictor record missing", "missing", paths=(rec,), mutate=rec.unlink)


if __name__ == "__main__":
    unittest.main()
