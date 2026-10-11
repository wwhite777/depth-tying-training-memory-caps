"""Tests for src/lb_s4/select_cells.py (card S3-A): synthetic profiles and select runs for the 744 registry
configurations in a temporary directory; freeze records in a patched temporary RECORD_DIR. Nothing is trained.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m unittest test/lb_s4/test_select_cells.py -v
Every ERROR case is a negative control of the happy path (exit 2, no output file); every tie and feasibility check
has its reversed control.
"""
import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s4fix as F  # noqa: E402
from lb import freeze  # noqa: E402
from lb_s4 import select_cells  # noqa: E402
from lb_s4.specs4 import name_of  # noqa: E402

REPORT = {}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = F.Env()
        t = cls.env.tmp
        cls.lr = F.write_lr_selected(t / "lr_selected.json", cls.env)
        cls.profiles, cls.runs = t / "profiles", t / "runs"
        F.write_profiles(cls.profiles, cls.env)
        F.write_select_runs(cls.runs, cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.out = self.env.tmp / "out" / (self.id().split(".")[-1] + ".json")

    def main(self, lr=None):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = select_cells.main(["--profiles", str(self.profiles), "--runs", str(self.runs),
                                    "--lr", str(lr or self.lr), "--out", str(self.out)])
        return rc, err.getvalue()

    def ok(self, **kw):
        rc, err = self.main(**kw)
        self.assertEqual(rc, 0, err)
        d = json.loads(self.out.read_text())
        self.out.unlink()
        return d

    def cell(self, d, task, cap):
        return next(c for c in d["cells"] if c["task"] == task and c["cap_kib"] == cap)

    def prof(self, cid):
        return self.profiles / name_of(cid) / "profile.json"

    def run_(self, cid):
        return self.runs / name_of(cid) / "result.json"


class TestSelection(Base):
    def test_happy_matches_bruteforce(self):
        d = self.ok()
        exp = F.expected_cells()
        self.assertEqual(d["n_cells"], 24)
        self.assertEqual(len(d["cells"]), 24)
        self.assertEqual(d["n_profiles"], 744)
        self.assertEqual(d["n_select_runs"], 744)
        self.assertEqual(len(d["inputs_sha256"]), 744 + 744 + 3)
        self.assertEqual(set(d["code_freeze"]), {"harness", "predictor"})
        for c in d["cells"]:
            for cls in ("tied", "untied"):
                want = exp[(c["task"], c["cap_kib"])][cls]
                got = c[cls]["config_id"] if c[cls] else None
                self.assertEqual(got, want, (c["task"], c["cap_kib"], cls))
                if c[cls]:
                    e = c[cls]
                    self.assertEqual(e["min_peak_bytes"], F.default_peak(want))
                    self.assertEqual(e["lr"], F.lr_of(want))
                    self.assertEqual(e["n_params"], F.REG[want]["n_params"])
                    self.assertEqual(e["final_val_acc"], F.default_acc(want))
                    self.assertEqual(e["argmin_knob"], {"microbatch": 1, "checkpointing": False})
                    if c["cap_kib"] == F.CONTROL:
                        self.assertIsNone(e["slack_bytes"])
                    else:
                        self.assertEqual(e["slack_bytes"], c["cap_kib"] * 1024 - e["min_peak_bytes"])
                        self.assertGreaterEqual(e["slack_bytes"], 0)
        self.assertEqual(len(d["feasibility"]), 3 * 2 * 8)
        for row in d["feasibility"]:
            n = sum(1 for cid, e in F.REG.items() if e["task"] == row["task"] and e["class"] == row["class"]
                    and (row["cap_kib"] == F.CONTROL or F.default_peak(cid) <= row["cap_kib"] * 1024))
            self.assertEqual(row["n_feasible"], n, row)
        REPORT["feasibility_fixture"] = {f"{r['task']}|{r['class']}|{r['cap_kib']}": r["n_feasible"]
                                         for r in d["feasibility"]}
        REPORT["infeasible_cells_fixture"] = [[c["task"], c["cap_kib"], c["infeasible_class"]] for c in d["cells"]
                                              if c["infeasible_class"]]

    def test_infeasible_class_and_control(self):
        d = self.ok()
        c = self.cell(d, "HAR", 256)
        self.assertIsNone(c["untied"])
        self.assertIsNotNone(c["tied"])
        self.assertEqual(c["infeasible_class"], ["untied"])
        row = next(r for r in d["feasibility"] if (r["task"], r["class"], r["cap_kib"]) == ("HAR", "untied", 256))
        self.assertEqual(row["n_feasible"], 0)
        # negative control: one HAR untied configuration brought to exactly 256 KiB becomes feasible and is selected
        cid = min((c for c, e in F.REG.items() if e["task"] == "HAR" and e["class"] == "untied"), key=F.default_peak)
        with F.restored(self.prof(cid)):
            F.dump(self.prof(cid), F.profile_doc(cid, 256 * 1024, self.env))
            c2 = self.cell(self.ok(), "HAR", 256)
        self.assertEqual(c2["untied"]["config_id"], cid)
        self.assertEqual(c2["untied"]["slack_bytes"], 0)
        self.assertEqual(c2["infeasible_class"], [])
        REPORT["infeasible_control"] = {"config": cid, "peak_at_cap": 256 * 1024, "selected": c2["untied"]["config_id"]}

    def test_feasibility_boundary(self):
        """peak == cap x 1024 is feasible; cap x 1024 + 1 is not (the best KWS untied configuration moved)."""
        pool = [c for c, e in F.REG.items() if e["task"] == "KWS" and e["class"] == "untied"]
        best = max(pool, key=F.default_acc)
        obs = {}
        for peak, at512 in ((512 * 1024, True), (512 * 1024 + 1, False)):
            with F.restored(self.prof(best)):
                F.dump(self.prof(best), F.profile_doc(best, peak, self.env))
                d = self.ok()
            sel512 = self.cell(d, "KWS", 512)["untied"]["config_id"]
            sel645 = self.cell(d, "KWS", 645)["untied"]["config_id"]
            self.assertEqual(sel512 == best, at512, peak)
            self.assertEqual(sel645, best)
            obs[peak] = sel512
        REPORT["feasibility_boundary"] = {"config": best, "selected_at_512_by_peak": obs}

    def test_unbounded_cell(self):
        """Every configuration is feasible at the control cap: a best-accuracy configuration above 1024 KiB is selected
        only in the unbounded cell."""
        big = next(c for c, e in F.REG.items() if e["task"] == "CIFAR10" and e["class"] == "untied"
                   and F.default_peak(c) > 1024 * 1024)
        d0 = self.ok()
        self.assertNotEqual(self.cell(d0, "CIFAR10", F.CONTROL)["untied"]["config_id"], big)   # control: not best yet
        with F.restored(self.run_(big)):
            F.edit_json(self.run_(big), final_val_acc=99.5)
            d = self.ok()
        self.assertEqual(self.cell(d, "CIFAR10", F.CONTROL)["untied"]["config_id"], big)
        for cap in F.CAPS:
            c = self.cell(d, "CIFAR10", cap)["untied"]
            self.assertTrue(c is None or c["config_id"] != big, cap)
        rows = [r for r in d["feasibility"] if r["cap_kib"] == F.CONTROL]
        self.assertTrue(all(r["n_feasible"] == r["n_total"] for r in rows))
        self.assertEqual(sorted({r["n_total"] for r in rows}), [120, 128])
        REPORT["unbounded"] = {"config": big, "peak": F.default_peak(big)}

    def test_diverged_run_keeps_accuracy(self):
        pool = [c for c, e in F.REG.items() if e["task"] == "HAR" and e["class"] == "tied"]
        worst = min(pool, key=F.default_acc)
        with F.restored(self.run_(worst)):
            F.edit_json(self.run_(worst), status="diverged", final_val_acc=99.0)
            d = self.ok()
        self.assertEqual(self.cell(d, "HAR", F.CONTROL)["tied"]["config_id"], worst)
        self.assertEqual(self.cell(d, "HAR", F.CONTROL)["tied"]["status"], "diverged")


class TestTies(Base):
    """Each tie-break level, unit level (choose) and through the CLI, each with its reversed control."""

    def _c(self, cid, acc, peak, n):
        return {"config_id": cid, "final_val_acc": acc, "min_peak_bytes": peak, "n_params": n}

    def test_levels_unit(self):
        a, b = "A|x", "B|x"
        cases = {
            "accuracy": ([self._c(a, 80.0, 900, 9), self._c(b, 79.999999, 100, 1)], a,
                         [self._c(a, 79.999999, 900, 9), self._c(b, 80.0, 100, 1)], b),
            "peak": ([self._c(a, 80.0, 100, 9), self._c(b, 80.0, 101, 1)], a,
                     [self._c(a, 80.0, 101, 9), self._c(b, 80.0, 100, 1)], b),
            "n_params": ([self._c(a, 80.0, 100, 1), self._c(b, 80.0, 100, 2)], a,
                         [self._c(a, 80.0, 100, 2), self._c(b, 80.0, 100, 1)], b),
            "config_id": ([self._c(a, 80.0, 100, 1), self._c(b, 80.0, 100, 1)], a,
                          [self._c("C|x", 80.0, 100, 1), self._c(b, 80.0, 100, 1)], b),
        }
        for level, (c1, w1, c2, w2) in cases.items():
            self.assertEqual(select_cells.choose(c1)["config_id"], w1, level)
            self.assertEqual(select_cells.choose(list(reversed(c1)))["config_id"], w1, level)   # order-free
            self.assertEqual(select_cells.choose(c2)["config_id"], w2, level)
        self.assertIsNone(select_cells.choose([]))
        REPORT["tie_levels_unit"] = sorted(cases)

    def _pair(self):
        """Two KWS tied configurations with different n_params, made the top two by accuracy."""
        pool = sorted((c for c, e in F.REG.items() if e["task"] == "KWS" and e["class"] == "tied"),
                      key=lambda c: F.REG[c]["n_params"])
        return pool[0], pool[-1]          # fewest and most parameters

    def test_levels_cli(self):
        small, large = self._pair()
        paths = [self.run_(small), self.run_(large), self.prof(small), self.prof(large)]
        sel = lambda d: self.cell(d, "KWS", F.CONTROL)["tied"]["config_id"]   # noqa: E731
        obs = {}
        with F.restored(*paths):
            F.edit_json(self.run_(small), final_val_acc=99.0)
            F.edit_json(self.run_(large), final_val_acc=99.0)
            # peak level: equal accuracy, lower peak wins (both directions)
            F.dump(self.prof(small), F.profile_doc(small, 300000, self.env))
            F.dump(self.prof(large), F.profile_doc(large, 299999, self.env))
            obs["peak_large_lower"] = sel(self.ok())
            self.assertEqual(obs["peak_large_lower"], large)
            F.dump(self.prof(large), F.profile_doc(large, 300001, self.env))
            obs["peak_small_lower"] = sel(self.ok())
            self.assertEqual(obs["peak_small_lower"], small)
            # n_params level: equal accuracy and peak, fewer parameters wins
            F.dump(self.prof(large), F.profile_doc(large, 300000, self.env))
            obs["equal_peak"] = sel(self.ok())
            self.assertEqual(obs["equal_peak"], small)
            # accuracy level: the larger, higher-peak configuration wins on accuracy
            F.edit_json(self.run_(large), final_val_acc=99.000001)
            obs["acc_large_higher"] = sel(self.ok())
            self.assertEqual(obs["acc_large_higher"], large)
        REPORT["tie_levels_cli"] = {"small": small, "large": large, **obs}

    def test_config_id_level_cli(self):
        """Two KWS tied configurations differing only in optimizer share n_params; with equal accuracy and peak the
        lexicographically smaller config_id wins; reversed control by changing which one is ahead by 1 byte."""
        a = "KWS|shared|d32|T4|adam"
        b = "KWS|shared|d32|T4|sgd"
        self.assertEqual(F.REG[a]["n_params"], F.REG[b]["n_params"])
        sel = lambda d: self.cell(d, "KWS", F.CONTROL)["tied"]["config_id"]   # noqa: E731
        with F.restored(self.run_(a), self.run_(b), self.prof(a), self.prof(b)):
            for cid in (a, b):
                F.edit_json(self.run_(cid), final_val_acc=99.0)
                F.dump(self.prof(cid), F.profile_doc(cid, 300000, self.env))
            self.assertEqual(sel(self.ok()), a)
            F.dump(self.prof(b), F.profile_doc(b, 299999, self.env))
            self.assertEqual(sel(self.ok()), b)


class TestErrors(Base):
    """Each ERROR: exit 2, no output file; fixtures restored after each case."""

    def _error(self, label, contains, *paths, mutate=None, lr=None):
        with F.restored(*paths):
            if mutate:
                mutate()
            rc, err = self.main(lr=lr)
        self.assertEqual(rc, 2, f"{label}: {err}")
        self.assertFalse(self.out.exists(), label)
        if contains:
            self.assertIn(contains, err, label)
        REPORT.setdefault("errors", {})[label] = err.strip()[-200:]

    CID = "KWS|untied|d32|T4|adam"

    def test_missing_profile(self):
        p = self.prof(self.CID)
        self._error("missing profile", "missing", p, mutate=p.unlink)

    def test_duplicate_profile(self):
        p, q = self.prof(self.CID), self.profiles / "dup" / "profile.json"
        self._error("duplicate profile", "duplicate", q, mutate=lambda: F.dump(q, json.loads(p.read_text())))
        shutil.rmtree(q.parent, ignore_errors=True)

    def test_extra_profile(self):
        q = self.profiles / "extra" / "profile.json"
        cal = "HAR|untied|d40|T2|adam"            # a calibration-grid configuration, not a registry member
        self._error("extra profile", "unexpected", q, mutate=lambda: F.dump(
            q, dict(F.profile_doc(self.CID, 300000, self.env), config_id=cal)))
        shutil.rmtree(q.parent, ignore_errors=True)

    def test_missing_run(self):
        p = self.run_(self.CID)
        self._error("missing select run", "missing", p, mutate=p.unlink)

    def test_duplicate_run(self):
        p, q = self.run_(self.CID), self.runs / "dup" / "result.json"
        self._error("duplicate select run", "duplicate", q, mutate=lambda: F.dump(q, json.loads(p.read_text())))
        shutil.rmtree(q.parent, ignore_errors=True)

    def test_extra_run(self):
        q = self.runs / "extra" / "result.json"
        self._error("extra select run", "unexpected", q, mutate=lambda: F.dump(
            q, F.run_doc("HAR|linear", self.env, "select", 0, 0.001, 50.0)))
        shutil.rmtree(q.parent, ignore_errors=True)

    def _edit_err(self, label, contains, path, **kw):
        self._error(label, contains, path, mutate=lambda: F.edit_json(path, **kw))

    def test_profile_prereg(self):
        self._edit_err("profile prereg hash", "prereg_sha256", self.prof(self.CID), prereg_sha256="0" * 64)

    def test_profile_code(self):
        p = self.prof(self.CID)
        self._edit_err("profile code hash", "code_sha256", p, code_sha256=dict(self.env.code, **{"train.py": "0" * 64}))

    def test_profile_code_freeze(self):
        self._edit_err("profile code_freeze", "code_freeze", self.prof(self.CID),
                       code_freeze={"harness": self.env.records["harness"], "predictor": "0" * 64})

    def test_profile_knobs_incomplete(self):
        p = self.prof(self.CID)
        d = json.loads(p.read_text())
        self._edit_err("profile knob set incomplete", "knobs", p, measurements=d["measurements"][:-1])

    def test_profile_min_inconsistent(self):
        self._edit_err("profile min inconsistent", "min over its measurements", self.prof(self.CID),
                       min_peak_bytes=1234)

    def test_profile_argmin_inconsistent(self):
        self._edit_err("profile argmin inconsistent", "argmin_knob", self.prof(self.CID),
                       argmin_knob={"microbatch": 64, "checkpointing": False})

    def test_run_prereg(self):
        self._edit_err("run prereg hash", "prereg_sha256", self.run_(self.CID), prereg_sha256="0" * 64)

    def test_run_code(self):
        self._edit_err("run code hash", "code_sha256", self.run_(self.CID),
                       code_sha256=dict(self.env.code, **{"models.py": "0" * 64}))

    def test_run_code_freeze(self):
        self._edit_err("run code_freeze", "code_freeze", self.run_(self.CID),
                       code_freeze={"harness": self.env.records["harness"]})

    def test_run_mode(self):
        self._edit_err("run mode", "mode", self.run_(self.CID), mode="dev")

    def test_run_status(self):
        self._edit_err("run status", "status", self.run_(self.CID), status="error")

    def test_run_nonfinite(self):
        self._edit_err("run null accuracy", "final_val_acc", self.run_(self.CID), final_val_acc=None)

    def test_run_seed(self):
        self._edit_err("run seed", "seed", self.run_(self.CID), seed=1)

    def test_run_lr(self):
        self._edit_err("run lr not the selected lr", "selected lr", self.run_(self.CID),
                       lr=F.lr_selections()["KWS"]["adam"]["tied"])

    def test_run_microbatch(self):
        self._edit_err("run microbatch", "microbatch", self.run_(self.CID), microbatch=8)

    def test_run_truncated(self):
        self._edit_err("run truncated (max_steps)", "max_steps", self.run_(self.CID), max_steps=5, truncated=True)

    def test_run_n_params(self):
        self._edit_err("run n_params", "n_params", self.run_(self.CID), n_params=1)

    def test_lr_file_prereg(self):
        q = self.env.tmp / "lr_bad.json"
        F.write_lr_selected(q, self.env, prereg_sha256="0" * 64)
        self._error("lr file prereg hash", "prereg_sha256", lr=q)

    def test_lr_file_off_grid(self):
        q = self.env.tmp / "lr_offgrid.json"
        sel = F.lr_selections()
        sel["HAR"]["sgd"]["tied"] = 0.2
        F.write_lr_selected(q, self.env, selections=sel)
        self._error("lr file off grid", "lr_grid", lr=q)

    def test_lr_file_code_freeze(self):
        q = self.env.tmp / "lr_freeze.json"
        F.write_lr_selected(q, self.env, code_freeze={"harness": "0" * 64})
        self._error("lr file code_freeze", "code_freeze", lr=q)

    def test_missing_predictor_record(self):
        rec = freeze.record_path("predictor")
        self._error("missing predictor record", "missing", rec, mutate=rec.unlink)

    def test_out_exists(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text("keep")
        rc, err = self.main()
        self.assertEqual(rc, 2)
        self.assertEqual(self.out.read_text(), "keep")
        self.out.unlink()
        REPORT.setdefault("errors", {})["out exists"] = err.strip()[-200:]


if __name__ == "__main__":
    unittest.main()
