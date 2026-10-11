"""Tests for src/lb_s4/specs4.py (card S3-A): counts, uniqueness and lr provenance of each spec kind; sampled specs
pass lb.launch.read_specs and lb.train.check_guards (guards only: no data is read, nothing is trained). Freeze
records in a patched temporary RECORD_DIR.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python -m unittest test/lb_s4/test_specs4.py -v
"""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import s4fix as F  # noqa: E402
from lb import launch, train  # noqa: E402
from lb_s4 import specs4  # noqa: E402

REPORT = {}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def args_of(spec):
    a = dict(zip(spec["args"][::2], spec["args"][1::2]))
    return a


def selection_doc(env, cells, **override):
    d = {"kind": "cell_selection", "prereg_sha256": F.YAML, "code_freeze": dict(env.records), "cells": cells}
    d.update(override)
    return d


def entry(cid, lr):
    return {"config_id": cid, "lr": lr}


def cells_with_repeats():
    """24 cells; every cell selects one of 3 tied and 3 untied configurations per task (repeats across cells), with
    lrs that differ from the S3 lr table so that provenance is visible; HAR 256 untied is infeasible."""
    cells = []
    for t in F.TASKS:
        tied = [f"{t}|shared|d24|T2|adam", f"{t}|shared|d32|T4|sgd", f"{t}|two_groups|d48|T8|adafactor"]
        untied = [f"{t}|untied|d24|T2|adam", f"{t}|wide_single|d96|T1|sgd_momentum", f"{t}|untied|d64|T8|sgd"]
        for i, cap in enumerate(F.CAPS + [F.CONTROL]):
            u = None if (t, cap) == ("HAR", 256) else entry(untied[i % 3], 0.0123)
            cells.append({"task": t, "cap_kib": cap, "tied": entry(tied[i % 3], 0.0456), "untied": u})
    return cells


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = F.Env()
        cls.lr = F.write_lr_selected(cls.env.tmp / "lr_selected.json", cls.env)
        cls.sel = F.dump(cls.env.tmp / "selection.json", selection_doc(cls.env, cells_with_repeats()))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.out = self.env.tmp / "out" / (self.id().split(".")[-1] + ".jsonl")

    def main(self, *a):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = specs4.main([*a, "--out", str(self.out)])
        return rc, err.getvalue()

    def unique(self, specs):
        self.assertEqual(len({s["run_id"] for s in specs}), len(specs))
        self.assertEqual(len({s["out"] for s in specs}), len(specs))

    def guards(self, specs, k=4):
        """lb.train.check_guards accepts sampled train specs (verifies the patched records; no data read)."""
        for s in specs[:k] + specs[-k:]:
            a = args_of(s)
            cfg = train.parse_config_id(a["--config-id"], F.P)
            rec = train.check_guards(cfg, a["--mode"], float(a["--lr"]), int(a["--seed"]), int(a["--microbatch"]),
                                     False, F.P)
            self.assertEqual(rec, self.env.records)


class TestKinds(Base):
    def test_profile_registry(self):
        specs = specs4.profile_registry_specs()
        self.assertEqual(len(specs), 744)
        self.unique(specs)
        self.assertEqual({args_of(s)["--config-id"] for s in specs}, set(F.REG))
        self.assertTrue(all(s["module"] == "lb.profiler" and s["args"][0] == "--config-id" and len(s["args"]) == 2
                            and s["out"].startswith("result/s4/profile_registry/") for s in specs))
        rc, err = self.main("--kind", "profile_registry")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(launch.read_specs(self.out)), 744)
        REPORT["profile_registry"] = {"n": len(specs), "example": specs[0]}

    def test_select(self):
        specs = specs4.select_specs(self.lr, F.P)
        self.assertEqual(len(specs), 744)
        self.unique(specs)
        n_tied = 0
        for s in specs:
            a = args_of(s)
            cid = a["--config-id"]
            self.assertEqual(float(a["--lr"]), F.lr_of(cid))                  # lr provenance: (task, optimizer, class)
            self.assertEqual((a["--mode"], a["--seed"], a["--microbatch"]), ("select", "0", "64"))
            self.assertEqual(s["module"], "lb.train")
            n_tied += F.REG[cid]["class"] == "tied"
        self.assertEqual(n_tied, 3 * 128)
        self.guards(specs)
        # negative control: swapping the class lrs in lr_selected.json moves every spec's lr
        sw = {t: {o: {"tied": c["untied"], "untied": c["tied"]} for o, c in v.items()} for t, v in F.lr_selections().items()}
        q = F.write_lr_selected(self.env.tmp / "lr_swapped.json", self.env, selections=sw)
        moved = sum(float(args_of(a)["--lr"]) != float(args_of(b)["--lr"])
                    for a, b in zip(specs, specs4.select_specs(q, F.P)))
        self.assertEqual(moved, 744)
        REPORT["select"] = {"n": len(specs), "n_tied": n_tied, "moved_by_swap": moved, "example": specs[0]}

    def test_confirm_dedup_and_provenance(self):
        specs = specs4.confirm_specs(self.sel, F.P)
        distinct = {e["config_id"] for c in cells_with_repeats() for e in (c["tied"], c["untied"]) if e}
        self.assertEqual(len(distinct), 18)                     # 24 cells, 47 entries, 18 distinct
        self.assertEqual(len(specs), 18 * len(F.CONF))
        self.unique(specs)
        for s in specs:
            a = args_of(s)
            want = 0.0456 if F.REG[a["--config-id"]]["class"] == "tied" else 0.0123
            self.assertEqual(float(a["--lr"]), want)            # the select-run lr recorded in the selection
            self.assertEqual(a["--mode"], "confirm")
            self.assertIn(int(a["--seed"]), F.CONF)
            self.assertTrue(s["out"].startswith("result/s4/confirm/"))
        self.assertEqual({args_of(s)["--config-id"] for s in specs}, distinct)
        REPORT["confirm"] = {"cells": 24, "entries": 47, "distinct": len(distinct), "n_specs": len(specs)}

    def test_top_up(self):
        conf = specs4.confirm_specs(self.sel, F.P)
        top = specs4.top_up_specs(self.sel, F.P)
        self.assertEqual(len(top), 18 * len(F.TOP))
        self.unique(top)
        self.unique(conf + top)                                 # same tree, disjoint seeds, no collision
        self.assertEqual({int(args_of(s)["--seed"]) for s in top}, set(F.TOP))
        self.assertEqual({args_of(s)["--config-id"] for s in top}, {args_of(s)["--config-id"] for s in conf})
        self.assertTrue(all(s["out"].startswith("result/s4/confirm/") for s in top))
        self.guards(top, k=2)
        REPORT["top_up"] = {"n_specs": len(top)}

    def test_bars_confirm(self):
        grid = [float(x) for x in F.P["training"]["lr_grid"]["adam"]]
        self.assertEqual(len(grid), 5)                                                     # v4 five-value grid
        acc = {("HAR", grid[0]): 60.0, ("HAR", grid[1]): 70.0, ("HAR", grid[2]): 65.0,      # best: grid[1]
               ("HAR", grid[3]): 64.0, ("HAR", grid[4]): 69.999999,
               ("KWS", grid[0]): 55.0, ("KWS", grid[1]): 55.0, ("KWS", grid[2]): 55.0,      # tie: smallest lr
               ("KWS", grid[3]): 55.0, ("KWS", grid[4]): 55.0,
               ("CIFAR10", grid[0]): 30.0, ("CIFAR10", grid[1]): 35.0, ("CIFAR10", grid[2]): 35.000001,
               ("CIFAR10", grid[3]): 34.0, ("CIFAR10", grid[4]): 20.0}
        bars = self.env.tmp / "bars"
        F.write_bars(bars, self.env, acc)
        n_bars = len(F.TASKS) * (1 + len(grid))
        self.assertEqual(n_bars, 18)
        self.assertEqual(len(list(bars.rglob("result.json"))), n_bars)
        specs = specs4.bars_confirm_specs(bars, F.P)
        self.assertEqual(len(specs), 3 * 2 * len(F.CONF))
        self.unique(specs)
        lin = {args_of(s)["--config-id"]: float(args_of(s)["--lr"]) for s in specs if "linear" in s["run_id"]}
        self.assertEqual(lin, {"HAR|linear": grid[1], "KWS|linear": grid[0], "CIFAR10|linear": grid[2]})
        self.assertEqual(sum(1 for s in specs if "majority" in s["run_id"]), 3 * len(F.CONF))
        self.assertTrue(all(args_of(s)["--mode"] == "confirm" and int(args_of(s)["--seed"]) in F.CONF for s in specs))
        self.guards(specs, k=2)
        REPORT["bars_confirm"] = {"n_specs": len(specs), "linear_lr": lin}
        # errors: missing, duplicate, wrong mode
        p = bars / "KWS__linear__0.003" / "result.json"
        with F.restored(p):
            p.unlink()
            with self.assertRaisesRegex(specs4.S4Error, "missing"):
                specs4.bars_confirm_specs(bars, F.P)
        q = bars / "dup" / "result.json"
        with F.restored(q):
            F.dump(q, json.loads(p.read_text()))
            with self.assertRaisesRegex(specs4.S4Error, "duplicate"):
                specs4.bars_confirm_specs(bars, F.P)
        with F.restored(p):
            F.edit_json(p, mode="select")
            with self.assertRaisesRegex(specs4.S4Error, "mode"):
                specs4.bars_confirm_specs(bars, F.P)
        with F.restored(p):
            F.edit_json(p, lr=0.002)
            with self.assertRaisesRegex(specs4.S4Error, "unexpected"):
                specs4.bars_confirm_specs(bars, F.P)


class TestErrors(Base):
    def _error(self, label, contains, *a):
        rc, err = self.main(*a)
        self.assertEqual(rc, 2, f"{label}: {err}")
        self.assertFalse(self.out.exists(), label)
        self.assertIn(contains, err, label)
        REPORT.setdefault("errors", {})[label] = err.strip()[-200:]

    def test_needs_inputs(self):
        self._error("confirm without --selection", "needs --selection", "--kind", "confirm")
        self._error("select without --lr", "needs --lr", "--kind", "select")

    def test_selection_prereg(self):
        q = F.dump(self.env.tmp / "sel_bad.json", selection_doc(self.env, cells_with_repeats(), prereg_sha256="0" * 64))
        self._error("selection prereg hash", "prereg_sha256", "--kind", "confirm", "--selection", str(q))

    def test_selection_code_freeze(self):
        q = F.dump(self.env.tmp / "sel_cf.json", selection_doc(self.env, cells_with_repeats(),
                                                               code_freeze={"harness": self.env.records["harness"]}))
        self._error("selection code_freeze", "code_freeze", "--kind", "top_up", "--selection", str(q))

    def test_selection_two_lrs(self):
        cells = cells_with_repeats()
        cells[1]["tied"] = entry(cells[1]["tied"]["config_id"], 0.001)
        cells[4]["tied"] = entry(cells[1]["tied"]["config_id"], 0.002)
        q = F.dump(self.env.tmp / "sel_2lr.json", selection_doc(self.env, cells))
        self._error("one configuration with two lrs", "two lrs", "--kind", "confirm", "--selection", str(q))

    def test_selection_wrong_class(self):
        cells = cells_with_repeats()
        cells[0]["tied"] = entry("HAR|untied|d24|T2|adam", 0.001)
        q = F.dump(self.env.tmp / "sel_cls.json", selection_doc(self.env, cells))
        self._error("untied configuration in the tied slot", "belongs to", "--kind", "confirm", "--selection", str(q))

    def test_lr_missing_entry(self):
        sel = F.lr_selections()
        del sel["CIFAR10"]["adafactor"]["untied"]
        q = F.write_lr_selected(self.env.tmp / "lr_missing.json", self.env, selections=sel)
        self._error("lr file missing an entry", "no selected lr", "--kind", "select", "--lr", str(q))

    def test_out_exists(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text("keep")
        rc, err = self.main("--kind", "profile_registry")
        self.assertEqual(rc, 2)
        self.assertEqual(self.out.read_text(), "keep")
        self.out.unlink()

    def test_guard_negative_control(self):
        """The guard check used above can fail: a select spec moved to seed 1 is refused by lb.train."""
        s = specs4.select_specs(self.lr, F.P)[0]
        a = args_of(s)
        cfg = train.parse_config_id(a["--config-id"], F.P)
        with self.assertRaises(train.GuardViolation):
            train.check_guards(cfg, "select", float(a["--lr"]), 1, 64, False, F.P)


if __name__ == "__main__":
    unittest.main()
