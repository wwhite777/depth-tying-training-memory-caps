"""S3 tests for src/lb/floor.py (card S3-T2): synthetic floor-run fixtures and a synthetic lr_selected.json in a
temporary directory; the harness code-freeze record is written into a patched temporary RECORD_DIR. No run is trained.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_floor.py -v
SD values are checked against numpy (ddof 1), computed independently of the module. Every ERROR case is a negative
control of the happy path (exit 2 and no output file).
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "test" / "lb"))

from lb import floor, freeze, prereg, select_lr, specs, train  # noqa: E402
from test_select_lr import spec_args, write_run  # noqa: E402

P = prereg.load()
REPORT = {}
SELECTED = {"HAR": 0.003, "KWS": 0.001, "CIFAR10": 0.01}
ACCS = {"HAR": [80.0, 82.0, 84.0, 86.0, 88.0],          # SD 3.162 -> floor 2 x SD = 6.32
        "KWS": [80.0, 80.1, 80.2, 80.1, 80.0],          # SD 0.084 -> floor 1.0
        "CIFAR10": [50.0, 55.0, 60.0, 65.0, 70.0]}      # SD 7.9, but one run diverged -> floor 1.0


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


class FloorCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._p = mock.patch.object(freeze, "RECORD_DIR", self.tmp / "records")
        self._p.start()
        freeze.write_record("harness")
        self.harness = select_lr.harness_code(freeze.load_record("harness"))
        self.lr_path = self.tmp / "lr_selected.json"
        sel = {t: {o: {"untied": SELECTED[t] if o == "adam" else float(P["training"]["lr_grid"][o][0]),
                       "tied": float(P["training"]["lr_grid"][o][0])} for o in P["registry"]["optimizers"]}
               for t in P["tasks"]}
        self.lr_path.write_text(json.dumps({"kind": "lr_selection", "prereg_sha256": prereg.yaml_sha256(),
                                            "selections": sel}))
        self.specs = specs.floor_specs(self.lr_path, P)
        self.runs = self.tmp / "runs"
        self.out = self.tmp / "floor_values.json"

    def tearDown(self):
        self._p.stop()
        self._td.cleanup()

    def populate(self, diverged=(("CIFAR10", 103),)):
        for s in self.specs:
            cid, _, seed, _ = spec_args(s)
            task = cid.split("|")[0]
            acc = ACCS[task][train.FLOOR_SEEDS.index(seed)]
            write_run(self.runs, s, acc, self.harness, status="diverged" if (task, seed) in diverged else "ok")

    def main(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = floor.main(["--runs", str(self.runs), "--lr", str(self.lr_path), "--out", str(self.out)])
        return rc, err.getvalue()


class TestFloor(FloorCase):
    def test_values(self):
        self.populate()
        rc, err = self.main()
        self.assertEqual(rc, 0, err)
        d = json.loads(self.out.read_text())
        f = d["floors"]
        sd = {t: float(np.std(ACCS[t], ddof=1)) for t in ACCS}
        for t in ACCS:
            self.assertAlmostEqual(f[t]["sd"], sd[t], places=12)
            self.assertEqual(f[t]["lr"], SELECTED[t])
            self.assertEqual(f[t]["seeds"], [101, 102, 103, 104, 105])
        self.assertAlmostEqual(d["floor_values"]["HAR"], 2 * sd["HAR"], places=12)        # 2 x SD > 1
        self.assertGreater(2 * sd["HAR"], 1.0)
        self.assertEqual(d["floor_values"]["KWS"], 1.0)                                      # 2 x SD < 1 -> 1.0
        self.assertLess(2 * sd["KWS"], 1.0)
        self.assertEqual(d["floor_values"]["CIFAR10"], 1.0)                                  # divergence -> 1.0
        self.assertGreater(2 * sd["CIFAR10"], 1.0)
        self.assertEqual(f["CIFAR10"]["diverged_seeds"], [103])
        n_floor = len(P["tasks"]) * len(train.FLOOR_SEEDS)
        self.assertEqual(n_floor, 15)
        self.assertEqual(len(self.specs), n_floor)
        self.assertEqual(len(d["inputs_sha256"]), n_floor + 2)     # 15 results + lr_selected.json + harness record
        # negative control of the divergence rule: without the divergence CIFAR10's floor is 2 x SD
        self.assertAlmostEqual(floor.floor_value(ACCS["CIFAR10"], False)["floor"], 2 * sd["CIFAR10"], places=12)
        rc2, _ = self.main()
        self.assertEqual(rc2, 2)                                   # never overwrites
        REPORT["floor_values_fixture"] = d["floor_values"]
        REPORT["sd_fixture"] = {t: f[t]["sd"] for t in f}

    def test_rule_text(self):
        self.assertIn("seeds 101-105", P["floor"]["rule"])
        self.assertIn("max(1.0, 2 x SD_task)", P["floor"]["rule"])
        self.assertIn("if a floor run diverges, that task uses 1.0", P["floor"]["instantiation"])


class TestFloorErrors(FloorCase):
    def _error(self, label, contains=None):
        rc, err = self.main()
        self.assertEqual(rc, 2, label)
        self.assertFalse(self.out.exists(), label)
        if contains:
            self.assertIn(contains, err, label)
        REPORT.setdefault("errors", {})[label] = err.strip()[-170:]

    def _edit(self, i, **kw):
        p = self.runs / self.specs[i]["run_id"] / "result.json"
        r = json.loads(p.read_text())
        r.update(kw)
        p.write_text(json.dumps(r))

    def test_missing_seed(self):
        self.populate()
        shutil.rmtree(self.runs / self.specs[4]["run_id"])
        self._error("missing seed", "missing")

    def test_wrong_lr(self):
        self.populate()
        self._edit(0, lr=0.01)                     # HAR selected 0.003
        self._error("wrong lr", "selected lr")

    def test_duplicate(self):
        self.populate()
        shutil.copytree(self.runs / self.specs[0]["run_id"], self.runs / "dup" / "x")
        self._error("duplicate", "duplicate")

    def test_wrong_mode(self):
        self.populate()
        self._edit(1, mode="lrcal")
        self._error("wrong mode", "mode")

    def test_seed_outside(self):
        self.populate()
        self._edit(1, seed=7)
        self._error("seed outside", "unexpected")

    def test_wrong_code_hash(self):
        self.populate()
        p = self.runs / self.specs[2]["run_id"] / "result.json"
        r = json.loads(p.read_text())
        r["code_sha256"]["models.py"] = "0" * 64
        p.write_text(json.dumps(r))
        self._error("wrong code hash", "code_sha256")

    def test_wrong_prereg_hash(self):
        self.populate()
        self._edit(3, prereg_sha256="1" * 64)
        self._error("wrong prereg hash", "prereg_sha256")

    def test_lr_selection_wrong_hash(self):
        self.populate()
        d = json.loads(self.lr_path.read_text())
        d["prereg_sha256"] = "2" * 64
        self.lr_path.write_text(json.dumps(d))
        self._error("lr_selected.json wrong prereg hash", "prereg_sha256")


if __name__ == "__main__":
    unittest.main()
