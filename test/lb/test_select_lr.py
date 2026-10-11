"""S3 tests for src/lb/select_lr.py (card S3-T2): synthetic result.json fixtures in a temporary directory; the
harness code-freeze record is written into a patched temporary RECORD_DIR. No run is trained.

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_select_lr.py -v
Every ERROR case is a negative control of the happy path (exit 2 and no output file).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lb import freeze, prereg, select_lr, specs  # noqa: E402

P = prereg.load()
REPORT = {}


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def spec_args(spec):
    a = dict(zip(spec["args"][::2], spec["args"][1::2]))
    return a["--config-id"], float(a["--lr"]), int(a["--seed"]), a["--mode"]


def write_run(runs, spec, acc, harness, status="ok", subdir=None, **override):
    cid, lr, seed, mode = spec_args(spec)
    r = {"config_id": cid, "lr": lr, "seed": seed, "mode": mode, "status": status, "final_val_acc": acc,
         "prereg_sha256": prereg.yaml_sha256(), "code_sha256": dict(harness)}
    r.update(override)
    d = runs / (subdir or spec["run_id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(r))
    return d / "result.json"


def default_acc(spec):
    """Deterministic fixture accuracy: the middle grid lr (index len // 2, not an edge) wins unless a test overrides it."""
    cid, lr, _, _ = spec_args(spec)
    opt = cid.split("|")[4]
    grid = [float(x) for x in P["training"]["lr_grid"][opt]]
    return 60.0 + 10.0 * (grid.index(lr) == len(grid) // 2) + {"HAR": 0.0, "KWS": 1.0, "CIFAR10": 2.0}[cid.split("|")[0]]


# lrcal count from the YAML, computed independently of lb.select_lr (v4: 3 tasks x 4 optimizers x 2 classes x 5 lrs)
N_LRCAL = sum(len(P["training"]["lr_grid"][o]) for _ in P["tasks"] for o in P["registry"]["optimizers"]
              for _c in ("untied", "tied"))


class SelectCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self._p = mock.patch.object(freeze, "RECORD_DIR", self.tmp / "records")
        self._p.start()
        freeze.write_record("harness")
        self.harness = select_lr.harness_code(freeze.load_record("harness"))
        self.runs = self.tmp / "runs"
        self.specs = specs.lrcal_specs(P)
        self.out = self.tmp / "out" / "lr_selected.json"

    def tearDown(self):
        self._p.stop()
        self._td.cleanup()

    def populate(self, acc=None, status=None):
        acc, status = acc or {}, status or {}
        for s in self.specs:
            write_run(self.runs, s, acc.get(s["run_id"], default_acc(s)), self.harness,
                      status=status.get(s["run_id"], "ok"))

    def main(self, *extra):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = select_lr.main(["--runs", str(self.runs), "--out", str(self.out), *extra])
        return rc, err.getvalue()


class TestSelection(SelectCase):
    def test_happy_tie_and_divergence(self):
        tie_a, tie_b = "lrcal/HAR__untied__adam__lr0.003__s0", "lrcal/HAR__untied__adam__lr0.01__s0"
        div = "lrcal/KWS__shared__sgd__lr0.3__s0"
        div_low = "lrcal/CIFAR10__untied__adafactor__lr0.03__s0"
        self.populate(acc={tie_a: 75.0, tie_b: 75.0, div: 90.0, div_low: 5.0},
                      status={div: "diverged", div_low: "diverged"})
        rc, err = self.main()
        self.assertEqual(rc, 0, err)
        d = json.loads(self.out.read_text())
        sel = d["selections"]
        self.assertEqual(sel["HAR"]["adam"]["untied"], 0.003)            # tie -> smaller lr
        self.assertEqual(sel["KWS"]["sgd"]["tied"], 0.3)                 # a diverged run keeps its accuracy
        self.assertEqual(sel["CIFAR10"]["adafactor"]["untied"], 0.01)    # the low-accuracy diverged run loses
        self.assertEqual(sel["HAR"]["sgd_momentum"]["tied"], 0.1)        # default fixture: middle lr wins
        self.assertEqual(N_LRCAL, 120)                                   # v4: five-value grids
        self.assertEqual(len(select_lr.expected_lrcal(P)), N_LRCAL)
        self.assertEqual(len(self.specs), N_LRCAL)
        self.assertEqual(d["n_runs"], N_LRCAL)
        self.assertEqual(len(d["runs"]), N_LRCAL)
        self.assertEqual(sum(1 for r in d["runs"] if r["status"] == "diverged"), 2)
        self.assertEqual(len(d["inputs_sha256"]), N_LRCAL + 1)           # 120 results + the harness record
        self.assertEqual(d["n_at_edge"], 0)                               # every selection is interior here
        self.assertEqual({v for o in d["edge_flags"].values() for c in o.values() for v in c.values()}, {None})
        self.assertEqual(d["prereg_sha256"], prereg.yaml_sha256())
        self.assertEqual(set(d["code_freeze"]), {"harness"})
        n_sel = sum(len(c) for o in sel.values() for c in o.values())
        self.assertEqual(n_sel, 24)                                       # 3 tasks x 4 optimizers x 2 classes
        # negative control of the tie rule: with lr 0.01 ahead by 1e-9 points the larger lr is chosen
        before = self.out.read_bytes()
        rc2, err2 = self.main()
        self.assertEqual(rc2, 2)                                          # never overwrites
        self.assertIn("exists", err2)
        self.assertEqual(self.out.read_bytes(), before)
        self.out.unlink()
        write_run(self.runs, next(s for s in self.specs if s["run_id"] == tie_b), 75.000000001, self.harness)
        self.assertEqual(self.main()[0], 0)
        self.assertEqual(json.loads(self.out.read_text())["selections"]["HAR"]["adam"]["untied"], 0.01)
        REPORT["selections_fixture"] = sel

    def test_edge_flags(self):
        """v4 training.lr_selection: a selection at the first or last grid value is flagged; the selection itself is
        the same argmax (ties to the smaller lr) as without flags."""
        g = {o: [float(x) for x in P["training"]["lr_grid"][o]] for o in P["registry"]["optimizers"]}
        first = f"lrcal/HAR__untied__adam__lr{g['adam'][0]:g}__s0"
        last = f"lrcal/KWS__shared__sgd__lr{g['sgd'][-1]:g}__s0"
        near = f"lrcal/CIFAR10__untied__adafactor__lr{g['adafactor'][1]:g}__s0"   # second value: not an edge
        tie_lo = f"lrcal/HAR__shared__sgd_momentum__lr{g['sgd_momentum'][0]:g}__s0"
        tie_hi = f"lrcal/HAR__shared__sgd_momentum__lr{g['sgd_momentum'][-1]:g}__s0"
        ids = {s["run_id"] for s in self.specs}
        self.assertTrue({first, last, near, tie_lo, tie_hi} <= ids)
        self.populate(acc={first: 99.0, last: 99.0, near: 99.0, tie_lo: 95.0, tie_hi: 95.0})
        rc, err = self.main()
        self.assertEqual(rc, 0, err)
        d = json.loads(self.out.read_text())
        sel, edge = d["selections"], d["edge_flags"]
        self.assertEqual(sel["HAR"]["adam"]["untied"], g["adam"][0])
        self.assertEqual(edge["HAR"]["adam"]["untied"], "first")
        self.assertEqual(sel["KWS"]["sgd"]["tied"], g["sgd"][-1])
        self.assertEqual(edge["KWS"]["sgd"]["tied"], "last")
        self.assertEqual(sel["CIFAR10"]["adafactor"]["untied"], g["adafactor"][1])
        self.assertIsNone(edge["CIFAR10"]["adafactor"]["untied"])          # negative control: next to the edge
        self.assertEqual(sel["HAR"]["sgd_momentum"]["tied"], g["sgd_momentum"][0])   # tie of both edges -> smaller lr
        self.assertEqual(edge["HAR"]["sgd_momentum"]["tied"], "first")
        self.assertEqual(d["n_at_edge"], 3)
        # independent recomputation of every flag from the selections and the YAML grids
        for t, o_ in sel.items():
            for o, c_ in o_.items():
                for c, lr in c_.items():
                    want = "first" if lr == g[o][0] else ("last" if lr == g[o][-1] else None)
                    self.assertEqual(edge[t][o][c], want, (t, o, c))
        self.assertEqual(select_lr.edge_flag(g["adam"][2], g["adam"]), None)
        self.assertEqual(select_lr.edge_flag(g["adam"][-1], g["adam"]), "last")
        REPORT["edge_flags_fixture"] = {"n_at_edge": d["n_at_edge"], "HAR adam untied": edge["HAR"]["adam"]["untied"],
                                        "KWS sgd tied": edge["KWS"]["sgd"]["tied"]}

    def test_predictor_entries_not_compared(self):
        self.populate()
        p = self.runs / self.specs[0]["run_id"] / "result.json"
        r = json.loads(p.read_text())
        r["code_sha256"]["predictor.py"] = "a" * 64
        p.write_text(json.dumps(r))
        self.assertEqual(self.main()[0], 0)


class TestErrors(SelectCase):
    """Each ERROR: exit 2, no output file."""

    def _error(self, label, contains=None):
        rc, err = self.main()
        self.assertEqual(rc, 2, label)
        self.assertFalse(self.out.exists(), label)
        if contains:
            self.assertIn(contains, err, label)
        REPORT.setdefault("errors", {})[label] = err.strip()[-170:]

    def _edit(self, run_id, **kw):
        p = self.runs / run_id / "result.json"
        r = json.loads(p.read_text())
        r.update(kw)
        p.write_text(json.dumps(r))

    def test_missing_run(self):
        self.populate()
        shutil.rmtree(self.runs / self.specs[5]["run_id"])
        self._error("missing run", "missing")

    def test_duplicate_run(self):
        self.populate()
        shutil.copytree(self.runs / self.specs[3]["run_id"], self.runs / "copy" / "again")
        self._error("duplicate run", "duplicate")

    def test_wrong_prereg_hash(self):
        self.populate()
        self._edit(self.specs[0]["run_id"], prereg_sha256="0" * 64)
        self._error("wrong prereg hash", "prereg_sha256")

    def test_wrong_code_hash(self):
        self.populate()
        p = self.runs / self.specs[1]["run_id"] / "result.json"
        r = json.loads(p.read_text())
        r["code_sha256"]["train.py"] = "0" * 64
        p.write_text(json.dumps(r))
        self._error("wrong code hash", "code_sha256")

    def test_wrong_mode(self):
        self.populate()
        self._edit(self.specs[2]["run_id"], mode="dev")
        self._error("wrong mode", "mode")

    def test_bad_status(self):
        self.populate()
        self._edit(self.specs[2]["run_id"], status="error")
        self._error("status error", "status")

    def test_lr_off_grid(self):
        self.populate()
        self._edit(self.specs[2]["run_id"], lr=0.002)
        self._error("lr off grid")

    def test_wrong_seed(self):
        self.populate()
        self._edit(self.specs[2]["run_id"], seed=1)
        self._error("wrong seed", "unexpected")

    def test_unexpected_config(self):
        self.populate()
        write_run(self.runs, specs._spec("lrcal", "HAR|untied|d32|T2|adam", 0.001, 0, "lrcal"), 50.0, self.harness)
        self._error("unexpected config", "unexpected")

    def test_nonfinite_accuracy(self):
        self.populate()
        p = self.runs / self.specs[2]["run_id"] / "result.json"
        before = p.read_text()
        self._edit(self.specs[2]["run_id"], final_val_acc=None)     # written as JSON null
        self.assertNotEqual(p.read_text(), before)
        self.assertIn('"final_val_acc": null', p.read_text())
        self._error("null accuracy", "final_val_acc")

    def test_missing_harness_record(self):
        self.populate()
        freeze.record_path("harness").unlink()
        self._error("missing harness record", "missing")

    def test_harness_record_mismatch_with_tree(self):
        self.populate()
        real = freeze._src_py()
        with mock.patch.object(freeze, "_src_py", return_value=real + [freeze.SRC_LB / "zz_new.py"]):
            self._error("tree differs from harness record", "file set changed")


if __name__ == "__main__":
    unittest.main()
