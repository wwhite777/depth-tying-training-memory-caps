"""Fixture tests for src/gates/freeze_prereg.py.

Each test copies the real YAML, contract and REJECT_MEMO.md into a fresh temporary directory, edits only
the copies, and runs the gate as a subprocess on them. The originals are never written.
Run: python -m unittest test/gates/test_freeze_prereg.py -v
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src" / "gates" / "freeze_prereg.py"
YAML_NAME = "PREREGISTERED_loopbudget_v1.yaml"
CONTRACT_NAME = "CONTRIBUTION_CONTRACT.md"
MEMO_NAME = "REJECT_MEMO.md"


class FreezeGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)
        for name in (YAML_NAME, CONTRACT_NAME, MEMO_NAME):
            shutil.copyfile(ROOT / name, self.d / name)
        self.yaml = self.d / YAML_NAME
        self.contract = self.d / CONTRACT_NAME
        self.out = self.d / "out"
        # The real v1 YAML is FROZEN since the S2 freeze (2026-10-10T05:20Z); the fixture is a DRAFT copy, as the
        # tests were written for (the freeze-path tests edit DRAFT -> FROZEN on the copy).
        self.edit(self.yaml, "\nstatus: FROZEN\n", "\nstatus: DRAFT\n")

    def tearDown(self):
        self._tmp.cleanup()

    def edit(self, path, old, new):
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count(old), 1, f"fixture edit target not unique in {path.name}: {old!r}")
        path.write_text(text.replace(old, new), encoding="utf-8")

    def run_gate(self, *mode, contract=None):
        cmd = [sys.executable, str(SCRIPT), "--yaml", str(self.yaml),
               "--contract", str(contract or self.contract), "--out-dir", str(self.out), *mode]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    def receipts(self):
        return sorted(self.out.glob("freeze_receipt_*.json"))

    def assertGate(self, res, code, status, *needles):
        self.assertEqual(res.returncode, code, res.stdout + res.stderr)
        self.assertIn(f"GATE freeze_prereg: {status}", res.stdout)
        for n in needles:
            self.assertIn(n, res.stdout)
        rec = json.loads(self.receipts()[-1].read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], status)
        self.assertEqual(rec["exit_status"], code)
        for n in needles:
            self.assertTrue(any(n in r for r in rec["reasons"]), f"{n!r} not in receipt reasons {rec['reasons']}")
        return rec

    def test_lr_grid_five_values_passes(self):
        self.edit(self.yaml, "adam: [0.001, 0.003, 0.01]", "adam: [0.0003, 0.001, 0.003, 0.01, 0.03]")
        res = self.run_gate("--check")
        self.assertGate(res, 0, "PASS")

    def test_lr_grid_two_values_fails(self):
        self.edit(self.yaml, "adam: [0.001, 0.003, 0.01]", "adam: [0.001, 0.003]")
        res = self.run_gate("--check")
        self.assertGate(res, 1, "FAIL", "training.lr_grid.adam")

    def test_lr_grid_not_increasing_fails(self):
        self.edit(self.yaml, "adam: [0.001, 0.003, 0.01]", "adam: [0.003, 0.001, 0.01]")
        res = self.run_gate("--check")
        self.assertGate(res, 1, "FAIL", "training.lr_grid.adam")

    def test_good_copy_check_passes(self):
        res = self.run_gate("--check")
        rec = self.assertGate(res, 0, "PASS")
        self.assertEqual(rec["reasons"], [])
        for k in ("gate", "mode", "status", "reasons", "yaml_path", "yaml_sha256", "contract_path",
                  "contract_sha256", "script_sha256", "command", "utc_start", "utc_end", "exit_status"):
            self.assertIn(k, rec)
        self.assertEqual(rec["mode"], "check")
        self.assertFalse((self.d / "PREREGISTERED_loopbudget_v1.sha256").exists())

    def test_missing_required_key_fails(self):
        self.edit(self.yaml, "  alpha: 0.05\n", "")
        self.assertGate(self.run_gate("--check"), 1, "FAIL", "missing key: inference.alpha")

    def test_contract_cap_mismatch_fails(self):
        self.edit(self.contract, "caps_kib: [256, 323, 406,", "caps_kib: [256, 324, 406,")
        self.assertGate(self.run_gate("--check"), 1, "FAIL", "contract mismatch: caps_kib")

    def test_registry_count_mismatch_fails(self):
        self.edit(self.yaml, "configurations_per_task: 248", "configurations_per_task: 247")
        self.assertGate(self.run_gate("--check"), 1, "FAIL", "registry count mismatch: recomputed 62 architectures")

    def test_overlapping_seeds_fail(self):
        self.edit(self.yaml, "top_up: [11, 12]", "top_up: [10, 11]")
        self.assertGate(self.run_gate("--check"), 1, "FAIL",
                        "seed lists overlap: seeds.confirmation and seeds.top_up share [10]")

    def test_freeze_with_draft_fails(self):
        self.assertGate(self.run_gate("--freeze"), 1, "FAIL", "--freeze requires status FROZEN")
        self.assertFalse((self.d / "PREREGISTERED_loopbudget_v1.sha256").exists())

    def _freeze_frozen(self):
        self.edit(self.yaml, "\nstatus: DRAFT\n", "\nstatus: FROZEN\n")
        return self.run_gate("--freeze")

    def test_freeze_with_frozen_passes_and_verifies(self):
        rec = self.assertGate(self._freeze_frozen(), 0, "PASS")
        self.assertEqual(rec["mode"], "freeze")
        sha = self.d / "PREREGISTERED_loopbudget_v1.sha256"
        self.assertTrue(sha.is_file())
        self.assertEqual(len(sha.read_text(encoding="utf-8").splitlines()), 2)
        v = subprocess.run(["sha256sum", "-c", sha.name], cwd=self.d, capture_output=True, text=True)
        self.assertEqual(v.returncode, 0, v.stdout + v.stderr)
        self.assertIn(f"{YAML_NAME}: OK", v.stdout)
        self.assertIn(f"{CONTRACT_NAME}: OK", v.stdout)

    def test_second_freeze_never_overwrites(self):
        self.assertEqual(self._freeze_frozen().returncode, 0)
        sha = self.d / "PREREGISTERED_loopbudget_v1.sha256"
        before = sha.read_bytes()
        self.assertGate(self.run_gate("--freeze"), 1, "FAIL", "freeze receipt exists; never overwrite")
        self.assertEqual(sha.read_bytes(), before)

    def test_unparsable_yaml_errors(self):
        with open(self.yaml, "a", encoding="utf-8") as f:
            f.write("broken: [unclosed\n")
        self.assertGate(self.run_gate("--check"), 2, "ERROR", "unparsable YAML")

    def test_missing_contract_errors(self):
        self.assertGate(self.run_gate("--check", contract=self.d / "NO_SUCH_CONTRACT.md"), 2, "ERROR",
                        "missing file:")


if __name__ == "__main__":
    unittest.main()
