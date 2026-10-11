"""S3 tests for src/lb/freeze.py (card S3-T2; the preregistration in force (lb.prereg; PREREGISTERED_loopbudget_v4.yaml), code_freeze_s3).

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_freeze.py -v
Every test patches freeze.RECORD_DIR to a temporary directory; the real records (v3: research/s3/CODE_FREEZE_*.json;
the versioned default RECORD_DIR research/s3/v4/) are never written and research/s3/v4/ is not created (checked before
and after). Every check has an in-file negative control showing that it can fail.
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lb import freeze, prereg  # noqa: E402

P = prereg.load()
DEFAULT_RECORD_DIR = freeze.RECORD_DIR                     # the unpatched module default
V4_DIR = ROOT / "research" / "s3" / "v4"
REAL = [d / f"CODE_FREEZE_{n}.json" for d in (ROOT / "research" / "s3", V4_DIR) for n in freeze.NAMES]
REAL_STATE = {p: (p.read_bytes() if p.exists() else None) for p in REAL}
V4_DIR_EXISTED = V4_DIR.exists()
REPORT = {}


def tearDownModule():
    after = {p: (p.read_bytes() if p.exists() else None) for p in REAL}
    REPORT["real_records_untouched"] = after == REAL_STATE
    REPORT["v4_dir_existence_unchanged"] = V4_DIR.exists() == V4_DIR_EXISTED
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))
    if after != REAL_STATE:
        raise AssertionError("a test changed the real research/s3/CODE_FREEZE_*.json")
    if V4_DIR.exists() != V4_DIR_EXISTED:
        raise AssertionError("a test created or removed research/s3/v4/")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FreezeCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.rd = self.tmp / "records"
        self._p = mock.patch.object(freeze, "RECORD_DIR", self.rd)
        self._p.start()
        self.art = self.tmp / "calibration_artifact.json"
        self.art.write_text('{"fit": [1, 2, 3]}\n')

    def tearDown(self):
        self._p.stop()
        self._td.cleanup()

    def write_both(self):
        h = freeze.write_record("harness")
        p = freeze.write_record("predictor", [self.art])
        return h, p


class TestRecordDir(unittest.TestCase):
    def test_default_is_versioned(self):
        """RECORD_DIR defaults to research/s3/v<version of the YAML in force>/ (v4), not the v3 research/s3/."""
        self.assertEqual(int(P["version"]), 4)
        self.assertEqual(DEFAULT_RECORD_DIR, ROOT / "research" / "s3" / f"v{int(P['version'])}")
        self.assertEqual(DEFAULT_RECORD_DIR, V4_DIR)
        self.assertNotEqual(DEFAULT_RECORD_DIR, ROOT / "research" / "s3")
        self.assertEqual(freeze.versioned_dir("research", "s3", p=P), V4_DIR)
        # negative control: the directory follows the YAML version
        self.assertEqual(freeze.versioned_dir("research", "s3", p=dict(P, version=3)), ROOT / "research" / "s3" / "v3")
        REPORT["default_record_dir"] = str(DEFAULT_RECORD_DIR.relative_to(ROOT))

    def test_record_path_follows_patch(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(freeze, "RECORD_DIR", Path(d)):
            self.assertEqual(freeze.record_path("harness"), Path(d) / "CODE_FREEZE_harness.json")
        self.assertEqual(freeze.record_path("predictor"), V4_DIR / "CODE_FREEZE_predictor.json")


class TestRecords(FreezeCase):
    def test_matching_records_pass(self):
        (hp, hh), (pp, ph) = self.write_both()
        self.assertEqual(hp, self.rd / "CODE_FREEZE_harness.json")
        r = freeze.verify("harness", "predictor")
        self.assertEqual(r, {"harness": _sha(hp), "predictor": _sha(pp)})
        self.assertEqual((hh, ph), (_sha(hp), _sha(pp)))
        rec = json.loads(hp.read_text())
        self.assertEqual(set(rec), {"name", "files", "prereg_sha256", "written_utc"})
        self.assertEqual(rec["prereg_sha256"], prereg.yaml_sha256())
        names = {Path(k).name for k in rec["files"]}
        self.assertIn("train.py", names)
        self.assertIn("freeze.py", names)
        self.assertFalse(any(n.startswith("predictor") for n in names))
        self.assertEqual(set(rec["files"]), {str(f.relative_to(ROOT)) for f in sorted((ROOT / "src/lb").glob("*.py"))
                                             if not f.name.startswith("predictor")})
        for k, h in rec["files"].items():
            self.assertEqual(h, _sha(ROOT / k))
        prec = json.loads(pp.read_text())
        self.assertIn(str(self.art.resolve()), prec["files"])
        REPORT["harness_files"] = sorted(rec["files"])
        REPORT["predictor_files_in_test"] = sorted(prec["files"])

    def test_missing_record_raises(self):
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify("harness")
        freeze.write_record("harness")
        freeze.verify("harness")                                  # control: present record passes
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify("harness", "predictor")
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify()
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify("nonsense")

    def test_changed_hash_raises(self):
        (hp, _), (pp, _) = self.write_both()
        freeze.verify("harness", "predictor")
        rec = json.loads(hp.read_text())
        k = sorted(rec["files"])[0]
        rec["files"][k] = "0" * 64
        hp.write_text(json.dumps(rec))
        with self.assertRaises(freeze.CodeFreezeError) as cm:
            freeze.verify("harness")
        self.assertIn(k, str(cm.exception))
        # a changed recorded file (the calibration artifact) raises too
        freeze.verify("predictor")
        self.art.write_text('{"fit": [1, 2, 4]}\n')
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify("predictor")
        REPORT["changed_hash"] = str(cm.exception)[:160]

    def test_prereg_hash_in_record_checked(self):
        hp, _ = freeze.write_record("harness")
        rec = json.loads(hp.read_text())
        rec["prereg_sha256"] = "f" * 64
        hp.write_text(json.dumps(rec))
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.verify("harness")

    def test_extra_unrecorded_py_raises(self):
        freeze.write_record("harness")
        freeze.write_record("predictor", [self.art])
        real = freeze._src_py()
        with mock.patch.object(freeze, "_src_py", return_value=real + [freeze.SRC_LB / "zz_unrecorded.py"]):
            with self.assertRaises(freeze.CodeFreezeError) as cm:
                freeze.verify("harness")
            self.assertIn("zz_unrecorded.py", str(cm.exception))
        with mock.patch.object(freeze, "_src_py", return_value=real[1:]):          # a recorded file disappears
            with self.assertRaises(freeze.CodeFreezeError):
                freeze.verify("harness")
        with mock.patch.object(freeze, "_src_py", return_value=real + [freeze.SRC_LB / "predictor_new.py"]):
            freeze.verify("harness")                               # predictor*.py is not part of the harness
            with self.assertRaises(freeze.CodeFreezeError):        # but an unrecorded predictor file is caught
                freeze.verify("predictor")
        freeze.verify("harness", "predictor")                      # control: the unpatched listing passes
        REPORT["extra_file"] = str(cm.exception)[:160]

    def test_never_overwrites(self):
        hp, _ = freeze.write_record("harness")
        before = hp.read_bytes()
        with self.assertRaises(FileExistsError):
            freeze.write_record("harness")
        self.assertEqual(hp.read_bytes(), before)
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.write_record("harness", [self.art])            # the harness record takes no extras
        with self.assertRaises(freeze.CodeFreezeError):
            freeze.write_record("predictor", [self.tmp / "missing.json"])
        with mock.patch.object(freeze, "_src_py", return_value=[]):
            with self.assertRaises(freeze.CodeFreezeError):        # nothing to record
                freeze.write_record("predictor")


class TestCLI(FreezeCase):
    def _main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = freeze.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_cli(self):
        res = {}
        res["verify_missing"] = self._main(["--verify", "harness"])[0]
        res["write_harness"] = self._main(["--write", "harness"])[0]
        res["write_harness_again"] = self._main(["--write", "harness"])[0]
        res["verify_harness"] = self._main(["--verify", "harness"])[0]
        res["verify_both_missing_predictor"] = self._main(["--verify", "harness", "predictor"])[0]
        res["write_predictor"] = self._main(["--write", "predictor", "--extra", str(self.art)])[0]
        rc, out, _ = self._main(["--verify", "harness", "predictor"])
        res["verify_both"] = rc
        self.assertEqual(set(json.loads(out)), {"harness", "predictor"})
        self.assertEqual(res, {"verify_missing": 2, "write_harness": 0, "write_harness_again": 2,
                               "verify_harness": 0, "verify_both_missing_predictor": 2, "write_predictor": 0,
                               "verify_both": 0})
        REPORT["cli_exit_codes"] = res


if __name__ == "__main__":
    unittest.main()
