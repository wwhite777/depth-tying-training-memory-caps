"""Oracle regression tests of research/s9/transfer_analysis.py, added 2026-10-14 after audit r005 (advisory F-r005-5):
the 12 frozen tests in test_transfer_analysis.py still passed when the margin was changed from 0.5 to 0.01 or the R2
standard deviation from ddof 1 to 0. These tests pin both rules against values computed here independently. The frozen
plan, analysis script and original tests are unchanged; this file only adds tests.

  TMPDIR=<scratch> ~/envs/jeongwoncheol_loopbudget_docs/bin/python -m unittest research/s9/test_transfer_analysis_oracle.py -v

TRANSFER_ANALYSIS_SCRIPT=<path> runs them against another copy of the analysis script (planted-failure checks).
"""
import importlib.util
import pathlib
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("tta", HERE / "test_transfer_analysis.py")
tta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tta)   # reuses its fake-tree builder and honours TRANSFER_ANALYSIS_SCRIPT


class TransferOracleTests(unittest.TestCase):
    def tree(self, *a, **k):
        t = tta.Tree(*a, **k)
        self.addCleanup(t.close)
        return t

    def test_margin_supported_well_inside(self):
        """Mean shift 0.30 pp with tiny noise: the 90% interval lies inside +-0.5 -> SUPPORTED (a 0.01 margin would
        call it CONTRADICTED)."""
        rng = np.random.default_rng(3)
        noise = {(c, s): float(rng.normal(0, 0.01)) for c in tta.CFGS for s in (1, 2, 3)}
        t = self.tree((1, 2, 3), lambda c, s: 0.30 + noise[(c, s)])
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        lo, hi = res["R1"]["ci90"]
        self.assertTrue(-0.5 < lo and hi < 0.5 and lo > 0.01, (lo, hi))
        self.assertEqual(res["R1"]["margin_pp"], 0.5)
        self.assertEqual(res["R1"]["verdict"], "SUPPORTED")

    def test_margin_unresolved_straddling(self):
        """Mean shift 0.47 pp whose 90% upper bound crosses 0.5 while |mean| < 0.5 -> UNRESOLVED (a 0.01 margin would
        call it CONTRADICTED)."""
        rng = np.random.default_rng(8)
        raw = {(c, s): float(rng.normal(0, 0.15)) for c in sorted(tta.CFGS) for s in (1, 2, 3)}
        m = float(np.mean(list(raw.values())))
        t = self.tree((1, 2, 3), lambda c, s: 0.47 + raw[(c, s)] - m)
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        self.assertAlmostEqual(res["R1"]["mean"], 0.47, places=9)
        self.assertGreater(res["R1"]["ci90"][1], 0.5)
        self.assertGreater(res["R1"]["ci95"][0], 0.0)
        self.assertEqual(res["R1"]["verdict"], "UNRESOLVED")

    def test_r2_sd_is_sample_sd(self):
        """Every run shifted by 0.2 pp; archived seeds 1-12 spread as linspace(-1.5, 1.5, 12) around each base, so the
        R2 ratio must equal 0.2 / (sample SD, ddof 1) for every run (ddof 0 gives a ratio about 4.4% larger)."""
        t = self.tree((1, 2, 3), lambda c, s: 0.2)
        rc, res, log = t.run("seeds123")
        self.assertEqual(rc, 0, log)
        x = np.linspace(-1.5, 1.5, 12)
        sd1 = float(np.sqrt(np.sum((x - x.mean()) ** 2) / (len(x) - 1)))
        self.assertAlmostEqual(res["R2"]["mean_ratio"], 0.2 / sd1, delta=1e-9)
        for r in res["R2"]["runs"]:
            self.assertAlmostEqual(r["archived_sd_pp"], sd1, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
