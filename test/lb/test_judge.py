"""Fixture tests for src/lb/judge.py (G6 judge). Every fixture under test/lb/fixtures/judge is SYNTHETIC: hand-built
numbers, no experimental result. Run: python -m unittest test/lb/test_judge.py -v

Fixture construction (all N1 fixtures): untied accuracy = 80.0 for every seed, tied = 80.0 + d_s, with
d_s = mean + a, mean - a, mean + a, ... (alternating), so for n = 10 seeds the sample SD = a * sqrt(10/9) and
SE = SD / sqrt(10) = a / 3 exactly (sum of squares 10 a^2 over df 9); for n = 12, SE = a / sqrt(11). All values are
binary-exact (multiples of 0.25), so d_s, mean and the deviations are exact. Background cells: mean 0, a 0.25.
Student-t tails are checked against the closed form for odd df (Abramowitz and Stegun 26.7.4), independent of scipy:
  A(t | nu) = (2/pi) [theta + sin(theta) (cos(theta) + (2/3) cos^3 + (2*4)/(3*5) cos^5 + ...)], theta = atan(t / sqrt(nu)),
  upper tail P(T >= t) = (1 - A) / 2.
"""
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lb import judge as J  # noqa: E402
from lb import prereg as P  # noqa: E402

FIX = ROOT / "test" / "lb" / "fixtures" / "judge"
PY = sys.executable
TASKS = ["HAR", "KWS", "CIFAR10"]
CAPS = [256, 323, 406, 512, 645, 813, 1024]
TOL = 1e-12


def t_sf_odd(t, nu):
    """P(T_nu >= t) for odd nu, closed form (no scipy)."""
    th = math.atan(t / math.sqrt(nu))
    c, s = math.cos(th), math.sin(th)
    term = ser = c
    for k in range(1, (nu - 3) // 2 + 1):
        term *= (2 * k) / (2 * k + 1) * c * c
        ser += term
    return (1 - (2 / math.pi) * (th + s * ser)) / 2


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def run(n1name, n2name=None):
    return J.judge(load(n1name), None if n2name is None else load(n2name))


def cell(res, task, cap):
    hits = [c for c in res["n1"]["cells"] if c["task"] == task and c["cap_kib"] == cap]
    assert len(hits) == 1, (task, cap)
    return hits[0]


class Statistics(unittest.TestCase):
    def test_closed_form_matches_scipy(self):
        for t in (-0.75, 0.75, 2.25, 6.0):
            self.assertAlmostEqual(t_sf_odd(t, 9), float(stats.t.sf(t, 9)), delta=1e-14)
        self.assertAlmostEqual(t_sf_odd(1.0, 11), float(stats.t.sf(1.0, 11)), delta=1e-14)

    def test_mcnemar_known_tables(self):
        # b = 9, c = 1: n = 10, two-sided p = 2 * [C(10,0) + C(10,1)] / 2^10 = 2 * 11 / 1024 = 0.021484375
        self.assertEqual(J.mcnemar_exact(9, 1), 0.021484375)
        self.assertEqual(J.mcnemar_exact(9, 1), 2 * (math.comb(10, 0) + math.comb(10, 1)) / 2 ** 10)
        self.assertEqual(J.mcnemar_exact(1, 9), 0.021484375)
        # b = 10, c = 0: 2 * 1 / 1024 = 0.001953125
        self.assertEqual(J.mcnemar_exact(10, 0), 0.001953125)
        # b = c = 5: 2 * (1+10+45+120+210+252) / 1024 = 1276/1024 > 1 -> capped at 1
        self.assertEqual(J.mcnemar_exact(5, 5), 1.0)
        # b = 4, c = 1: n = 5, 2 * (1 + 5) / 32 = 0.375
        self.assertEqual(J.mcnemar_exact(4, 1), 0.375)
        self.assertEqual(J.mcnemar_exact(0, 0), 1.0)

    def test_holm_function(self):
        # p = [0.01, 0.04, 0.03, 0.005], m = 4. Sorted: 0.005 x4 = 0.02; 0.01 x3 = 0.03; 0.03 x2 = 0.06;
        # 0.04 x1 = 0.04 -> cumulative max 0.06. In input order: [0.03, 0.06, 0.06, 0.02].
        adj = J.holm([0.01, 0.04, 0.03, 0.005])
        for got, exp in zip(adj, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, exp, delta=TOL)
        self.assertEqual(J.holm([0.9, 0.8]), [1.0, 1.0])  # min(1, 2 * 0.8) = 1

    def test_classify_over_tested_caps(self):
        self.assertEqual(J._classify([256, 323, 406], [], []), "no event")
        self.assertEqual(J._classify([256, 323, 406], [], [323]), "no event")  # PIN: E_minus only
        self.assertEqual(J._classify([256, 323, 406], [323, 406], []), "band")
        self.assertEqual(J._classify([256, 323, 406], [256, 323, 406], []), "everywhere tested")
        self.assertEqual(J._classify([256, 323, 406], [256, 406], []), "multiple crossings")
        self.assertEqual(J._classify([256, 323, 406], [256], [406]), "multiple crossings")
        # PIN: an untested cap (512 infeasible) does not break contiguity: tested [406, 645] are neighbours
        self.assertEqual(J._classify([256, 406, 645], [406, 645], []), "band")


class N1Fixtures(unittest.TestCase):
    def assert_expected(self, res, name):
        exp = load(name + ".expected.json")
        self.assertEqual(res["status"], exp["status"], res["reasons"])
        self.assertEqual(res["n1_1"]["status"], exp["n1_1"])
        self.assertEqual(res["disposition"], exp["disposition"])
        self.assertEqual(res["n1"]["holm_family_size"], exp["holm_family_size"])
        self.assertEqual({t: res["n1_2"][t]["class"] for t in TASKS}, exp["classes"])
        for key in ("E_plus_caps", "E_minus_caps"):
            if key in exp:
                self.assertEqual({t: res["n1_2"][t][key] for t in TASKS}, exp[key])
        self.assertEqual(res["reasons"], [])

    def background_ok(self, c, floor=1.0):
        # mean 0, a 0.25: SE = 0.25/3; t_plus = (0 - floor)/SE = -12 floor; zero_sign t = 0 -> p = 2 * 0.5 = 1
        self.assertEqual(c["mean"], 0.0)
        self.assertAlmostEqual(c["se"], 0.25 / 3, delta=TOL)
        self.assertAlmostEqual(c["t_plus"], -12.0 * floor, delta=1e-9)
        self.assertAlmostEqual(c["p_zero"], 1.0, delta=TOL)
        self.assertFalse(c["E_plus"] or c["E_minus"] or c["zero_sign"])

    def test_pass_go(self):
        res = run("synthetic_pass_go.n1.json")
        self.assert_expected(res, "synthetic_pass_go")
        c = cell(res, "HAR", 512)
        # mean 2.0, a 0.5: SD = 0.5 sqrt(10/9) = 0.527046276694730, SE = 1/6, t = (2 - 1) * 6 = 6
        self.assertEqual(c["mean"], 2.0)
        self.assertAlmostEqual(c["sd"], 0.5 * math.sqrt(10 / 9), delta=TOL)
        self.assertAlmostEqual(c["se"], 1 / 6, delta=TOL)
        self.assertAlmostEqual(c["t_plus"], 6.0, delta=1e-9)
        p = t_sf_odd(6.0, 9)  # 1.0124966103e-4
        self.assertAlmostEqual(c["p_plus"], p, delta=1e-13)
        # smallest of 21 p values -> Holm multiplier 21: 0.0021262428817 < 0.05
        self.assertAlmostEqual(c["p_plus_holm"], 21 * p, delta=1e-12)
        self.assertTrue(c["E_plus"])
        self.assertEqual(res["n1_1"]["cells"], [["HAR", 512]])
        self.background_ok(cell(res, "KWS", 256))

    def test_fail_kill(self):
        res = run("synthetic_fail_kill.n1.json")
        self.assert_expected(res, "synthetic_fail_kill")
        tcrit = float(stats.t.ppf(0.95, 9))  # 1.833112932656237
        ub = 0.0 + tcrit * 0.25 / 3  # 0.152759411054686, below every floor (1.0, 1.25, 2.0)
        for t, f in (("HAR", 1.0), ("KWS", 1.25), ("CIFAR10", 2.0)):
            for cap in CAPS:
                c = cell(res, t, cap)
                self.assertEqual(c["floor"], f)
                self.assertAlmostEqual(c["upper_bound_95"], ub, delta=1e-12)
                self.assertTrue(c["upper_bound_below_floor"])
                self.background_ok(c, f)

    def test_noisy_miss_top_up_required_10_seeds(self):
        res = run("synthetic_noisy_seeds10.n1.json")
        self.assert_expected(res, "synthetic_noisy_seeds10")
        c = cell(res, "CIFAR10", 813)
        # mean 0.5, a 2: SE = 2/3, t_plus = (0.5 - 1) * 3/2 = -0.75; UB = 0.5 + 1.8331129 * 2/3 = 1.72207529 >= 1
        self.assertAlmostEqual(c["t_plus"], -0.75, delta=1e-9)
        self.assertAlmostEqual(c["p_plus"], t_sf_odd(-0.75, 9), delta=1e-13)
        self.assertAlmostEqual(c["upper_bound_95"], 0.5 + float(stats.t.ppf(0.95, 9)) * 2 / 3, delta=1e-12)
        self.assertFalse(c["upper_bound_below_floor"])
        self.assertEqual(res["n1"]["seed_set"], list(range(1, 11)))

    def test_noisy_inconclusive_12_seeds(self):
        res = run("synthetic_noisy_seeds12.n1.json")
        self.assert_expected(res, "synthetic_noisy_seeds12")
        c = cell(res, "HAR", 256)
        # n = 12: SD = 2 sqrt(12/11), SE = 2/sqrt(11) = 0.603022689, t_plus = -0.5 sqrt(11)/2 = -0.829156198, df 11
        self.assertEqual(c["df"], 11)
        self.assertAlmostEqual(c["se"], 2 / math.sqrt(11), delta=TOL)
        self.assertAlmostEqual(c["t_plus"], -0.25 * math.sqrt(11), delta=1e-9)
        self.assertAlmostEqual(c["p_plus"], t_sf_odd(-0.25 * math.sqrt(11), 11), delta=1e-13)
        self.assertAlmostEqual(c["upper_bound_95"], 0.5 + float(stats.t.ppf(0.95, 11)) * 2 / math.sqrt(11), delta=1e-12)
        self.assertEqual(res["n1"]["seed_set"], list(range(1, 13)))

    def test_holm_boundary(self):
        res = run("synthetic_holm_boundary.n1.json")
        self.assert_expected(res, "synthetic_holm_boundary")
        # HAR@256 and KWS@645: mean 1.75, a 1: SE = 1/3, t = 0.75 * 3 = 2.25, df 9.
        # Raw p = P(T_9 >= 2.25) = 0.025501630353475 < 0.05 (passes alpha unadjusted).
        # Holm, m = 21: the two tied smallest p get multipliers 21 and 20: 21p = 0.535534237422983,
        # 20p = 0.510032607069508 -> cumulative max keeps 21p for both. Every other p is ~1 -> adjusted 1.0.
        # Adjusted 0.5355 >= 0.05: not significant; UB = 1.75 + 1.8331129/3 = 2.36103764 >= 1 -> MISS.
        p = t_sf_odd(2.25, 9)
        self.assertAlmostEqual(p, 0.025501630353475, delta=1e-13)
        self.assertLess(p, 0.05)
        for t, cap in (("HAR", 256), ("KWS", 645)):
            c = cell(res, t, cap)
            self.assertAlmostEqual(c["t_plus"], 2.25, delta=1e-9)
            self.assertAlmostEqual(c["p_plus"], p, delta=1e-13)
            self.assertAlmostEqual(c["p_plus_holm"], 21 * p, delta=1e-12)
            self.assertAlmostEqual(c["p_plus_holm"], 0.535534237422983, delta=1e-12)
            self.assertFalse(c["E_plus"])
            self.assertAlmostEqual(c["upper_bound_95"], 1.75 + float(stats.t.ppf(0.95, 9)) / 3, delta=1e-12)
        self.assertEqual(cell(res, "CIFAR10", 1024)["p_plus_holm"], 1.0)
        self.assertEqual(sorted(res["n1_1"]["cells_not_below_floor"]), [["HAR", 256], ["KWS", 645]])

    def test_infeasible_cell(self):
        res = run("synthetic_infeasible.n1.json")
        self.assert_expected(res, "synthetic_infeasible")
        c = cell(res, "CIFAR10", 256)
        self.assertFalse(c["feasible"])
        self.assertEqual(c["infeasible_class"], "tied")
        self.assertFalse(c["in_holm_family"])
        self.assertNotIn("p_plus", c)
        self.assertIsNone(c["p_plus_holm"])
        self.assertEqual(res["n1_2"]["CIFAR10"]["untested_caps"], [256])
        # family m = 20, so the smallest p (HAR@323, t = 2.25) gets multiplier 20: 0.510032607069508
        h = cell(res, "HAR", 323)
        self.assertAlmostEqual(h["p_plus_holm"], 20 * t_sf_odd(2.25, 9), delta=1e-12)

    def test_unbounded_cell_excluded(self):
        res = run("synthetic_unbounded_excluded.n1.json")
        self.assert_expected(res, "synthetic_unbounded_excluded")
        for t in TASKS:
            c = cell(res, t, "unbounded")
            # mean 3, a 0.25: SE = 1/12, t_plus = 2 * 12 = 24 -> raw p 9.04e-10 (would pass), but excluded
            self.assertAlmostEqual(c["t_plus"], 24.0, delta=1e-9)
            self.assertAlmostEqual(c["p_plus"], t_sf_odd(24.0, 9), delta=1e-15)
            self.assertLess(c["p_plus"] * 21, 0.05)
            self.assertFalse(c["in_holm_family"])
            self.assertIsNone(c["p_plus_holm"])
            self.assertFalse(c["E_plus"])
        self.assertEqual(res["n1"]["n_cells_reported"], 24)

    def test_sd_zero(self):
        res = run("synthetic_sd_zero.n1.json")
        self.assert_expected(res, "synthetic_sd_zero")
        a, b, c, d = (cell(res, "HAR", cap) for cap in (256, 323, 406, 512))
        # constant d = 2: SD 0, mean - floor = 1 > 0 -> t_plus +inf, p_plus 0, Holm 0; UB = mean = 2
        self.assertEqual((a["sd"], a["se"], a["t_plus"], a["p_plus"], a["p_plus_holm"]), (0.0, 0.0, math.inf, 0.0, 0.0))
        self.assertEqual(a["upper_bound_95"], 2.0)
        self.assertTrue(a["E_plus"] and a["zero_sign"])
        # constant d = -2: t_plus -inf, p_plus 1; mean + floor = -1 < 0 -> t_minus -inf, p_minus 0; zero p 0
        self.assertEqual((b["t_plus"], b["p_plus"], b["t_minus"], b["p_minus"], b["p_zero"]), (-math.inf, 1.0, -math.inf, 0.0, 0.0))
        self.assertTrue(b["E_minus"])
        # constant d = 1 = floor: t_plus undefined (None), p_plus 1 (PIN); zero p 0 (mean 1 != 0)
        self.assertIsNone(c["t_plus"])
        self.assertEqual((c["p_plus"], c["p_zero"], c["upper_bound_95"]), (1.0, 0.0, 1.0))
        self.assertFalse(c["upper_bound_below_floor"])
        # constant d = 0: t_zero undefined, p_zero 1, p_plus 1 (t -inf), p_minus 1 (t +inf), UB 0
        self.assertIsNone(d["t_zero"])
        self.assertEqual((d["p_zero"], d["p_plus"], d["p_minus"], d["t_minus"], d["upper_bound_95"]), (1.0, 1.0, 1.0, math.inf, 0.0))

    def test_band_everywhere_multiple(self):
        res = run("synthetic_bands_a.n1.json")
        self.assert_expected(res, "synthetic_bands_a")
        # E_plus cells: mean 3, a 0.25 -> t_plus 24, raw p 9.04e-10; 12 such cells, max Holm multiplier 21 -> < 2e-8
        for t, cap in (("HAR", 323), ("KWS", 1024), ("CIFAR10", 256)):
            self.assertLess(cell(res, t, cap)["p_plus_holm"], 21 * t_sf_odd(24.0, 9) + 1e-15)

    def test_crossing_and_no_event(self):
        res = run("synthetic_bands_b.n1.json")
        self.assert_expected(res, "synthetic_bands_b")
        # mean -3, a 0.25: t_minus = (-3 + 1) * 12 = -24
        self.assertAlmostEqual(cell(res, "HAR", 1024)["t_minus"], -24.0, delta=1e-9)
        self.assertEqual(res["n1_2"]["CIFAR10"]["zero_sign_caps"], [406, 512])


class Errors(unittest.TestCase):
    def assert_error(self, name, needle):
        res = run(name)
        self.assertEqual(res["status"], "ERROR")
        self.assertEqual(res["disposition"], "ERROR")
        self.assertTrue(any(needle in r for r in res["reasons"]), res["reasons"])

    def test_nan(self):
        self.assert_error("synthetic_err_nan.n1.json", "non-finite")

    def test_seed_length_mismatch(self):
        self.assert_error("synthetic_err_seed_length.n1.json", "seed mismatch")

    def test_seed_set_mismatch_between_cells(self):
        self.assert_error("synthetic_err_seed_set.n1.json", "seed mismatch")

    def test_wrong_prereg_hash(self):
        self.assert_error("synthetic_err_wrong_hash.n1.json", "prereg_sha256")

    def test_task_outside_yaml(self):
        self.assert_error("synthetic_err_unknown_task.n1.json", "outside the YAML tasks")

    def test_missing_field(self):
        self.assert_error("synthetic_err_missing_field.n1.json", "missing field 'untied_config'")

    def test_seed_list_not_confirmation(self):
        res = run("synthetic_err_other_seeds.n1.json")
        self.assertEqual(res["n1_1"]["status"], "PASS")  # computed, but the seed list is not a registered one
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("neither the confirmation seeds", res["reasons"][0])

    def test_missing_cell_and_floor_below_one(self):
        x = load("synthetic_pass_go.n1.json")
        x["cells"] = x["cells"][1:]
        self.assertEqual(J.judge(x)["status"], "ERROR")
        y = load("synthetic_pass_go.n1.json")
        y["floors"]["KWS"] = 0.5
        res = J.judge(y)
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("below the YAML minimum floor", res["reasons"][0])

    def test_frozen_hash_is_the_yaml(self):
        self.assertEqual(load("synthetic_pass_go.n1.json")["prereg_sha256"], P.yaml_sha256())


class N2Fixtures(unittest.TestCase):
    N1 = "synthetic_fail_kill.n1.json"

    def check(self, name):
        res = run(self.N1, name + ".n2.json")
        exp = load(name + ".expected.json")
        self.assertEqual(res["n2_1"]["status"], exp["n2_1"])
        self.assertEqual(res["n2_2"]["status"], exp["n2_2"])
        self.assertEqual(res["n2_1"]["n_boundary_pairs"], exp["n_boundary_pairs"])
        if "b" in exp:
            for comp in res["n2_2"]["comparisons"]:
                self.assertEqual((comp["b"], comp["c"]), (exp["b"], exp["c"]))
        return res

    def test_perfect_predictor_pass(self):
        res = self.check("synthetic_n2_perfect")
        # 5 configs at 220000 B (window of cap 256 only; fit) + 5 at 1200000 B (window of cap 1024 only; no fit)
        # -> 10 boundary pairs; predictor exact: share 12/12 = 1, accuracy 10/10 = 1.
        # Baselines wrong on all 10 -> b = 10, c = 0, p = 2/1024 = 0.001953125; Holm m = 3 (all equal) -> 0.005859375
        self.assertEqual(res["n2_1"]["share_within_5pct"], 1.0)
        self.assertEqual(res["n2_1"]["boundary_fit_accuracy"], 1.0)
        for comp in res["n2_2"]["comparisons"]:
            self.assertEqual(comp["p"], 0.001953125)
            self.assertAlmostEqual(comp["p_holm"], 0.005859375, delta=TOL)
        self.assertEqual(res["status"], "KILL")  # N1 companion; N2 PASS does not change the disposition

    def test_biased_predictor_fail(self):
        res = self.check("synthetic_n2_biased")
        # predictor = 1.25 x measured: rel_err 0.25 everywhere -> share 0/12; 10 low pairs predicted 275000 > 262144
        # (wrong), 2 high pairs predicted not to fit (right): accuracy 2/12. Baselines exact: b = 0, c = 10 -> FAIL.
        self.assertEqual(res["n2_1"]["share_within_5pct"], 0.0)
        self.assertAlmostEqual(res["n2_1"]["boundary_fit_accuracy"], 2 / 12, delta=TOL)
        self.assertAlmostEqual(res["n2_1"]["max_rel_err"], 0.25, delta=TOL)
        for comp in res["n2_2"]["comparisons"]:
            self.assertAlmostEqual(comp["p_holm"], 0.005859375, delta=TOL)

    def test_share_exactly_090_is_miss(self):
        res = self.check("synthetic_n2_miss")
        # 18 exact + 2 at 1.10 x measured (still fit cap 256): share 18/20 = 0.90 (not < 0.90, not >= 0.95) -> MISS;
        # predictor and baselines all right: b = c = 0 -> p = 1 -> MISS
        self.assertEqual(res["n2_1"]["share_within_5pct"], 0.9)
        for comp in res["n2_2"]["comparisons"]:
            self.assertEqual((comp["p"], comp["p_holm"]), (1.0, 1.0))

    def test_no_boundary_pairs_error(self):
        res = self.check("synthetic_n2_no_boundary")
        self.assertEqual(res["status"], "ERROR")
        self.assertIn("N2.1 ERROR: no boundary pairs", res["reasons"])

    def test_n2_caps_must_match_yaml(self):
        x = load("synthetic_n2_perfect.n2.json")
        x["caps_kib"] = [256, 512]
        res = J.judge(load(self.N1), x)
        self.assertEqual((res["status"], res["n2_1"]["status"]), ("ERROR", "ERROR"))


class Cli(unittest.TestCase):
    def cli(self, *args):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "res.json"
            p = subprocess.run([PY, str(ROOT / "src" / "lb" / "judge.py"), *args, "--out", str(out)],
                               capture_output=True, text=True)
            def bad(tok):
                raise ValueError(tok)
            res = json.loads(out.read_text(), parse_constant=bad)  # strict JSON: no NaN / Infinity tokens
            return p.returncode, res

    def test_go_exit_0_with_metadata(self):
        n1 = FIX / "synthetic_pass_go.n1.json"
        rc, res = self.cli("--n1", str(n1), "--n2", str(FIX / "synthetic_n2_perfect.n2.json"))
        self.assertEqual(rc, 0)
        self.assertEqual((res["gate"], res["status"], res["n2_2"]["status"]), ("G6 judge", "GO", "PASS"))
        self.assertEqual(res["prereg_sha256"], P.yaml_sha256())
        self.assertEqual(res["judge_sha256"], J._sha_file(ROOT / "src" / "lb" / "judge.py"))
        self.assertEqual(res["input_sha256"]["n1"], J._sha_file(n1))
        for k in ("n1", "n1_1", "n1_2", "disposition", "n2_1", "reasons", "command", "utc"):
            self.assertIn(k, res)

    def test_infinite_t_written_as_string(self):
        rc, res = self.cli("--n1", str(FIX / "synthetic_sd_zero.n1.json"))
        self.assertEqual(rc, 0)
        c = [x for x in res["n1"]["cells"] if x["task"] == "HAR" and x["cap_kib"] == 256][0]
        self.assertEqual(c["t_plus"], "+inf")

    def test_error_exit_2(self):
        rc, res = self.cli("--n1", str(FIX / "synthetic_err_wrong_hash.n1.json"))
        self.assertEqual((rc, res["status"]), (2, "ERROR"))
        rc, res = self.cli("--n1", str(FIX / "synthetic_err_nan.n1.json"))
        self.assertEqual((rc, res["status"]), (2, "ERROR"))
        rc, res = self.cli("--n1", str(FIX / "synthetic_fail_kill.n1.json"), "--n2", str(FIX / "synthetic_n2_no_boundary.n2.json"))
        self.assertEqual((rc, res["status"], res["disposition"]), (2, "ERROR", "KILL"))

    def test_verdict_exit_codes(self):
        for name, status in (("synthetic_fail_kill", "KILL"), ("synthetic_noisy_seeds10", "MISS_TOP_UP_REQUIRED"),
                             ("synthetic_noisy_seeds12", "INCONCLUSIVE")):
            rc, res = self.cli("--n1", str(FIX / f"{name}.n1.json"))
            self.assertEqual((rc, res["status"]), (0, status))


if __name__ == "__main__":
    unittest.main()
