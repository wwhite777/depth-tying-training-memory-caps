"""S3 tests for src/lb/specs.py (card S3-T2): counts, uniqueness, run-id format, launcher compatibility, and that every
lrcal and floor spec passes lb.train's guards (check_guards with freeze.verify patched; nothing is trained).

Run (CPU, one thread):
  cd /home/wjeong/loopbudget && CUDA_VISIBLE_DEVICES="" python -m unittest test/lb/test_specs.py -v
Every check has an in-file negative control showing that it can fail.
"""
import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lb import freeze, launch, prereg, specs, train  # noqa: E402

P = prereg.load()
REPORT = {}
RESULT_BEFORE = sorted((ROOT / "result").rglob("*"))
GRID = {o: [float(x) for x in P["training"]["lr_grid"][o]] for o in P["registry"]["optimizers"]}
# counts from the YAML, computed independently of lb.specs (v4: 120, 15, 18)
COUNTS = {"lrcal": len(P["tasks"]) * 2 * sum(len(GRID[o]) for o in GRID) * len(P["seeds"]["selection"]),
          "floor": len(P["tasks"]) * len(train.FLOOR_SEEDS),
          "bars": len(P["tasks"]) * (1 + len(GRID["adam"]))}
OUT_ROOT = f"result/s3/v{int(P['version'])}"
RUN_ID = re.compile(r"^(lrcal|floor|bars)/(HAR|KWS|CIFAR10)__[a-z_]+__[a-z_]+__lr[0-9.e-]+__s\d+$")


def tearDownModule():
    print("\nREPORT " + json.dumps(REPORT, indent=1, default=str))


def args_of(spec):
    a = dict(zip(spec["args"][::2], spec["args"][1::2]))
    return a["--config-id"], float(a["--lr"]), int(a["--seed"]), a["--mode"]


def guard(spec):
    """lb.train's guards for a spec, with verify patched (records are not needed and nothing runs)."""
    cid, lr, seed, mode = args_of(spec)
    calls = []
    with mock.patch.object(freeze, "verify", side_effect=lambda *n: calls.append(n) or {k: "x" for k in n}):
        train.check_guards(train.parse_config_id(cid, P), mode, lr, seed, 64, False, P)
    return calls


class SpecCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)
        self.lr_path = self.tmp / "lr_selected.json"
        sel = {t: {o: {c: float(P["training"]["lr_grid"][o][1]) for c in ("untied", "tied")}
                   for o in P["registry"]["optimizers"]} for t in P["tasks"]}
        self.lr_path.write_text(json.dumps({"kind": "lr_selection", "prereg_sha256": prereg.yaml_sha256(),
                                            "selections": sel}))

    def tearDown(self):
        self._td.cleanup()

    def all_specs(self):
        return {"lrcal": specs.lrcal_specs(P), "floor": specs.floor_specs(self.lr_path, P), "bars": specs.bars_specs(P)}


class TestSpecs(SpecCase):
    def test_counts_ids_and_format(self):
        sp = self.all_specs()
        self.assertEqual(COUNTS, {"lrcal": 120, "floor": 15, "bars": 18})
        self.assertEqual({k: len(v) for k, v in sp.items()}, COUNTS)
        self.assertEqual(OUT_ROOT, "result/s3/v4")
        self.assertEqual(specs.out_root(P), OUT_ROOT)
        self.assertEqual(specs.out_root(dict(P, version=3)), "result/s3/v3")   # negative control: version-driven
        ids = [s["run_id"] for v in sp.values() for s in v]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len({s["out"] for v in sp.values() for s in v}), len(ids))
        for kind, v in sp.items():
            for s in v:
                self.assertRegex(s["run_id"], RUN_ID)
                self.assertTrue(s["run_id"].startswith(kind + "/"))
                self.assertEqual(s["out"], OUT_ROOT + "/" + s["run_id"])
                self.assertNotEqual(s["out"], "result/s3/" + s["run_id"])       # never the v3 location
                self.assertEqual(s["module"], "lb.train")
                self.assertNotIn("--out", s["args"])
        lr = sp["lrcal"]
        self.assertEqual({args_of(s)[3] for s in lr}, {"lrcal"})
        self.assertEqual({args_of(s)[2] for s in lr}, {0})
        self.assertEqual({args_of(s)[0] for s in lr},
                         {f"{t}|{st}|d32|T4|{o}" for t in P["tasks"] for st in ("untied", "shared")
                          for o in P["registry"]["optimizers"]})
        for s in lr:
            cid, rate, _, _ = args_of(s)
            self.assertIn(rate, [float(x) for x in P["training"]["lr_grid"][cid.split("|")[4]]])
        fl = sp["floor"]
        self.assertEqual(sorted((args_of(s)[0], args_of(s)[2]) for s in fl),
                         sorted((f"{t}|untied|d32|T4|adam", seed) for t in P["tasks"] for seed in range(101, 106)))
        self.assertEqual({args_of(s)[1] for s in fl}, {float(P["training"]["lr_grid"]["adam"][1])})
        bars = sp["bars"]
        self.assertEqual(sorted(args_of(s)[0] for s in bars),
                         sorted([f"{t}|majority" for t in P["tasks"]] + [f"{t}|linear" for t in P["tasks"]] * len(GRID["adam"])))
        self.assertEqual({(args_of(s)[2], args_of(s)[3]) for s in bars}, {(0, "dev")})
        self.assertEqual(sorted(s["run_id"] for s in bars if "HAR" in s["run_id"]),
                         ["bars/HAR__linear__adam__lr0.0003__s0", "bars/HAR__linear__adam__lr0.001__s0",
                          "bars/HAR__linear__adam__lr0.003__s0", "bars/HAR__linear__adam__lr0.01__s0",
                          "bars/HAR__linear__adam__lr0.03__s0", "bars/HAR__majority__none__lr0__s0"])
        # negative control: the uniqueness check detects a duplicate
        dup = [s["run_id"] for s in lr] + [lr[0]["run_id"]]
        self.assertNotEqual(len(dup), len(set(dup)))
        REPORT["counts"] = {k: len(v) for k, v in sp.items()}
        REPORT["examples"] = {k: v[0] for k, v in sp.items()}

    def test_every_spec_passes_guards(self):
        sp = self.all_specs()
        verified = {}
        for kind, v in sp.items():
            for s in v:
                verified.setdefault(kind, set()).update(guard(s))
        self.assertEqual(verified, {"lrcal": {("harness",)}, "floor": {("harness",)}, "bars": set()})
        # negative controls: a spec mutated off the protocol fails the guard
        bad = json.loads(json.dumps(sp["lrcal"][0]))
        bad["args"][bad["args"].index("--lr") + 1] = "0.002"
        with self.assertRaises(train.GuardViolation):
            guard(bad)
        bad = json.loads(json.dumps(sp["floor"][0]))
        bad["args"][bad["args"].index("--seed") + 1] = "1"
        with self.assertRaises(train.GuardViolation):
            guard(bad)
        REPORT["guards"] = {k: sorted(v) for k, v in verified.items()}

    def test_write_cli_and_launcher_reads(self):
        res = {}
        for kind in specs.KINDS:
            out = self.tmp / f"{kind}.jsonl"
            argv = ["--kind", kind, "--out", str(out)] + (["--lr", str(self.lr_path)] if kind == "floor" else [])
            with contextlib.redirect_stdout(io.StringIO()):
                res[kind] = specs.main(argv)
            got = launch.read_specs(out)                       # the launcher accepts the file
            self.assertEqual(len(got), COUNTS[kind])
            self.assertTrue(all(str(g["out"]).startswith(str(ROOT / "result" / "s3" / "v4") + "/") for g in got))
            before = out.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(specs.main(argv), 2)          # never overwrites
            self.assertEqual(out.read_bytes(), before)
        with contextlib.redirect_stderr(io.StringIO()):
            res["floor_without_lr"] = specs.main(["--kind", "floor", "--out", str(self.tmp / "x.jsonl")])
        self.assertEqual(res, {"lrcal": 0, "floor": 0, "bars": 0, "floor_without_lr": 2})
        self.assertFalse((self.tmp / "x.jsonl").exists())
        self.assertEqual(sorted((ROOT / "result").rglob("*")), RESULT_BEFORE)   # specs never create run dirs
        REPORT["cli"] = res


if __name__ == "__main__":
    unittest.main()
