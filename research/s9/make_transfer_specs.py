"""Launch specs (JSONL, lb.launch format) for the cap-respecting transfer check (research/s9/TRANSFER_PLAN_v1.md).
Writes specs only; launches nothing.

  ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s9/make_transfer_specs.py --design {seed1,seeds123,precondE} --out OUT.jsonl

Designs:
  seed1     the 22 distinct configurations selected in the 21 finite-cap cells (research/s4/selection.json), seed 1,
            at the recorded argmin knob (microbatch 1; --checkpointing when argmin_knob.checkpointing is true);
  seeds123  the same 22 configurations, seeds 1, 2 and 3;
  precondE  precondition E: seed-1 microbatch-64 reruns, without checkpointing, of KWS|shared|d32|T2|adafactor and
            CIFAR10|shared|d24|T2|adafactor.
Outputs: result/s9/transfer/<TASK>__<structure>__d<d>__T<T>__<opt>__lr<lr>__s<seed>__m1[_ckpt] and
result/s9/transfer_env/<archived run name>. Order: longest first by the probe's per-config estimate
(research/s9/transfer_probe/timing_20261014T070439Z.json; additive estimate for microbatch 1, archived_wall_s for the
microbatch-64 reruns), ties by run_id. Refuses to overwrite OUT.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SELECTION = "research/s4/selection.json"
PROBE = "research/s9/transfer_probe/timing_20261014T070439Z.json"
CONFIRM = "result/s4/confirm"
PRECOND_E = ("KWS|shared|d32|T2|adafactor", "CIFAR10|shared|d24|T2|adafactor")
SEEDS = {"seed1": (1,), "seeds123": (1, 2, 3), "precondE": (1,)}


def stem(config_id):
    return config_id.replace("|", "__")


def fmt_lr(lr):
    return repr(float(lr))


def selected_configs(sel):
    """{config_id: entry} over the finite-cap cells; asserts lr and argmin knob agree across cells."""
    cfgs = {}
    finite = [c for c in sel["cells"] if c["cap_kib"] != "unbounded"]
    assert len(finite) == 21, f"{len(finite)} finite cells, expected 21"
    for c in finite:
        for cls in ("tied", "untied"):
            e = c[cls]
            assert e is not None, f"cell {c['task']} {c['cap_kib']}: no {cls} entry"
            prev = cfgs.setdefault(e["config_id"], e)
            assert prev["lr"] == e["lr"] and prev["argmin_knob"] == e["argmin_knob"], e["config_id"]
    return cfgs


def archived_name(config_id, lr, seed):
    return f"{stem(config_id)}__lr{fmt_lr(lr)}__s{seed}"


def check_archive(cfgs, root):
    """Every configuration's archived confirm directory carries the selection lr, and seeds 1-3 exist (DONE.json and
    result.json, microbatch 64, no checkpointing, the same lr)."""
    for cid, e in cfgs.items():
        dirs = sorted((root / CONFIRM).glob(f"{stem(cid)}__lr*__s*"))
        lrs = {d.name[len(stem(cid)) + 4:].rsplit("__s", 1)[0] for d in dirs}
        assert lrs == {fmt_lr(e["lr"])}, f"{cid}: archived lr {lrs} != selection {fmt_lr(e['lr'])}"
        for s in (1, 2, 3):
            d = root / CONFIRM / archived_name(cid, e["lr"], s)
            assert (d / "DONE.json").exists() and (d / "result.json").exists(), f"missing archived run {d}"
            r = json.loads((d / "result.json").read_text())
            assert (r["config_id"], r["seed"], r["microbatch"], r["checkpointing"]) == (cid, s, 64, False), d
            assert r["lr"] == e["lr"], f"{d}: result lr {r['lr']} != selection {e['lr']}"


def build(design, root=ROOT):
    sel = json.loads((root / SELECTION).read_text())
    probe = {r["config_id"]: r for r in json.loads((root / PROBE).read_text())["rows"]}
    cfgs = selected_configs(sel)
    assert len(cfgs) == 22, f"{len(cfgs)} configurations, expected 22"
    n_ckpt = sum(bool(e["argmin_knob"]["checkpointing"]) for e in cfgs.values())
    assert (n_ckpt, len(cfgs) - n_ckpt) == (18, 4), f"checkpointing {n_ckpt}, without {len(cfgs) - n_ckpt}"
    assert all(e["argmin_knob"]["microbatch"] == 1 for e in cfgs.values()), "an argmin knob is not microbatch 1"
    assert set(probe) == set(cfgs), "probe rows do not match the 22 configurations"
    for cid, e in cfgs.items():
        assert probe[cid]["lr"] == e["lr"] and probe[cid]["argmin_knob"] == e["argmin_knob"], cid
    check_archive(cfgs, root)

    specs = []
    if design == "precondE":
        for cid in PRECOND_E:
            e = cfgs[cid]
            name = archived_name(cid, e["lr"], 1)
            args = ["--config-id", cid, "--lr", fmt_lr(e["lr"]), "--seed", "1", "--mode", "confirm",
                    "--microbatch", "64"]
            specs.append(({"run_id": f"transfer_env/{name}", "module": "lb.train",
                           "out": f"result/s9/transfer_env/{name}", "args": args},
                          probe[cid]["archived_wall_s"]))
    else:
        for cid, e in cfgs.items():
            ckpt = bool(e["argmin_knob"]["checkpointing"])
            for s in SEEDS[design]:
                name = archived_name(cid, e["lr"], s) + "__m1" + ("_ckpt" if ckpt else "")
                args = ["--config-id", cid, "--lr", fmt_lr(e["lr"]), "--seed", str(s), "--mode", "confirm",
                        "--microbatch", "1"] + (["--checkpointing"] if ckpt else [])
                specs.append(({"run_id": f"transfer/{name}", "module": "lb.train",
                               "out": f"result/s9/transfer/{name}", "args": args},
                              probe[cid]["estimate"]["additive_s"]))
    specs.sort(key=lambda x: (-x[1], x[0]["run_id"]))
    ids = [s["run_id"] for s, _ in specs]
    assert len(ids) == len(set(ids)), "duplicate run_id"
    want = {"seed1": 22, "seeds123": 66, "precondE": 2}[design]
    assert len(specs) == want, f"{len(specs)} specs, expected {want}"
    return specs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--design", required=True, choices=("seed1", "seeds123", "precondE"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    specs = build(a.design)
    with open(out, "x") as f:
        for s, _ in specs:
            f.write(json.dumps(s) + "\n")
    est = sum(e for _, e in specs)
    print(json.dumps({"design": a.design, "out": str(out), "n": len(specs),
                      "n_checkpointing": sum("--checkpointing" in s["args"] for s, _ in specs),
                      "estimate_core_hours": est / 3600, "first": specs[0][0]["run_id"],
                      "last": specs[-1][0]["run_id"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
