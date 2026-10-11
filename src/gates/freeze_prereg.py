#!/usr/bin/env python3
"""Fail-closed freeze gate for PREREGISTERED_loopbudget_v1.yaml and CONTRIBUTION_CONTRACT.md.

Usage: freeze_prereg.py [--yaml PATH] [--contract PATH] [--out-dir DIR] (--check | --freeze)
Prints "GATE freeze_prereg: PASS|FAIL|ERROR" plus reasons and writes <out-dir>/freeze_receipt_<UTC>.json.
Exit codes: 0 PASS, 1 FAIL, 2 ERROR (missing file, unparsable YAML or contract FREEZE-KEYS block).
Standard library plus pyyaml only.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YAML = ROOT / "PREREGISTERED_loopbudget_v1.yaml"
DEFAULT_CONTRACT = ROOT / "CONTRIBUTION_CONTRACT.md"
DEFAULT_OUT = ROOT / "research" / "s2"

TASKS = ("HAR", "KWS", "CIFAR10")
STRUCTURES = {"shared", "two_groups", "reversible_shared", "untied", "reversible_untied", "wide_single"}
REQUIRED_HEADINGS = ("## First line", "## Claims", "## Forbidden wording")

# contract FREEZE-KEYS key -> YAML path
CONTRACT_MAP = {
    "protocol": ("protocol",),
    "version": ("version",),
    "caps_kib": ("caps_kib",),
    "effective_batch": ("memory_knobs", "effective_batch"),
    "microbatch": ("memory_knobs", "microbatch"),
    "widths": ("registry", "widths"),
    "wide_widths": ("registry", "wide_widths"),
    "depths": ("registry", "depths"),
    "optimizers": ("registry", "optimizers"),
    "epochs": ("training", "epochs"),
    "seeds_selection": ("seeds", "selection"),
    "seeds_confirmation": ("seeds", "confirmation"),
    "seeds_top_up": ("seeds", "top_up"),
    "alpha": ("inference", "alpha"),
    "configurations_per_task": ("registry", "configurations_per_task"),
}


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x):
    return (isinstance(x, (int, float))) and not isinstance(x, bool)


TYPES = {
    "str": (lambda x: isinstance(x, str)),
    "int": is_int,
    "num": is_num,
    "dict": (lambda x: isinstance(x, dict)),
    "list": (lambda x: isinstance(x, list)),
    "list_int": (lambda x: isinstance(x, list) and all(is_int(v) for v in x)),
    "list_str": (lambda x: isinstance(x, list) and all(isinstance(v, str) for v in x)),
}

# (path, type) in parent-before-child order
SCHEMA = [
    (("protocol",), "str"),
    (("version",), "int"),
    (("status",), "str"),
    (("central_claim",), "str"),
    (("measured_object",), "dict"),
    (("measured_object", "name"), "str"),
    (("measured_object", "unit"), "str"),
    (("measured_object", "includes"), "list"),
    (("measured_object", "excludes"), "list"),
    (("measured_object", "threads"), "int"),
    (("tasks",), "dict"),
]
for _t in TASKS:
    SCHEMA += [(("tasks", _t), "dict"), (("tasks", _t, "split"), "str"), (("tasks", _t, "classes"), "int")]
SCHEMA += [
    (("registry",), "dict"),
    (("registry", "widths"), "list_int"),
    (("registry", "wide_widths"), "list_int"),
    (("registry", "depths"), "list_int"),
    (("registry", "tied_class"), "list_str"),
    (("registry", "untied_class"), "list_str"),
    (("registry", "optimizers"), "list_str"),
    (("registry", "configurations_per_task"), "int"),
    (("memory_knobs",), "dict"),
    (("memory_knobs", "effective_batch"), "int"),
    (("memory_knobs", "microbatch"), "list_int"),
    (("memory_knobs", "checkpointing"), "list"),
    (("memory_knobs", "feasibility_rule"), "str"),
    (("training",), "dict"),
    (("training", "epochs"), "dict"),
]
for _t in TASKS:
    SCHEMA += [(("training", "epochs", _t), "int")]
SCHEMA += [
    (("training", "lr_grid"), "dict"),
    (("training", "lr_selection"), "str"),
    (("seeds",), "dict"),
    (("seeds", "selection"), "list_int"),
    (("seeds", "confirmation"), "list_int"),
    (("seeds", "top_up"), "list_int"),
    (("caps_kib",), "list_int"),
    (("selection",), "dict"),
    (("estimand",), "dict"),
    (("inference",), "dict"),
    (("inference", "alpha"), "num"),
    (("inference", "multiplicity"), "str"),
    (("inference", "events"), "dict"),
    (("inference", "events", "E_plus"), "str"),
    (("inference", "events", "E_minus"), "str"),
    (("inference", "events", "zero_sign"), "str"),
    (("floor",), "dict"),
    (("floor", "rule"), "str"),
    (("floor", "justification"), "str"),
    (("conjuncts",), "dict"),
]
for _c in ("N1.1", "N1.2", "N2.1", "N2.2"):
    SCHEMA += [(("conjuncts", _c), "dict")]
    if _c != "N1.2":
        SCHEMA += [(("conjuncts", _c, k), "str") for k in ("pass", "fail", "miss")]
SCHEMA += [
    (("judge",), "dict"),
    (("judge", "disposition"), "dict"),
    (("judge", "disposition", "GO"), "str"),
    (("judge", "disposition", "KILL"), "str"),
    (("judge", "disposition", "INCONCLUSIVE"), "str"),
    (("judge", "top_up"), "str"),
    (("judge", "early_stop"), "str"),
    (("classical_bars",), "list"),
    (("controls",), "dict"),
    (("interim_looks",), "str"),
    (("budget",), "dict"),
    (("forbidden_wording_source",), "str"),
]


def dotted(path):
    return ".".join(path)


def lookup(d, path):
    """Return (True, value) if path exists in nested dicts, else (False, None)."""
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


def canon(v):
    return json.dumps(v, sort_keys=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def check_schema(doc, reasons):
    bad = []  # paths reported missing/mistyped; children are skipped
    for path, typ in SCHEMA:
        if any(path[: len(b)] == b for b in bad):
            continue
        found, val = lookup(doc, path)
        if not found:
            reasons.append(f"missing key: {dotted(path)}")
            bad.append(path)
        elif not TYPES[typ](val):
            reasons.append(f"wrong type: {dotted(path)} (expected {typ}, got {type(val).__name__})")
            bad.append(path)
    if doc.get("protocol") != "loopbudget":
        reasons.append(f"protocol must be 'loopbudget', got {doc.get('protocol')!r}")
    if "status" in doc and doc["status"] not in ("DRAFT", "FROZEN"):
        reasons.append(f"status must be DRAFT or FROZEN, got {doc['status']!r}")
    tasks = doc.get("tasks")
    if isinstance(tasks, dict) and set(tasks) != set(TASKS):
        reasons.append(f"tasks must be exactly {list(TASKS)}, got {sorted(tasks)}")
    epochs = lookup(doc, ("training", "epochs"))[1]
    if isinstance(epochs, dict) and set(epochs) != set(TASKS):
        reasons.append(f"training.epochs must have exactly {list(TASKS)}, got {sorted(epochs)}")
    lr_grid = lookup(doc, ("training", "lr_grid"))[1]
    opts = lookup(doc, ("registry", "optimizers"))[1]
    if isinstance(lr_grid, dict) and isinstance(opts, list) and set(lr_grid) != set(opts):
        reasons.append(f"training.lr_grid keys {sorted(lr_grid)} != registry.optimizers {sorted(opts)}")


def check_registry(doc, reasons):
    reg = doc["registry"]
    widths, wide, depths = reg["widths"], reg["wide_widths"], reg["depths"]
    tied, untied = reg["tied_class"], reg["untied_class"]
    if set(tied) & set(untied) or set(tied) | set(untied) != STRUCTURES or len(tied) + len(untied) != len(STRUCTURES):
        reasons.append(f"registry classes must partition {sorted(STRUCTURES)}; got tied {tied}, untied {untied}")
    n_w = len(widths)
    t2 = len([t for t in depths if t >= 2])
    t4 = len([t for t in depths if t >= 4])
    counts = {
        "wide_single": len(widths) + len(wide),
        "shared": n_w * t2,
        "reversible_shared": n_w * t2,
        "untied": n_w * t2,
        "reversible_untied": n_w * t2,
        "two_groups": n_w * t4,
    }
    n_arch = sum(counts.values())
    total = n_arch * len(reg["optimizers"])
    if total != reg["configurations_per_task"]:
        reasons.append(
            f"registry count mismatch: recomputed {n_arch} architectures x {len(reg['optimizers'])} optimizers = {total} "
            f"!= registry.configurations_per_task {reg['configurations_per_task']} ({counts})")


def check_caps(doc, reasons):
    caps = doc["caps_kib"]
    if not caps or not all(is_int(c) for c in caps):
        reasons.append(f"caps_kib must be a non-empty list of integers, got {caps!r}")
        return
    if any(b <= a for a, b in zip(caps, caps[1:])):
        reasons.append(f"caps_kib not strictly increasing: {caps}")
    if any(c < 256 or c > 1024 for c in caps):
        reasons.append(f"caps_kib outside [256, 1024]: {caps}")


def check_microbatch(doc, reasons):
    eb = doc["memory_knobs"]["effective_batch"]
    mbs = doc["memory_knobs"]["microbatch"]
    if not is_int(eb) or eb <= 0:
        reasons.append(f"memory_knobs.effective_batch must be a positive int, got {eb!r}")
        return
    bad = [m for m in mbs if not is_int(m) or m <= 0 or eb % m != 0]
    if bad or not mbs:
        reasons.append(f"memory_knobs.microbatch values must divide effective_batch {eb}: offending {bad or mbs}")


def check_lr_grid(doc, reasons):
    """Each optimizer's grid: a list of at least 3 positive values in strictly increasing order (v1 to v3 used 3
    values; amendment v4, PI decision 2026-10-10, extends each grid to 5)."""
    for opt, grid in doc["training"]["lr_grid"].items():
        if not (isinstance(grid, list) and len(grid) >= 3 and all(is_num(v) and v > 0 for v in grid)
                and all(a < b for a, b in zip(grid, grid[1:]))):
            reasons.append(f"training.lr_grid.{opt} must be a list of at least 3 positive, strictly increasing "
                           f"values, got {grid!r}")


def check_seeds(doc, reasons):
    s = doc["seeds"]
    names = ("selection", "confirmation", "top_up")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(s[a]) & set(s[b]))
            if common:
                reasons.append(f"seed lists overlap: seeds.{a} and seeds.{b} share {common}")
    for a in names:
        if len(set(s[a])) != len(s[a]):
            reasons.append(f"seeds.{a} has duplicates: {s[a]}")


def check_alpha(doc, reasons):
    a = doc["inference"]["alpha"]
    if not (is_num(a) and 0 < a < 1):
        reasons.append(f"inference.alpha must be in (0, 1), got {a!r}")


def check_contract_keys(doc, block, reasons):
    for k in block:
        if k not in CONTRACT_MAP:
            reasons.append(f"contract FREEZE-KEYS extra key: {k}")
    for k, path in CONTRACT_MAP.items():
        if k not in block:
            reasons.append(f"contract FREEZE-KEYS missing key: {k}")
            continue
        found, val = lookup(doc, path)
        if not found:
            reasons.append(f"contract key {k}: YAML path {dotted(path)} missing")
        elif canon(val) != canon(block[k]):
            reasons.append(f"contract mismatch: {k}: contract {canon(block[k])} != YAML {dotted(path)} {canon(val)}")


def check_contract_text(doc, contract_text, yaml_dir, reasons):
    lines = contract_text.splitlines()
    for h in REQUIRED_HEADINGS:
        if not any(ln.startswith(h) for ln in lines):
            reasons.append(f"contract heading missing: {h}")
    src = doc["forbidden_wording_source"].split()
    if not src:
        reasons.append("forbidden_wording_source is empty")
        return
    p = yaml_dir / src[0]
    if not p.is_file():
        reasons.append(f"forbidden_wording_source file missing: {p}")
    elif "Forbidden wording" not in p.read_text(encoding="utf-8", errors="replace"):
        reasons.append(f"forbidden_wording_source file lacks 'Forbidden wording': {p}")


def extract_freeze_block(contract_text):
    """Return the parsed fenced yaml block after the '## FREEZE-KEYS' heading; raise ValueError if absent."""
    lines = contract_text.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.startswith("## FREEZE-KEYS")]
    if len(idx) != 1:
        raise ValueError(f"expected exactly one '## FREEZE-KEYS' heading, found {len(idx)}")
    i = idx[0] + 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != "```yaml":
        raise ValueError("no ```yaml fence directly after the FREEZE-KEYS heading")
    body = []
    i += 1
    while i < len(lines) and lines[i].strip() != "```":
        body.append(lines[i])
        i += 1
    if i >= len(lines):
        raise ValueError("unterminated ```yaml fence in FREEZE-KEYS")
    block = yaml.safe_load("\n".join(body))
    if not isinstance(block, dict):
        raise ValueError("FREEZE-KEYS block is not a mapping")
    return block


def guarded(label, fn, reasons, *args):
    try:
        fn(*args, reasons)
    except Exception as e:  # malformed input makes the check fail, never pass
        reasons.append(f"check {label} not evaluable: {type(e).__name__}: {e}")


def write_receipt(out_dir, receipt):
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    n = 0
    while True:
        name = f"freeze_receipt_{stamp}.json" if n == 0 else f"freeze_receipt_{stamp}_{n}.json"
        p = out_dir / name
        try:
            with open(p, "x", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
                f.write("\n")
            return p
        except FileExistsError:
            n += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--yaml", default=str(DEFAULT_YAML))
    ap.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)

    t0 = utc_now()
    yaml_path = Path(args.yaml).resolve()
    contract_path = Path(args.contract).resolve()
    out_dir = Path(args.out_dir).resolve()
    mode_name = "freeze" if args.freeze else "check"
    errors, reasons = [], []
    yaml_bytes = contract_bytes = None

    for p in (yaml_path, contract_path):
        if not p.is_file():
            errors.append(f"missing file: {p}")
    if yaml_path.is_file():
        yaml_bytes = yaml_path.read_bytes()
    if contract_path.is_file():
        contract_bytes = contract_path.read_bytes()

    doc = block = None
    if yaml_bytes is not None:
        try:
            doc = yaml.safe_load(yaml_bytes.decode("utf-8"))
            if not isinstance(doc, dict):
                errors.append("unparsable YAML: top level is not a mapping")
        except Exception as e:
            errors.append(f"unparsable YAML: {type(e).__name__}: {e}")
    if contract_bytes is not None:
        try:
            block = extract_freeze_block(contract_bytes.decode("utf-8"))
        except Exception as e:
            errors.append(f"unparsable contract FREEZE-KEYS block: {type(e).__name__}: {e}")

    sha_file = yaml_path.with_suffix(".sha256")
    if not errors:
        check_schema(doc, reasons)
        guarded("registry_count", check_registry, reasons, doc)
        guarded("caps_kib", check_caps, reasons, doc)
        guarded("microbatch", check_microbatch, reasons, doc)
        guarded("lr_grid", check_lr_grid, reasons, doc)
        guarded("seeds", check_seeds, reasons, doc)
        guarded("alpha", check_alpha, reasons, doc)
        guarded("contract_keys", check_contract_keys, reasons, doc, block)
        guarded("contract_text", check_contract_text, reasons, doc,
                contract_bytes.decode("utf-8"), yaml_path.parent)
        if args.freeze:
            if doc.get("status") != "FROZEN":
                reasons.append(f"--freeze requires status FROZEN in the YAML, got {doc.get('status')!r}")
            if sha_file.exists():
                reasons.append(f"freeze receipt exists; never overwrite: {sha_file}")

    if errors:
        status, code, all_reasons = "ERROR", 2, errors
    elif reasons:
        status, code, all_reasons = "FAIL", 1, reasons
    else:
        status, code, all_reasons = "PASS", 0, []

    if status == "PASS" and args.freeze:
        # hash exactly the validated bytes; refuse if a file changed during validation
        if yaml_path.read_bytes() != yaml_bytes or contract_path.read_bytes() != contract_bytes:
            status, code, all_reasons = "FAIL", 1, ["input file changed during validation"]
        else:
            rel_c = os.path.relpath(contract_path, yaml_path.parent)
            text = (f"{sha256_bytes(yaml_bytes)}  {yaml_path.name}\n"
                    f"{sha256_bytes(contract_bytes)}  {rel_c}\n")
            try:
                with open(sha_file, "x", encoding="utf-8") as f:
                    f.write(text)
            except FileExistsError:
                status, code, all_reasons = "FAIL", 1, [f"freeze receipt exists; never overwrite: {sha_file}"]

    receipt = {
        "gate": "freeze_prereg",
        "mode": mode_name,
        "status": status,
        "reasons": all_reasons,
        "yaml_path": str(yaml_path),
        "yaml_sha256": sha256_bytes(yaml_bytes) if yaml_bytes is not None else None,
        "contract_path": str(contract_path),
        "contract_sha256": sha256_bytes(contract_bytes) if contract_bytes is not None else None,
        "script_sha256": sha256_bytes(Path(__file__).resolve().read_bytes()),
        "command": shlex.join(sys.argv if argv is None else [sys.argv[0]] + list(argv)),
        "utc_start": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "utc_end": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exit_status": code,
    }
    if status == "PASS" and args.freeze:
        receipt["sha256_file"] = str(sha_file)

    print(f"GATE freeze_prereg: {status}")
    for r in all_reasons:
        print(f"  - {r}")
    try:
        rp = write_receipt(out_dir, receipt)
        print(f"receipt: {rp}")
    except OSError as e:
        print(f"GATE freeze_prereg: ERROR (receipt not written: {e})")
        return 2
    return code


if __name__ == "__main__":
    sys.exit(main())
