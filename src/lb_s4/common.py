"""Shared fail-closed readers of the S4 tools (the preregistration in force through lb.prereg.load(): registry,
memory_knobs, seeds, training.lr_selection, code_freeze_s3). Nothing here writes outside the caller's output file.

Checks implemented (each violation raises S4Error; the CLIs stop with exit 2 and write nothing):
  * prereg_sha256 of every input equals lb.prereg.yaml_sha256() (the YAML in force);
  * code_sha256 of a run or profile, restricted to the non-predictor files, equals the harness code-freeze record
    (lb.select_lr.check_run's rule; the record is itself verified against the tree by lb.freeze.verify first);
  * a code_freeze value recorded in an input equals lb.freeze.verify(...) of the same record names now
    (code_freeze_s3: confirmatory consumers stop with ERROR on a mismatch);
  * profiles: exactly one min-over-knobs profile.json per registry configuration, with the full knob set of
    memory_knobs (checkpointing only for non-reversible structures) and min_peak_bytes / argmin_knob consistent with
    lb.profiler.min_over_knobs's rule (lowest peak, then checkpointing off, then the larger microbatch).
"""
import datetime
import functools
import hashlib
import json
import math
import pathlib

from lb import freeze, prereg, registry
from lb.models import REVERSIBLE
from lb.select_lr import harness_code as _harness_code

ROOT = prereg.ROOT
CLASSES = ("tied", "untied")


class S4Error(RuntimeError):
    """ERROR of an S4 tool: the inputs are not exactly the expected, hash-consistent artifacts."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def inputs_sha(paths):
    return {freeze._key(p): sha256(p) for p in sorted({pathlib.Path(x) for x in paths})}


def write_x(path, obj):
    """Write JSON with open "x" (never overwrites); returns the sha256 of the written file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    return sha256(path)


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def read_json(path, what):
    path = pathlib.Path(path)
    if not path.is_file():
        raise S4Error(f"{what} {path} is missing")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise S4Error(f"{what} {path}: unreadable ({e})") from e


def read_tree(root, name):
    """[(path, dict)] for every file called `name` under root (sorted by path)."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise S4Error(f"{root} is not a directory")
    return [(p, read_json(p, name)) for p in sorted(root.rglob(name))]


# ------------------------------------------------------------------------------------------------ hashes
def check_prereg(path, doc, yaml_sha):
    if not isinstance(doc, dict) or doc.get("prereg_sha256") != yaml_sha:
        got = doc.get("prereg_sha256") if isinstance(doc, dict) else None
        raise S4Error(f"{path}: prereg_sha256 {got} != YAML in force {yaml_sha}")


def check_code(path, doc, harness):
    """code_sha256 restricted to the harness files (predictor*.py entries dropped) equals the harness record."""
    code = doc.get("code_sha256")
    if not isinstance(code, dict):
        raise S4Error(f"{path}: no code_sha256")
    got = {k: v for k, v in code.items() if not freeze._is_predictor(k)}
    if got != harness:
        diff = sorted(set(got.items()) ^ set(harness.items()))
        raise S4Error(f"{path}: code_sha256 differs from the harness record ({diff[:4]})")


def check_recorded_freeze(path, recorded, required=None):
    """A code_freeze dict recorded in an input must equal freeze.verify() of its record names now; `required` (a
    tuple of names) additionally fixes which names it must carry."""
    if not isinstance(recorded, dict) or not recorded:
        raise S4Error(f"{path}: code_freeze {recorded!r} is not a verified record set")
    if required is not None and set(recorded) != set(required):
        raise S4Error(f"{path}: code_freeze names {sorted(recorded)} != {sorted(required)}")
    try:
        now = freeze.verify(*sorted(recorded))
    except freeze.CodeFreezeError as e:
        raise S4Error(f"{path}: code-freeze records do not verify now: {e}") from e
    if recorded != now:
        raise S4Error(f"{path}: code_freeze {recorded} != current records {now}")


def check_complete(path, r):
    """A full-length run: max_steps null and truncated false (selection.rule and estimand use the final epoch;
    lb.train's --max-steps truncates training for smoke tests only)."""
    if r.get("max_steps") is not None or r.get("truncated") is not False:
        raise S4Error(f"{path}: max_steps {r.get('max_steps')!r} / truncated {r.get('truncated')!r}; "
                      "only full-length runs count")


def harness_and_records(*names):
    """Verify the named code-freeze records against the tree; return (harness code dict, {name: record sha256})."""
    try:
        recs = freeze.verify(*names)
        harness = _harness_code(freeze.load_record("harness"))
    except freeze.CodeFreezeError as e:
        raise S4Error(f"code-freeze records: {e}") from e
    return harness, recs


# ------------------------------------------------------------------------------------------------ registry
@functools.lru_cache(maxsize=1)
def registry_index():
    """{config_id: registry entry + "class"} over the YAML tasks (registry.enumerate; 3 x 248 = 744).
    Class tied iff structure in registry.tied_class (YAML), cross-checked against the registry's own flag."""
    p = prereg.load()
    tied = set(p["registry"]["tied_class"])
    out = {}
    for task in p["tasks"]:
        for c in registry.enumerate(task):
            cls = "tied" if c["structure"] in tied else "untied"
            if (cls == "tied") != bool(c["tied"]):
                raise S4Error(f"{c['config_id']}: class disagrees with lb.registry")
            if c["config_id"] in out:
                raise S4Error(f"duplicate registry config_id {c['config_id']}")
            out[c["config_id"]] = dict(c, **{"class": cls})
    n = len(p["tasks"]) * int(p["registry"]["configurations_per_task"])
    if len(out) != n:
        raise S4Error(f"registry has {len(out)} configurations != {n}")
    return out


def expected_knobs(structure, p):
    ck = [False] if structure in REVERSIBLE else [False, True]
    return {(int(m), c) for c in ck for m in p["memory_knobs"]["microbatch"]}


def _argmin_rule(ms):
    """lb.profiler.min_over_knobs's rule: lowest peak, then checkpointing off, then the larger microbatch."""
    return min(ms, key=lambda r: (r["peak_bytes"], r["checkpointing"], -r["microbatch"]))


def load_profiles(root, p, harness, records, yaml_sha):
    """{config_id: (path, profile)}: exactly one min-over-knobs profile.json per registry configuration.
    Each carries the prereg hash, code_sha256 equal to the harness record on the harness files, and code_freeze
    equal to the current harness + predictor records (lb.profiler records them for registry members; YAML
    code_freeze_s3: no registry configuration is profiled before both records exist)."""
    reg = registry_index()
    found = {}
    for path, prof in read_tree(root, "profile.json"):
        check_prereg(path, prof, yaml_sha)
        check_code(path, prof, harness)
        if prof.get("code_freeze") != records:
            raise S4Error(f"{path}: code_freeze {prof.get('code_freeze')} != current records {records}")
        cid = prof.get("config_id")
        if cid not in reg:
            raise S4Error(f"{path}: unexpected profile {cid!r} (not a registry configuration)")
        if cid in found:
            raise S4Error(f"{path}: duplicate profile for {cid} (also {found[cid][0]})")
        peak, ms, am = prof.get("min_peak_bytes"), prof.get("measurements"), prof.get("argmin_knob")
        if not is_int(peak) or peak <= 0:
            raise S4Error(f"{path}: min_peak_bytes {peak!r} is not a positive integer")
        if not isinstance(ms, list) or not ms or not isinstance(am, dict):
            raise S4Error(f"{path}: not a min-over-knobs profile (measurements, argmin_knob)")
        rows = []
        for m in ms:
            if not (isinstance(m, dict) and is_int(m.get("microbatch")) and isinstance(m.get("checkpointing"), bool)
                    and is_int(m.get("peak_bytes")) and m["peak_bytes"] > 0):
                raise S4Error(f"{path}: malformed measurement {str(m)[:120]}")
            rows.append(m)
        got = [(m["microbatch"], m["checkpointing"]) for m in rows]
        want = expected_knobs(reg[cid]["structure"], p)
        if len(got) != len(set(got)) or set(got) != want:
            raise S4Error(f"{path}: knobs {sorted(got)} != {sorted(want)}")
        best = _argmin_rule(rows)
        if best["peak_bytes"] != peak:
            raise S4Error(f"{path}: min_peak_bytes {peak} != min over its measurements {best['peak_bytes']}")
        if am != {"microbatch": best["microbatch"], "checkpointing": best["checkpointing"]}:
            raise S4Error(f"{path}: argmin_knob {am} disagrees with the profiler's tie rule")
        found[cid] = (path, prof)
    missing = sorted(set(reg) - set(found))
    if missing:
        raise S4Error(f"{len(missing)} of {len(reg)} registry profiles missing, e.g. {missing[:3]}")
    if len(found) != len(reg):
        raise S4Error(f"{len(found)} profiles != {len(reg)}")
    return found


# ------------------------------------------------------------------------------------------------ S3 artifacts
def load_lr_selected(path, p, yaml_sha):
    """{(task, optimizer, class): lr} from lr_selected.json (lb.select_lr output; training.lr_selection): kind,
    prereg hash, recorded code_freeze (harness) and every lr in lr_grid[optimizer]."""
    d = read_json(path, "lr selection")
    if d.get("kind") != "lr_selection":
        raise S4Error(f"{path}: kind {d.get('kind')!r} != 'lr_selection'")
    check_prereg(path, d, yaml_sha)
    check_recorded_freeze(path, d.get("code_freeze"), required=("harness",))
    out = {}
    for task in p["tasks"]:
        for opt in p["registry"]["optimizers"]:
            for cls in CLASSES:
                try:
                    lr = d["selections"][task][opt][cls]
                except (KeyError, TypeError) as e:
                    raise S4Error(f"{path}: no selected lr for ({task}, {opt}, {cls})") from e
                if not is_num(lr) or not any(float(lr) == float(v) for v in p["training"]["lr_grid"][opt]):
                    raise S4Error(f"{path}: ({task}, {opt}, {cls}) lr {lr!r} not in lr_grid[{opt}]")
                out[(task, opt, cls)] = float(lr)
    return out


def load_selection(path, yaml_sha):
    """select_cells output (kind cell_selection): prereg hash and recorded code_freeze (harness + predictor)."""
    d = read_json(path, "cell selection")
    if d.get("kind") != "cell_selection":
        raise S4Error(f"{path}: kind {d.get('kind')!r} != 'cell_selection'")
    check_prereg(path, d, yaml_sha)
    check_recorded_freeze(path, d.get("code_freeze"), required=("harness", "predictor"))
    if not isinstance(d.get("cells"), list) or not d["cells"]:
        raise S4Error(f"{path}: no cells")
    return d


def selected_configs(sel, path):
    """{config_id: lr} of every distinct selected configuration across all cells; one lr per configuration."""
    out = {}
    for cell in sel["cells"]:
        for cls in CLASSES:
            e = cell.get(cls)
            if e is None:
                continue
            cid, lr = e.get("config_id"), e.get("lr")
            if cid not in registry_index() or not is_num(lr):
                raise S4Error(f"{path}: cell ({cell.get('task')}, {cell.get('cap_kib')}) {cls}: bad entry {cid!r} {lr!r}")
            if registry_index()[cid]["class"] != cls:
                raise S4Error(f"{path}: {cid} selected as {cls} but belongs to {registry_index()[cid]['class']}")
            if cid in out and out[cid] != float(lr):
                raise S4Error(f"{path}: {cid} selected with two lrs {out[cid]} and {lr}")
            out[cid] = float(lr)
    if not out:
        raise S4Error(f"{path}: no configuration selected in any cell")
    return out
