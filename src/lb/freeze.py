"""S3 code-freeze records (the preregistration in force (lb.prereg; PREREGISTERED_loopbudget_v4.yaml): code_freeze_s3; DECISION_LOG.md 2026-10-10T05:52:05Z).

Two records in RECORD_DIR = research/s3/v<version>/ (version = the top-level "version" of the YAML in force:
research/s3/v4/ under v4; the v3 records stay in research/s3/):
  CODE_FREEZE_harness.json   every src/lb/*.py except files named predictor*.py (training harness, profiler, launcher,
                             S3 scripts), written before the S3 learning-rate and floor runs;
  CODE_FREEZE_predictor.json src/lb/predictor*.py plus any calibration artifact paths passed to the writer, written
                             before any registry configuration is profiled.
Record: {"name", "files": {path relative to the project root (absolute if outside it): sha256}, "prereg_sha256",
"written_utc"}; written with open(..., "x"), so an existing record is never overwritten.

verify(*names) raises CodeFreezeError on a missing record, any sha256 mismatch (a recorded file, or the record's
prereg_sha256 against the YAML in force), or a changed file set (harness: src/lb/*.py minus predictor*.py; predictor:
src/lb/predictor*.py); it returns {name: sha256 of the record file}.

CLI (from the project root, src/ on PYTHONPATH; the conductor alone writes the real records):
  python -m lb.freeze --write harness|predictor [--extra PATH ...]
  python -m lb.freeze --verify harness [predictor]
Exit codes: 0 done; 2 ERROR (CodeFreezeError, or a record that already exists).
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import sys

from lb import prereg

ROOT = prereg.ROOT
SRC_LB = ROOT / "src" / "lb"


def versioned_dir(*parts, p=None):
    """ROOT/<parts>/v<version> of the YAML in force (read without the hash check, which every reader and writer of a
    record applies itself)."""
    p = p or prereg.load(verify_hash=False)
    return ROOT.joinpath(*parts) / f"v{int(p['version'])}"


RECORD_DIR = versioned_dir("research", "s3")
NAMES = ("harness", "predictor")


class CodeFreezeError(RuntimeError):
    pass


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _src_py():
    """Every src/lb/*.py (the file-listing function; tests patch it)."""
    return sorted(SRC_LB.glob("*.py"))


def _is_predictor(path):
    return pathlib.Path(path).name.startswith("predictor")


def harness_files():
    return [f for f in _src_py() if not _is_predictor(f)]


def predictor_files():
    return [f for f in _src_py() if _is_predictor(f)]


def _key(path):
    path = pathlib.Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve(key):
    path = pathlib.Path(key)
    return path if path.is_absolute() else ROOT / path


def record_path(name):
    if name not in NAMES:
        raise CodeFreezeError(f"record name {name!r} not in {NAMES}")
    return pathlib.Path(RECORD_DIR) / f"CODE_FREEZE_{name}.json"


def _src_py_keys(files, predictor):
    """Recorded keys that are src/lb/*.py files of the given kind."""
    out = set()
    for k in files:
        p = pathlib.Path(k)
        if not p.is_absolute() and p.parent == pathlib.Path("src/lb") and p.suffix == ".py" \
                and _is_predictor(p) == predictor:
            out.add(k)
    return out


def write_record(name, extra_paths=()):
    """Write CODE_FREEZE_<name>.json (never overwrites). Returns (record path, sha256 of the record file)."""
    path = record_path(name)
    if name == "harness":
        if extra_paths:
            raise CodeFreezeError("the harness record takes no extra paths")
        paths = harness_files()
    else:
        paths = predictor_files() + [pathlib.Path(x) for x in extra_paths]
    if not paths:
        raise CodeFreezeError(f"{name}: no files to record")
    files = {}
    for f in paths:
        f = pathlib.Path(f)
        if not f.is_file():
            raise CodeFreezeError(f"{name}: {f} is not a file")
        k = _key(f)
        if k in files:
            raise CodeFreezeError(f"{name}: {k} listed twice")
        files[k] = _sha256(f)
    prereg.verify()
    rec = {"name": name, "files": files, "prereg_sha256": prereg.yaml_sha256(),
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return path, _sha256(path)


def load_record(name):
    path = record_path(name)
    if not path.is_file():
        raise CodeFreezeError(f"code-freeze record {path} is missing")
    rec = json.loads(path.read_text())
    if rec.get("name") != name or not isinstance(rec.get("files"), dict) or not rec["files"]:
        raise CodeFreezeError(f"{path}: malformed record")
    return rec


def verify(*names):
    """Raise CodeFreezeError unless every named record exists and matches; return {name: record sha256}."""
    if not names:
        raise CodeFreezeError("verify() needs at least one record name")
    out = {}
    yaml_sha = prereg.yaml_sha256()
    for name in names:
        rec = load_record(name)
        path = record_path(name)
        if rec.get("prereg_sha256") != yaml_sha:
            raise CodeFreezeError(f"{name}: record prereg_sha256 {rec.get('prereg_sha256')} != YAML in force {yaml_sha}")
        for k, h in sorted(rec["files"].items()):
            f = _resolve(k)
            if not f.is_file():
                raise CodeFreezeError(f"{name}: recorded file {k} is missing")
            got = _sha256(f)
            if got != h:
                raise CodeFreezeError(f"{name}: {k} sha256 {got} != recorded {h}")
        if name == "harness":
            current, recorded = {_key(f) for f in harness_files()}, set(rec["files"])
        else:
            current, recorded = {_key(f) for f in predictor_files()}, _src_py_keys(rec["files"], predictor=True)
        if current != recorded:
            raise CodeFreezeError(f"{name}: file set changed; unrecorded {sorted(current - recorded)}, "
                                  f"missing {sorted(recorded - current)}")
        out[name] = _sha256(path)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="write or verify the S3 code-freeze records")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", choices=NAMES)
    g.add_argument("--verify", nargs="+", choices=NAMES)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args(argv)
    try:
        if a.write:
            path, h = write_record(a.write, a.extra)
            print(json.dumps({"written": str(path), "sha256": h,
                              "files": sorted(load_record(a.write)["files"])}, indent=1))
        else:
            if a.extra:
                ap.error("--extra applies to --write only")
            print(json.dumps(verify(*a.verify), indent=1))
    except (CodeFreezeError, FileExistsError, prereg.PreregMismatch) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
