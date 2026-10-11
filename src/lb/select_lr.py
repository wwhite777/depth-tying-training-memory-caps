"""S3 learning-rate selection (the preregistration in force (lb.prereg; PREREGISTERED_loopbudget_v4.yaml): training.lr_grid, training.lr_selection,
training.reference_configs, seeds.selection, code_freeze_s3).

  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.select_lr --runs DIR --out research/s3/v4/lr_selected.json

Reads every result.json under DIR (v4: result/s3/v4/lrcal) and requires exactly the lrcal runs of the YAML: tasks x optimizers x 2 reference
configurations (classes untied and tied) x len(lr_grid[optimizer]) learning rates, seed 0, mode lrcal (v4: 3 x 4 x 2
x 5 = 120; the count is derived from the YAML, expected_lrcal). Every run must have status ok or
diverged, a finite final_val_acc, prereg_sha256 equal to the YAML in force, and code_sha256 equal to the files of the
harness code-freeze record (verified against the current tree first; predictor*.py entries of a run are not part of
the harness and are not compared). Per (task, optimizer, class) the lr with the highest final_val_acc is selected,
ties to the smaller lr (a diverged run keeps its accuracy, non-finite logits counted wrong by lb.train).
Edge flags (v4 training.lr_selection: "a selection at an edge of the grid is reported and the grid is not extended
again"): edge_flags[task][optimizer][class] = "first" or "last" when the selected lr equals lr_grid[optimizer][0] or
[-1], else null; n_at_edge counts them. The flags are reported only; the selection is unchanged.
Output (open "x", never overwritten): selections, edge flags, the per-run table, the sha256 of every input.
A missing run, a duplicate, an unexpected run, a wrong hash, a wrong mode or status is an ERROR: exit 2, no output.
"""
import argparse
import datetime
import hashlib
import json
import math
import pathlib
import sys

from lb import freeze, prereg, train

ROOT = prereg.ROOT
CLASSES = ("untied", "tied")


class SelectionError(RuntimeError):
    """ERROR of an S3 script: the inputs are not exactly the expected, hash-consistent runs."""


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def read_results(runs_dir):
    """[(path, result dict)] for every result.json under runs_dir (sorted by path)."""
    runs_dir = pathlib.Path(runs_dir)
    if not runs_dir.is_dir():
        raise SelectionError(f"{runs_dir} is not a directory")
    out = []
    for path in sorted(runs_dir.rglob("result.json")):
        try:
            out.append((path, json.loads(path.read_text())))
        except (OSError, ValueError) as e:
            raise SelectionError(f"{path}: unreadable ({e})") from e
    return out


def harness_code(rec):
    """{file name: sha256} of the harness record, the form of result.json code_sha256."""
    out = {}
    for k, h in rec["files"].items():
        name = pathlib.Path(k).name
        if name in out:
            raise SelectionError(f"harness record lists two files named {name}")
        out[name] = h
    return out


def check_run(path, r, mode, harness, yaml_sha):
    """Mode, status, accuracy and hash checks shared by the S3 scripts."""
    if r.get("mode") != mode:
        raise SelectionError(f"{path}: mode {r.get('mode')!r} != {mode!r}")
    if r.get("status") not in ("ok", "diverged"):
        raise SelectionError(f"{path}: status {r.get('status')!r} not ok or diverged")
    acc = r.get("final_val_acc")
    if not isinstance(acc, (int, float)) or isinstance(acc, bool) or not math.isfinite(acc):
        raise SelectionError(f"{path}: final_val_acc {acc!r} is not a finite number")
    if r.get("prereg_sha256") != yaml_sha:
        raise SelectionError(f"{path}: prereg_sha256 {r.get('prereg_sha256')} != YAML in force {yaml_sha}")
    code = {k: v for k, v in (r.get("code_sha256") or {}).items() if not freeze._is_predictor(k)}
    if code != harness:
        diff = sorted(set(code.items()) ^ set(harness.items()))
        raise SelectionError(f"{path}: code_sha256 differs from the harness record ({diff[:4]})")


def inputs_sha(paths):
    return {freeze._key(p): sha256(p) for p in paths}


def write_x(path, obj):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write("\n")
    return sha256(path)


def expected_lrcal(p):
    """{(task, optimizer, class, lr): config_id} of the lrcal runs (tasks x optimizers x classes x lr_grid; v4: 120)."""
    out = {}
    for task in p["tasks"]:
        for opt in p["registry"]["optimizers"]:
            for cls in CLASSES:
                cid = train.reference_config_id(task, cls, opt, p)
                for lr in p["training"]["lr_grid"][opt]:
                    out[(task, opt, cls, float(lr))] = cid
    return out


def edge_flag(lr, grid):
    """Edge flag of a selected lr: first or last if it equals the first or last grid value, else None (reported only)."""
    grid = [float(x) for x in grid]
    if float(lr) == grid[0]:
        return "first"
    if float(lr) == grid[-1]:
        return "last"
    return None


def select(runs_dir):
    """Validate the lrcal runs and return the output object (nothing is written)."""
    p = prereg.load()
    yaml_sha = prereg.yaml_sha256()
    record_sha = freeze.verify("harness")
    harness = harness_code(freeze.load_record("harness"))
    expected = expected_lrcal(p)
    by_cid = {}
    for (task, opt, cls, lr), cid in expected.items():
        by_cid[cid] = (task, opt, cls)
    selection_seeds = list(p["seeds"]["selection"])
    found = {}
    for path, r in read_results(runs_dir):
        check_run(path, r, "lrcal", harness, yaml_sha)
        cid, lr, seed = r.get("config_id"), r.get("lr"), r.get("seed")
        if cid not in by_cid or seed not in selection_seeds or not isinstance(lr, (int, float)):
            raise SelectionError(f"{path}: unexpected run {cid!r} lr {lr!r} seed {seed!r}")
        task, opt, cls = by_cid[cid]
        key = (task, opt, cls, float(lr))
        if key not in expected:
            raise SelectionError(f"{path}: lr {lr!r} not in lr_grid[{opt}]")
        if key in found:
            raise SelectionError(f"{path}: duplicate of {found[key][0]} for {key}")
        found[key] = (path, r)
    missing = sorted(set(expected) - set(found))
    if missing:
        raise SelectionError(f"{len(missing)} of {len(expected)} lrcal runs missing, e.g. {missing[:3]}")
    if len(found) != len(expected):
        raise SelectionError(f"{len(found)} lrcal runs != {len(expected)}")
    rows = []
    for (task, opt, cls, lr), (path, r) in sorted(found.items()):
        rows.append({"task": task, "optimizer": opt, "class": cls, "config_id": r["config_id"], "lr": lr,
                     "seed": r["seed"], "status": r["status"], "final_val_acc": r["final_val_acc"],
                     "result": freeze._key(path)})
    selections, edges = {}, {}
    for task in p["tasks"]:
        for opt in p["registry"]["optimizers"]:
            for cls in CLASSES:
                cand = [x for x in rows if (x["task"], x["optimizer"], x["class"]) == (task, opt, cls)]
                best = min(cand, key=lambda x: (-x["final_val_acc"], x["lr"]))
                selections.setdefault(task, {}).setdefault(opt, {})[cls] = best["lr"]
                edges.setdefault(task, {}).setdefault(opt, {})[cls] = edge_flag(best["lr"], p["training"]["lr_grid"][opt])
    paths = [path for path, _ in found.values()] + [freeze.record_path("harness")]
    return {"kind": "lr_selection", "rule": p["training"]["lr_selection"],
            "reference_configs": p["training"]["reference_configs"], "selections": selections, "edge_flags": edges,
            "n_at_edge": sum(v is not None for o in edges.values() for c in o.values() for v in c.values()), "runs": rows,
            "n_runs": len(rows), "prereg_sha256": yaml_sha, "code_freeze": record_sha,
            "inputs_sha256": inputs_sha(sorted(paths)), "written_utc": utc_now()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="S3 learning-rate selection from the lrcal runs of the YAML in force")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        if pathlib.Path(a.out).exists():
            raise FileExistsError(f"{a.out} exists; refusing to overwrite")
        obj = select(a.runs)
        h = write_x(a.out, obj)
    except (SelectionError, freeze.CodeFreezeError, prereg.PreregMismatch, FileExistsError) as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps({"out": str(a.out), "sha256": h, "selections": obj["selections"], "edge_flags": obj["edge_flags"],
                      "n_at_edge": obj["n_at_edge"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
