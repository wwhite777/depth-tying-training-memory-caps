"""Table 2 of the S8 manuscript: where the cap goes in the selected configurations (persistent state L0 = W + G + O
against the measured minimum peak), one row per distinct (depth-tied, distinct-parameter) pair at the finite caps
plus one row per unbounded cell, generated from the frozen selection, profile-summary and confirmation records.

  OMP_NUM_THREADS=1 ~/envs/jeongwoncheol_loopbudget_docs/bin/python research/s8/make_table2.py \
      --out-dir research/s8/tables --ts <UTC> > research/s8/tables/table2_<UTC>.log 2>&1
Writes table2_<UTC>.md (rows for the manuscript) and table2_<UTC>.csv (exact bytes, argmin knobs, the peak at microbatch
64 without checkpointing and its ratio to 1024 x the smallest cap of the row). Display rounding as in make_table1
(half away from zero): KiB to 1 decimal, share as an integer percent. Before writing, the configurations and peaks of every
cell are checked against Table 1's CSV (--table1); exit 2 on any mismatch, a hash mismatch of a frozen input, or an
existing output (nothing is written).
"""
import argparse
import collections
import csv
import glob
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from make_table1 import JUDGE, OPT, ROOT, SELECTION, STRUCT, TASK, load, r  # noqa: E402

SUMMARY = ("research/s4/profile_registry_summary.json", "429bc046221d1b33b2aeda2c4da96f63d8312dedaf41a95882441b7e117cb94c")
TABLE1 = "research/s8/tables/table1_20261011T095730Z.csv"
TASKS = ("HAR", "KWS", "CIFAR10")
SIDES = (("tied", "Depth-tied"), ("untied", "Distinct-parameter"))


def label(x):
    """Selection label exactly as Table 1 writes it, without the peak in parentheses."""
    t = "" if x["structure"] == "wide_single" else f", T {x['T']}"
    return f"{STRUCT[x['structure']]}, d {x['d']}{t}{OPT[x['optimizer']]}"


def argmin_meas(p, cid):
    ak = p["argmin_knob"]
    hit = [m for m in p["measurements"] if m["microbatch"] == ak["microbatch"] and m["checkpointing"] == ak["checkpointing"]]
    if len(hit) != 1 or hit[0]["peak_bytes"] != p["min_peak_bytes"]:
        raise ValueError(f"argmin measurement inconsistent for {cid}")
    m = hit[0]
    if m["baseline_bytes"] != m["W_bytes"] + m["G_bytes"] + m["O_bytes"]:
        raise ValueError(f"baseline_bytes != W + G + O for {cid}")
    return m


def peak64(p, cid):
    hit = [m["peak_bytes"] for m in p["measurements"] if m["microbatch"] == 64 and not m["checkpointing"]]
    if len(hit) != 1:
        raise ValueError(f"no unique microbatch-64 no-checkpointing measurement for {cid}")
    return hit[0]


def check_table1(path, sel, reg):
    """Mismatches between Table 1's CSV and the selection / registry (configurations, peak bytes, peak KiB)."""
    bad = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    seen = set()
    for row in rows:
        cap = row["cap_kib"] if row["cap_kib"] == "unbounded" else int(row["cap_kib"])
        k = (row["task"], cap)
        seen.add(k)
        if k not in sel:
            bad.append(f"{k}: cell not in selection.json")
            continue
        for cl in ("tied", "untied"):
            x = sel[k][cl]
            if row[f"{cl}_config"] != x["config_id"]:
                bad.append(f"{k} {cl}: config {row[cl + '_config']} != {x['config_id']}")
                continue
            t1 = int(row[f"{cl}_min_peak_bytes"])
            rp = reg[x["config_id"]]["min_peak_bytes"]
            if t1 != x["min_peak_bytes"] or t1 != rp:
                bad.append(f"{k} {cl}: peak bytes table1 {t1}, selection {x['min_peak_bytes']}, registry {rp}")
            if r(t1 / 1024, 1) != r(rp / 1024, 1):
                bad.append(f"{k} {cl}: peak KiB table1 {r(t1 / 1024, 1)} != registry {r(rp / 1024, 1)}")
    if seen != set(sel):
        bad.append(f"cell sets differ: only in table1 {sorted(map(str, seen - set(sel)))}, "
                   f"only in selection {sorted(map(str, set(sel) - seen))}")
    return len(rows), bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ts", required=True, help="UTC stamp YYYYMMDDTHHMMSSZ taken from the clock by the caller")
    ap.add_argument("--table1", default=TABLE1, help="Table 1 CSV to check against (default: the manuscript's)")
    a = ap.parse_args()
    j, s, summ = load(*JUDGE), load(*SELECTION), load(*SUMMARY)
    reg = summ["profiles"]
    sel = {(c["task"], c["cap_kib"]): c for c in s["cells"]}
    for c in j["n1"]["cells"]:
        k = (c["task"], c["cap_kib"])
        if sel[k]["tied"]["config_id"] != c["tied_config"] or sel[k]["untied"]["config_id"] != c["untied_config"]:
            print(f"ERROR: selection and judge disagree at {k}", file=sys.stderr)
            return 2

    t1path = pathlib.Path(a.table1)
    t1path = t1path if t1path.is_absolute() else ROOT / t1path
    t1sha = hashlib.sha256(t1path.read_bytes()).hexdigest()
    n_t1, bad = check_table1(t1path, sel, reg)
    print(f"table1 {a.table1} sha256 {t1sha}: {n_t1} rows checked, {len(bad)} mismatches")
    if bad:
        for b in bad:
            print(f"ERROR: table1 mismatch: {b}", file=sys.stderr)
        return 2

    npar = collections.defaultdict(set)
    files = sorted(glob.glob(str(ROOT / "result/s4/confirm/*/result.json")))
    for f in files:
        rec = json.loads(pathlib.Path(f).read_text())
        npar[rec["config_id"]].add(rec["n_params"])
    if len(files) != 300 or any(len(v) != 1 for v in npar.values()):
        print(f"ERROR: confirm records: {len(files)} files or n_params not unique per configuration", file=sys.stderr)
        return 2
    npar = {k: next(iter(v)) for k, v in npar.items()}
    for c in s["cells"]:
        for cl in ("tied", "untied"):
            if c[cl]["n_params"] != npar[c[cl]["config_id"]]:
                print(f"ERROR: n_params of {c[cl]['config_id']} differs from the confirm records", file=sys.stderr)
                return 2

    # group adjacent finite caps that share a pair; a pair may not reappear after another pair
    groups = []
    for t in TASKS:
        caps = [c for c in s["caps_kib"] if (t, c) in sel]
        done = set()
        for cap in caps:
            pr = (sel[(t, cap)]["tied"]["config_id"], sel[(t, cap)]["untied"]["config_id"])
            if groups and groups[-1][0] == t and groups[-1][1] == pr and groups[-1][2][-1] != "unbounded":
                groups[-1][2].append(cap)
                continue
            if pr in done:
                print(f"ERROR: pair {pr} recurs non-adjacently in {t}", file=sys.stderr)
                return 2
            done.add(pr)
            groups.append((t, pr, [cap]))
        groups.append((t, None, ["unbounded"]))
    n_fin = sum(1 for g in groups if g[2] != ["unbounded"])
    if n_fin != 14 or len(groups) != 17:
        print(f"ERROR: expected 14 finite pair rows and 3 unbounded rows, got {n_fin} and {len(groups) - n_fin}",
              file=sys.stderr)
        return 2

    out = pathlib.Path(a.out_dir)
    md, cs = out / f"table2_{a.ts}.md", out / f"table2_{a.ts}.csv"
    if md.exists() or cs.exists():
        print("ERROR: output exists", file=sys.stderr)
        return 2
    lines = ["| Task | Caps (KiB) | Depth-tied selection | Params | Persistent / peak (KiB) | Share (%) "
             "| Distinct-parameter selection | Params | Persistent / peak (KiB) | Share (%) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    try:
        for t, _, caps in groups:
            c0 = sel[(t, caps[0])]
            cap_lab = "unbounded" if caps == ["unbounded"] else (str(caps[0]) if len(caps) == 1
                                                                  else f"{caps[0]}–{caps[-1]}")
            md_cells = [TASK[t], cap_lab]
            row = {"task": t, "caps_kib": cap_lab, "cells": " ".join(str(c) for c in caps),
                   "smallest_cap_kib": "" if caps == ["unbounded"] else caps[0]}
            for cl, _ in SIDES:
                x = c0[cl]
                cid = x["config_id"]
                p = reg[cid]
                m = argmin_meas(p, cid)
                base, peak = m["baseline_bytes"], p["min_peak_bytes"]
                if peak != x["min_peak_bytes"]:
                    raise ValueError(f"selection and registry peaks differ for {cid}")
                p64 = peak64(p, cid)
                lab = label(x)
                pk, bk, sh = r(peak / 1024, 1), r(base / 1024, 1), r(100 * base / peak, 0)
                md_cells += [lab, f"{npar[cid]:,}", f"{bk} / {pk}", sh]
                row.update({f"{cl}_selection": lab, f"{cl}_params": npar[cid], f"{cl}_persistent_kib": bk,
                            f"{cl}_peak_kib": pk, f"{cl}_share_pct": sh, f"{cl}_config": cid,
                            f"{cl}_baseline_bytes": base, f"{cl}_min_peak_bytes": peak,
                            f"{cl}_share_exact": repr(base / peak),
                            f"{cl}_argmin_microbatch": p["argmin_knob"]["microbatch"],
                            f"{cl}_argmin_checkpointing": p["argmin_knob"]["checkpointing"],
                            f"{cl}_peak_mb64_nockpt_bytes": p64,
                            f"{cl}_peak_mb64_nockpt_over_smallest_cap": ("" if caps == ["unbounded"]
                                                                         else repr(p64 / (1024 * caps[0])))})
            lines.append("| " + " | ".join(md_cells) + " |")
            rows.append(row)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n")
    with cs.open("w", newline="") as f:
        f.write(f"# sources: {SELECTION[0]} {SELECTION[1]}; {JUDGE[0]} {JUDGE[1]}; {SUMMARY[0]} {SUMMARY[1]}; "
                f"result/s4/confirm/*/result.json (n_params); checked against {a.table1} {t1sha}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written {md} and {cs} ({len(rows)} rows: {n_fin} finite pairs, {len(rows) - n_fin} unbounded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
