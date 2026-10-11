"""Table 1 of the S8 manuscript (all 24 cells after the top-up), generated from the frozen judge and selection files.

  ~/envs/jeongwoncheol_loopbudget/bin/python research/s8/make_table1.py --out-dir research/s8/tables
Writes table1_<UTC>.md (the rows pasted into the manuscript) and table1_<UTC>.csv (exact values). Exit 2 on a hash
mismatch or an existing output. Display rounding: 2 decimals for percentage points, 1 decimal for KiB, half away from zero after
removing floating-point noise (values are first rounded to 9 decimals), so exact decimal ties such as -0.385 and
-0.775 (12-seed means of accuracies in steps of 0.01 points) round the same way.
"""
import argparse
import csv
import datetime
import decimal
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
JUDGE = ("research/s4/judge_final_20261011T054347Z.json", "e0bc089578fbff704af37098269b2caa334bc12ff65094c7eacec242476e35b1")
SELECTION = ("research/s4/selection.json", "c50acf65dbe9e1b2f4416903c5f50fb71b99126803f8dc6d7b7f7cec13402904")
TASK = {"HAR": "UCI-HAR", "KWS": "Speech Commands", "CIFAR10": "CIFAR-10"}
STRUCT = {"shared": "shared", "two_groups": "two groups", "reversible_shared": "rev. shared", "untied": "untied",
          "reversible_untied": "rev. untied", "wide_single": "wide"}
OPT = {"adafactor": "", "adam": ", Adam", "sgd": ", SGD", "sgd_momentum": ", SGD-m"}


def r(x, nd, sign=False):
    """Round half away from zero after removing binary noise; returns a string."""
    q = decimal.Decimal(1).scaleb(-nd)
    d = decimal.Decimal(repr(round(x, 9))).quantize(q, rounding=decimal.ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    t = f"{d:.{nd}f}".replace("-", "\u2212")          # typographic minus sign in the displayed table
    return ("+" + t if d > 0 else t) if sign else t


def load(rel, want):
    p = ROOT / rel
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != want:
        raise SystemExit(f"ERROR: hash mismatch {rel}")
    return json.loads(p.read_text())


def name(x):
    s = STRUCT[x["structure"]]
    t = "" if x["structure"] == "wide_single" else f", T {x['T']}"
    return f"{s}, d {x['d']}{t}{OPT[x['optimizer']]} ({r(x['min_peak_bytes'] / 1024, 1)})"


def events(c):
    if not c["in_holm_family"]:
        return "control"
    ev = [lab for key, lab in (("E_plus", "E+"), ("E_minus", "E−"), ("zero_sign", "Δ≠0")) if c[key]]
    return "; ".join(ev) if ev else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    j, s = load(*JUDGE), load(*SELECTION)
    sel = {(c["task"], c["cap_kib"]): c for c in s["cells"]}
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, cs = out / f"table1_{ts}.md", out / f"table1_{ts}.csv"
    if md.exists() or cs.exists():
        raise SystemExit("ERROR: output exists")
    rows, lines = [], []
    lines.append("| Task | Cap (KiB) | Depth-tied selection (peak, KiB) | Distinct-parameter selection (peak, KiB) "
                 "| Δ (pp) | SD (pp) | Upper 95% bound (pp) | Floor (pp) | Events |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in j["n1"]["cells"]:
        k = (c["task"], c["cap_kib"])
        t, u = sel[k]["tied"], sel[k]["untied"]
        if t["config_id"] != c["tied_config"] or u["config_id"] != c["untied_config"]:
            raise SystemExit(f"ERROR: selection and judge disagree at {k}")
        cap = "unbounded" if c["cap_kib"] == "unbounded" else str(c["cap_kib"])
        lines.append(f"| {TASK[c['task']]} | {cap} | {name(t)} | {name(u)} | {r(c['mean'], 2, True)} | "
                     f"{r(c['sd'], 2)} | {r(c['upper_bound_95'], 2, True)} | {r(c['floor'], 2)} | {events(c)} |")
        rows.append({"task": c["task"], "cap_kib": c["cap_kib"], "tied_config": c["tied_config"],
                     "tied_min_peak_bytes": t["min_peak_bytes"], "untied_config": c["untied_config"],
                     "untied_min_peak_bytes": u["min_peak_bytes"], "n": c["n"], "mean_pp": repr(c["mean"]),
                     "sd_pp": repr(c["sd"]), "upper_bound_95_pp": repr(c["upper_bound_95"]), "floor_pp": repr(c["floor"]),
                     "in_holm_family": c["in_holm_family"], "E_plus": c["E_plus"], "E_minus": c["E_minus"],
                     "zero_sign": c["zero_sign"], "p_minus_holm": c["p_minus_holm"], "p_zero_holm": c["p_zero_holm"]})
    md.write_text("\n".join(lines) + "\n")
    with cs.open("w", newline="") as f:
        f.write(f"# sources: {JUDGE[0]} {JUDGE[1]}; {SELECTION[0]} {SELECTION[1]}\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"written {md} and {cs} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
