"""Figure 1 (study design): scripted vector schematic (matplotlib patches and text; no generated image).
Every number shown is read from the frozen S4 files: tasks, registry sizes and class totals (n2_input.json,
selection.json), caps and control (selection.json), knob range (profile_registry_summary.json), selection seed
(selection.json result paths), confirmation and top-up seeds (judge_20261011T045319Z.json seed set vs the final
judge's seed set), floors and Holm family size (judge_final_20261011T054347Z.json).
v2 (S9 revision, 2026-10-14): label changes only. The training microbatch and checkpointing of the selection and
confirmation runs are read from the run result files that the frozen inputs pin by sha256 (selection.json
inputs_sha256 for the 744 selection runs; n1_input_topup.json, pinned by judge_final input_sha256.n1, for the 300
confirmation runs); every hash is verified. The pipeline boxes are 0.3 cm taller so the added lines fit."""
import hashlib
import json
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import common as C

NAME = "fig1_design_v2"
C.FROZEN["n1_input"] = C.S4 / "n1_input_topup.json"
TITLE = "Figure 1. Study design"
CAPTION = ("Figure 1. Study design. Two parameter classes form a registry of configurations per task. Each "
           "configuration's minimum-over-knobs peak of requested PyTorch CPU tensor bytes decides feasibility under "
           "seven caps and an unbounded control. Within each class the best feasible configuration on validation "
           "(seed 0) is selected per task and cap and confirmed on test (seeds 1\u201310, plus one top-up of seeds "
           "11\u201312); the frozen judge compares the difference with preregistered floors (A1). In parallel, a "
           "lifetime-aware accounting predicts each configuration's peak before any registry measurement (A2). "
           "Selection and confirmation runs are trained at microbatch 64 without checkpointing; feasibility uses "
           "each configuration's minimum-over-knobs peak. "
           "Drawn by a Python script (matplotlib 3.11.2) written with Anthropic's Claude Opus 5.5 (model identifier "
           "claude-opus-5-5).")
FORBIDDEN = ("first", "sram", "mcu", "on-device", "gpu")
TEXTS = []


def facts():
    n2 = C.load("n2_input")
    sel = C.load("selection")
    prof = C.load("profile_summary")
    jf = C.load("judge_final")
    j1 = C.load("judge_first")
    tasks = sorted({c["config_id"].split("|")[0] for c in n2["configs"]}, key=C.TASKS.index)
    per_task = {t: sum(1 for c in n2["configs"] if c["config_id"].startswith(t + "|")) for t in tasks}
    assert len(set(per_task.values())) == 1
    tot = {}
    for f in sel["feasibility"]:
        tot.setdefault(f["class"], set()).add(f["n_total"])
    assert all(len(v) == 1 for v in tot.values())
    mbs = sorted({m["microbatch"] for p in prof["profiles"].values() for m in p["measurements"]})
    ckpt = sorted({m["checkpointing"] for p in prof["profiles"].values() for m in p["measurements"]})
    sel_seeds = set()
    for cell in sel["cells"]:
        for k in ("tied", "untied"):
            if cell[k] is not None:
                sel_seeds.add(int(re.search(r"__s(\d+)/result\.json$", cell[k]["result"]).group(1)))
    first = list(j1["n1"]["seed_set"])
    final = list(jf["n1"]["seed_set"])
    topup = [s for s in final if s not in first]
    n1 = C.load("n1_input")
    if C.sha256(C.FROZEN["n1_input"]) != jf["input_sha256"]["n1"]:
        sys.exit(f"ASSERTION FAILED ({NAME}): n1_input_topup.json is not the file judge_final pins")
    (mb_s, ck_s), n_s = train_knobs(sel["inputs_sha256"], "select")
    (mb_c, ck_c), n_c = train_knobs(n1["_provenance"]["inputs_sha256"], "confirm")
    if (mb_s, ck_s) != (mb_c, ck_c) or ck_s is not False:
        sys.exit(f"ASSERTION FAILED ({NAME}): selection {mb_s, ck_s} vs confirmation {mb_c, ck_c}")
    return {
        "n_tasks": len(tasks), "tasks": tasks, "configs_per_task": per_task[tasks[0]],
        "configs_total": len(n2["configs"]), "tied_per_task": tot["tied"].pop(), "untied_per_task": tot["untied"].pop(),
        "caps_kib": sel["caps_kib"], "control_cap": sel["control_cap"], "microbatch_min": mbs[0],
        "microbatch_max": mbs[-1], "checkpointing_values": ckpt, "selection_seeds": sorted(sel_seeds),
        "confirmation_seeds": first, "topup_seeds": topup, "holm_family_size": jf["n1"]["holm_family_size"],
        "floors": jf["n1"]["floors"],
        "train_microbatch_select": mb_s, "train_checkpointing_select": ck_s, "n_select_runs": n_s,
        "train_microbatch_confirm": mb_c, "train_checkpointing_confirm": ck_c, "n_confirm_runs": n_c,
    }


def train_knobs(pins, mode):
    """(microbatch, checkpointing) set over the pinned result.json files of one run mode; hashes verified."""
    knobs = set()
    n = 0
    for rel, digest in pins.items():
        if not rel.endswith("/result.json"):
            continue
        raw = (C.ROOT / rel).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            sys.exit(f"ASSERTION FAILED ({NAME}): sha256 mismatch for {rel}")
        r = json.loads(raw)
        if r["mode"] != mode:
            sys.exit(f"ASSERTION FAILED ({NAME}): {rel} has mode {r['mode']}, expected {mode}")
        knobs.add((r["microbatch"], r["checkpointing"]))
        n += 1
    if len(knobs) != 1:
        sys.exit(f"ASSERTION FAILED ({NAME}): {mode} runs use several training knobs {sorted(knobs)}")
    return knobs.pop(), n


def rng(seq):
    return f"{seq[0]}–{seq[-1]}"


def text(ax, x, y, s, **kw):
    TEXTS.append(s)
    kw.setdefault("fontsize", 7)
    return ax.text(x, y, s, **kw)


def box(ax, x0, y0, x1, y1, title, body, ec="#222222", fc="white", ls="-", body_fs=7):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec=ec, lw=0.8, ls=ls))
    text(ax, x0 + 0.15, y1 - 0.18, title, ha="left", va="top", fontsize=7.5, fontweight="bold")
    text(ax, x0 + 0.15, y1 - 0.55, "\n".join(body), ha="left", va="top", fontsize=body_fs, linespacing=1.25)


def arrow(ax, p0, p1, **kw):
    kw.setdefault("color", "#222222")
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.8, shrinkA=0, shrinkB=0, **kw))


def small_block(ax, x0, y0, w, h, label, fc):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec="#222222",
                                lw=0.7))
    text(ax, x0 + w / 2, y0 + h / 2, label, ha="center", va="center")


def main():
    C.setup_style()
    F = facts()
    W, H, YB = 19.0, 11.0, 1.05  # drawing in cm; the view starts at y = YB
    fig = plt.figure(figsize=(W * C.CM, (H - YB) * C.CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(YB, H)
    ax.axis("off")
    tied_fc, untied_fc = "#cce0ef", "#f6d5c2"  # light tints of Okabe-Ito blue and vermillion

    # Registry panel
    ax.add_patch(FancyBboxPatch((0.2, 7.1), 18.6, 3.75, boxstyle="round,pad=0,rounding_size=0.2", fc="#f4f4f4",
                                ec="#888888", lw=0.8))
    tnames = ", ".join(C.TASK_NAME[t] for t in F["tasks"])
    text(ax, 0.45, 10.62, f"Registry: {F['n_tasks']} tasks ({tnames}) × {F['configs_per_task']} "
         f"configurations ({F['configs_total']} in total)", ha="left", va="top", fontsize=8, fontweight="bold")
    text(ax, 0.45, 10.15, "configuration = structure × width × depth T × optimizer", ha="left",
         va="top")
    ax.plot([9.5, 9.5], [7.35, 9.75], color="#aaaaaa", lw=0.6)
    # depth-tied
    text(ax, 0.45, 9.6, f"Depth-tied class: {F['tied_per_task']} per task", ha="left", va="top", fontsize=7.5,
         fontweight="bold", color=C.OI["blue"])
    arrow(ax, (0.6, 8.1), (1.3, 8.1))
    small_block(ax, 1.3, 7.7, 2.3, 0.8, "shared block", tied_fc)
    arrow(ax, (3.6, 8.1), (4.3, 8.1))
    ax.add_patch(FancyArrowPatch((3.3, 8.5), (1.6, 8.5), connectionstyle="arc3,rad=0.55", arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color="#222222", shrinkA=0, shrinkB=0))
    text(ax, 2.45, 9.03, "× T", ha="center", va="bottom", fontsize=7.5)
    text(ax, 4.75, 9.15, "one block applied T times\nVariants: shared,\ntwo groups,\nreversible shared",
         ha="left", va="top", linespacing=1.25)
    # distinct-parameter
    text(ax, 9.75, 9.6, f"Distinct-parameter class: {F['untied_per_task']} per task", ha="left", va="top",
         fontsize=7.5, fontweight="bold", color=C.OI["vermillion"])
    x = 9.8
    arrow(ax, (x, 8.1), (x + 0.4, 8.1))
    x += 0.4
    for i, lab in enumerate(["block 1", "block 2", None, "block T"]):
        if lab is None:
            text(ax, x + 0.25, 8.1, "…", ha="center", va="center", fontsize=8)
            x += 0.5
        else:
            small_block(ax, x, 7.75, 1.15, 0.7, lab, untied_fc)
            x += 1.15
        arrow(ax, (x, 8.1), (x + 0.35, 8.1))
        x += 0.35
    text(ax, x + 0.3, 9.15, "T blocks, each with\nits own parameters\nVariants: untied,\nreversible untied,\n"
         "one wide layer", ha="left", va="top", linespacing=1.25)

    # Pipeline row
    x0, gap, w = 1.0, 0.45, (17.8 - 4 * 0.45) / 5
    y0, y1 = 3.45, 6.45
    caps = F["caps_kib"]
    cap_l1 = ", ".join(str(c) for c in caps[:4]) + ","
    cap_l2 = ", ".join(str(c) for c in caps[4:]) + " KiB"
    fl = F["floors"]
    boxes = [
        ("Measured peak", ["peak requested PyTorch", "CPU tensor bytes,", "minimum over knobs:",
                           f"microbatch {F['microbatch_min']}–{F['microbatch_max']} with",
                           "accumulation; exact", "checkpointing", "(used for feasibility)"]),
        (f"{len(caps)} caps", [cap_l1, cap_l2, f"+ {F['control_cap']} control", "feasible: peak ≤ cap"]),
        ("Select on validation", [f"seed {', '.join(map(str, F['selection_seeds']))}; per task, cap",
                                  "and class: best feasible", "configuration",
                                  "trained at microbatch", f"{F['train_microbatch_select']}, no checkpointing"]),
        ("Confirm on test", [f"seeds {rng(F['confirmation_seeds'])}, plus one", f"top-up: seeds {rng(F['topup_seeds'])}",
                             "Δ = tied − distinct", "test accuracy",
                             "trained at microbatch", f"{F['train_microbatch_confirm']}, no checkpointing"]),
        ("Frozen judge → A1", ["Δ against floors", "(pp): " + ", ".join(f"{fl[t]:.2f}" for t in F["tasks"]),
                                    f"Holm over {F['holm_family_size']} cells"]),
    ]
    xs = []
    for i, (t, b) in enumerate(boxes):
        bx = x0 + i * (w + gap)
        xs.append(bx)
        box(ax, bx, y0, bx + w, y1, t, b)
        if i:
            arrow(ax, (bx - gap, (y0 + y1) / 2), (bx, (y0 + y1) / 2))
    arrow(ax, (xs[0] + w / 2, 7.1), (xs[0] + w / 2, y1))

    # Parallel accounting branch
    b0, b1 = 1.25, 2.85
    acc_x1 = xs[1] + w
    box(ax, xs[0], b0, acc_x1, b1, "Lifetime-aware accounting (in parallel)",
        ["replays the training update's allocations from tensor", "shapes; no fitted constants; each configuration's",
         "peak predicted before any registry measurement"], ec=C.OI["green"])
    a2x0, a2x1 = xs[2], xs[3] + w
    box(ax, a2x0, b0, a2x1, b1, "A2: prediction vs measurement",
        ["predicted vs measured minimum-over-knobs peak", "per configuration; fit at each cap"], ec=C.OI["green"])
    arrow(ax, (acc_x1, (b0 + b1) / 2), (a2x0, (b0 + b1) / 2))
    ax.plot([0.55, 0.55], [7.1, (b0 + b1) / 2], color="#222222", lw=0.8)
    arrow(ax, (0.55, (b0 + b1) / 2), (xs[0], (b0 + b1) / 2))
    mx = xs[0] + w / 2
    ym = (y0 + b1) / 2
    ax.plot([mx, mx, (a2x0 + a2x1) / 2], [y0, ym, ym], color="#222222", lw=0.8)
    arrow(ax, ((a2x0 + a2x1) / 2, ym), ((a2x0 + a2x1) / 2, b1))

    joined = " ".join(TEXTS).lower()
    for word in FORBIDDEN:
        assert word not in joined, f"forbidden word in figure text: {word}"
    C.save(fig, NAME)
    rows = [[k, ";".join(map(str, v)) if isinstance(v, list) else
             (";".join(f"{t}={v[t]!r}" for t in v) if isinstance(v, dict) else v)] for k, v in F.items()]
    C.write_csv(NAME, [CAPTION, "quantities shown in the schematic, read from the frozen sources"],
                ["quantity", "value"], rows, ["n2_input", "selection", "profile_summary", "judge_final",
                                              "judge_first", "n1_input"])
    C.write_pptx(NAME, TITLE, CAPTION)


if __name__ == "__main__":
    main()
