"""Figure 2 (descriptive): number of feasible configurations per class (depth-tied of 128, distinct-parameter of 120)
against cap, one panel per task. Reads only research/s4/selection.json (feasibility)."""
import matplotlib.pyplot as plt

import common as C

NAME = "fig2_feasible_counts"
TITLE = "Figure 2. Feasible configurations per class under each cap"
CAPTION = ("Figure 2. Number of registry configurations whose minimum-over-knobs measured peak fits each cap, per "
           "class (depth-tied: of 128 per task; distinct-parameter: of 120 per task). Descriptive; not judged.")
UNB_X = 8.0


def main():
    C.setup_style()
    s = C.load("selection")
    caps = s["caps_kib"]
    xpos = {cap: i for i, cap in enumerate(caps)}
    xpos["unbounded"] = UNB_X
    order = caps + ["unbounded"]
    feas = {(f["task"], f["class"], f["cap_kib"]): f for f in s["feasibility"]}
    assert len(feas) == len(s["feasibility"]) == 3 * 2 * 8
    totals = {"tied": 128, "untied": 120}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(19 * C.CM, 6.6 * C.CM), sharey=True)
    marker = {"tied": "o", "untied": "^"}
    for ax, task in zip(axes, C.TASKS):
        for cls in ("tied", "untied"):
            pts = [feas[(task, cls, cap)] for cap in order]
            assert all(p["n_total"] == totals[cls] for p in pts)
            xs = [xpos[cap] for cap in order]
            ys = [p["n_feasible"] for p in pts]
            mfc = C.CLASS_COLOR[cls] if cls == "tied" else "white"
            z = 3 if cls == "tied" else 4
            ax.plot(xs[:-1], ys[:-1], color=C.CLASS_COLOR[cls], marker=marker[cls], ms=3.8, mfc=mfc, lw=0.9, zorder=z)
            ax.plot(xs[-1], ys[-1], color=C.CLASS_COLOR[cls], marker=marker[cls], ms=3.8, mfc=mfc, ls="none",
                    zorder=z)
            for cap, x, p in zip(order, xs, pts):
                rows.append([task, cls, cap, x, p["n_feasible"], p["n_total"]])
        ax.axvline(7.0, color=C.OI["grey"], lw=0.5, ls=":")
        ax.set_xticks([xpos[c] for c in order])
        ax.set_xticklabels([str(c) for c in order], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, UNB_X + 0.6)
        ax.set_ylim(-4, 136)
        ax.set_title(C.TASK_NAME[task], fontsize=8)
        ax.set_xlabel("Cap (KiB)")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Feasible configurations")
    hs = [plt.Line2D([], [], color=C.CLASS_COLOR[k], marker=marker[k], ms=3.8, lw=0.9,
                     mfc=C.CLASS_COLOR[k] if k == "tied" else "white") for k in ("tied", "untied")]
    fig.legend(hs, [f"{C.CLASS_NAME[k]} (of {totals[k]})" for k in ("tied", "untied")], loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.27, wspace=0.12)
    C.save(fig, NAME)
    C.write_csv(NAME, [CAPTION, "x_position: plotting position (caps evenly spaced 0-6; unbounded set apart)"],
                ["task", "class", "cap_kib", "x_position", "n_feasible", "n_total"], rows, ["selection"])
    C.write_pptx(NAME, TITLE, CAPTION)


if __name__ == "__main__":
    main()
