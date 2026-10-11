"""Figure 3 (A1 per cell): mean Delta (tied minus distinct test accuracy, pp, seeds 1-12) per task and cap, with the
one-sided 95 percent upper bound, +/- floor and the frozen judge's E_minus (E−) / zero_sign events.
Reads only research/s4/judge_final_20261011T054347Z.json."""
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

import common as C

NAME = "fig3_a1_cells"
TITLE = "Figure 3. A1: depth-tied minus distinct-parameter test accuracy per cell"
CAPTION = ("Figure 3. Difference in test accuracy between the selected depth-tied and distinct-parameter "
           "configurations (\u0394, percentage points; mean over seeds 1\u201312) in each task and cap. Bars run from "
           "the mean to the one-sided 95% upper bound. Solid line: +floor; dashed line: \u2212floor; thin line: zero. "
           "Triangle: E\u2212 (\u0394 below \u2212floor, Holm-adjusted p < 0.05); open square: \u0394 \u2260 0 "
           "(two-sided, Holm-adjusted p < 0.05). No cell shows E+. Orange: unbounded control cells, outside the Holm "
           "family. Adjacent caps with the same selected pair show the same value.")
UNB_X = 8.0
SEP_X = 7.0


class UpperBoundHandle:
    """Placeholder legend handle for the one-sided bound glyph."""


class HandlerUpper(HandlerBase):
    """Legend glyph: an error-bar handle, a vertical line spanning more than the text height with a cap at its top
    (one-sided bound), drawn like the bars in the panels."""

    def create_artists(self, legend, orig, xd, yd, w, h, fontsize, trans):
        x = xd + w / 2
        y0, y1 = yd - 0.45 * h, yd + 1.45 * h
        bar = Line2D([x, x], [y0, y1], color="k", lw=0.9, transform=trans)
        cap = Line2D([x - 0.22 * w, x + 0.22 * w], [y1, y1], color="k", lw=0.9, transform=trans)
        return [bar, cap]


def rows_from_judge(j):
    caps = None
    rows = []
    for c in j["n1"]["cells"]:
        rows.append(c)
    caps = sorted({c["cap_kib"] for c in rows if c["finite_cap"]})
    return caps, rows


def main():
    C.setup_style()
    j = C.load("judge_final")
    caps, cells = rows_from_judge(j)
    assert len(caps) == 7 and j["n1"]["holm_family_size"] == 21
    xpos = {cap: i for i, cap in enumerate(caps)}
    xpos["unbounded"] = UNB_X
    assert not any(c["E_plus"] for c in cells), "an E_plus cell exists; the figure assumes none"
    for c in cells:
        assert c["in_holm_family"] == c["finite_cap"]
        if not c["in_holm_family"]:
            assert not (c["E_plus"] or c["E_minus"] or c["zero_sign"])
        assert c["feasible"] and c["mean"] is not None and c["upper_bound_95"] is not None

    csv_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(19 * C.CM, 8.8 * C.CM))
    for ax, task in zip(axes, C.TASKS):
        tc = [c for c in cells if c["task"] == task]
        floor = j["n1"]["floors"][task]
        assert all(abs(c["floor"] - floor) == 0 for c in tc)
        ax.axhline(0, color=C.OI["grey"], lw=0.5, zorder=1)
        ax.axhline(floor, color="#555555", lw=0.8, ls="-", zorder=1)
        ax.axhline(-floor, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.axvline(SEP_X, color=C.OI["grey"], lw=0.5, ls=":", zorder=1)
        for c in tc:
            x = xpos[c["cap_kib"]]
            m, ub = c["mean"], c["upper_bound_95"]
            col = C.OI["black"] if c["in_holm_family"] else C.OI["orange"]
            ax.plot([x, x], [m, ub], color=col, lw=0.9, zorder=2)
            ax.plot([x - 0.22, x + 0.22], [ub, ub], color=col, lw=0.9, zorder=2)
            mk = "v" if c["E_minus"] else "o"
            ax.plot(x, m, marker=mk, ms=4.2, color=col, mec=col, ls="none", zorder=3)
            if c["zero_sign"]:
                ax.plot(x, m, marker="s", ms=8.5, mfc="none", mec=C.OI["black"], mew=0.8, ls="none", zorder=3)
            csv_rows.append([task, c["cap_kib"], x, repr(m), repr(ub), repr(floor), c["E_plus"], c["E_minus"],
                             c["zero_sign"], c["in_holm_family"]])
        ax.set_xticks(list(range(len(caps))) + [UNB_X])
        ax.set_xticklabels([str(cp) for cp in caps] + ["unbounded"], rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_xlim(-0.6, UNB_X + 1.5)
        lo = min(min(c["mean"] for c in tc), -floor)
        hi = max(max(c["upper_bound_95"] for c in tc), floor)
        pad = 0.18 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        xr = ax.get_xlim()[1] - 0.05
        ax.text(xr, floor, "floor", ha="right", va="bottom", fontsize=7, color="#333333")
        ax.text(xr, -floor, "\u2212floor", ha="right", va="top", fontsize=7, color="#333333")
        ax.text(SEP_X + 0.32, 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1]), "outside the Holm family",
                rotation=90, ha="center", va="center", fontsize=7, color="#7a5200", zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        ax.set_title(f"{C.TASK_NAME[task]} (floor {floor:.2f} pp)", fontsize=8)
        ax.set_xlabel("Cap (KiB)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Δ test accuracy, tied − distinct (pp)")

    handles = [
        Line2D([], [], marker="o", color="k", ls="none", ms=4.2),
        Line2D([], [], marker="v", color="k", ls="none", ms=4.2),
        Line2D([], [], marker="s", mfc="none", mec="k", ls="none", ms=8.5, mew=0.8),
        Line2D([], [], marker="o", color=C.OI["orange"], ls="none", ms=4.2),
        UpperBoundHandle(),
        Line2D([], [], color="#555555", lw=0.8),
        Line2D([], [], color="#555555", lw=0.8, ls=(0, (4, 2))),
    ]
    labels = ["mean Δ, capped cell",
              "E−: Δ < −floor (Holm-adjusted p < 0.05)",
              "Δ ≠ 0 (two-sided, Holm-adjusted p < 0.05)",
              "mean Δ, unbounded control",
              "one-sided 95% upper bound",
              "+floor",
              "−floor"]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, handler_map={UpperBoundHandle: HandlerUpper()},
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.92, bottom=0.36, wspace=0.22)
    C.save(fig, NAME)
    C.write_csv(NAME, [CAPTION, "x_position: plotting position (caps evenly spaced 0-6; unbounded set apart)"],
                ["task", "cap_kib", "x_position", "mean_delta_pp", "upper_bound_95_pp", "floor_pp", "E_plus",
                 "E_minus", "zero_sign", "in_holm_family"], csv_rows, ["judge_final"])
    C.write_pptx(NAME, TITLE, CAPTION)


if __name__ == "__main__":
    main()
