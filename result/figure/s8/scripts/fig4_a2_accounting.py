"""Figure 4 (A2): (a) measured vs predicted min-over-knobs peak bytes for the 744 registry configurations;
(b) boundary-pair fit accuracy of the lifetime-aware accounting and the three graded baselines, recomputed here from
research/s4/n2_input.json with the frozen boundary rule and asserted equal to the judge's n2_1 / n2_2 values."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import common as C

NAME = "fig4_a2_accounting"
TITLE = "Figure 4. A2: lifetime-aware accounting of peak tensor bytes"
CAPTION = ("Figure 4. (a) Measured versus predicted minimum-over-knobs peak requested PyTorch CPU tensor bytes for "
           "all 744 registry configurations (log-log; the line is identity). Colour: task; marker: class. "
           "(b) Share of the 928 boundary pairs (configuration\u2013cap pairs whose measured peak lies within 20% of "
           "the cap) whose fit each method classifies correctly; numbers in brackets are counts.")


def boundary_accuracies(n2):
    caps = n2["caps_kib"]
    pairs = []
    for c in n2["configs"]:
        meas = c["measured_min_peak_bytes"]
        for cap in caps:
            cb = cap * 1024
            if abs(meas - cb) <= 0.2 * cb:
                truth = meas <= cb
                right = {"accounting": (c["predicted_min_peak_bytes"] <= cb) == truth}
                for b in C.BASELINES:
                    right[b] = (c["baselines"][b] <= cb) == truth
                pairs.append(right)
    out = {}
    for k in ["accounting", *C.BASELINES]:
        n_right = sum(1 for p in pairs if p[k])
        out[k] = (n_right, len(pairs), n_right / len(pairs))
    return out


def main():
    C.setup_style()
    n2 = C.load("n2_input")
    j = C.load("judge_final")
    configs = n2["configs"]
    assert len(configs) == j["n2_1"]["n_configs"] == 744
    acc = boundary_accuracies(n2)
    assert acc["accounting"][1] == j["n2_1"]["n_boundary_pairs"] == 928
    assert acc["accounting"][0] == j["n2_1"]["n_boundary_right"]
    assert acc["accounting"][2] == j["n2_1"]["boundary_fit_accuracy"]
    for comp in j["n2_2"]["comparisons"]:
        assert acc[comp["baseline"]][2] == comp["baseline_boundary_fit_accuracy"], comp["baseline"]
    n_exact = sum(1 for c in configs if c["measured_min_peak_bytes"] == c["predicted_min_peak_bytes"])

    fig = plt.figure(figsize=(19 * C.CM, 8.0 * C.CM))
    axa = fig.add_axes([0.075, 0.15, 0.33, 0.76])
    axb = fig.add_axes([0.665, 0.15, 0.30, 0.76])
    rows = []
    marker = {"tied": "o", "untied": "^"}
    for task in C.TASKS:
        for cls in ("tied", "untied"):
            sub = [c for c in configs if c["config_id"].split("|")[0] == task and C.class_of(c["config_id"]) == cls]
            axa.scatter([c["predicted_min_peak_bytes"] for c in sub], [c["measured_min_peak_bytes"] for c in sub],
                        s=9, marker=marker[cls], facecolors="none", edgecolors=C.TASK_COLOR[task], linewidths=0.6,
                        zorder=3)
            for c in sub:
                rows.append(["a", c["config_id"], task, cls, c["measured_min_peak_bytes"],
                             c["predicted_min_peak_bytes"], "", "", ""])
    vals = [c["measured_min_peak_bytes"] for c in configs] + [c["predicted_min_peak_bytes"] for c in configs]
    lo, hi = min(vals) / 1.3, max(vals) * 1.3
    axa.plot([lo, hi], [lo, hi], color=C.OI["grey"], lw=0.7, zorder=2)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(lo, hi)
    axa.set_ylim(lo, hi)
    axa.set_aspect("equal")
    axa.set_xlabel("Predicted peak (bytes)")
    axa.set_ylabel("Measured peak (bytes)")
    axa.text(0.97, 0.04, f"{n_exact}/{len(configs)} exact", transform=axa.transAxes, ha="right", va="bottom",
             fontsize=7.5)
    h = [Line2D([], [], marker="s", ls="none", mfc=C.TASK_COLOR[t], mec=C.TASK_COLOR[t], ms=5) for t in C.TASKS]
    h += [Line2D([], [], marker=marker[k], ls="none", mfc="none", mec="#444444", ms=4.5) for k in ("tied", "untied")]
    lab = [C.TASK_NAME[t] for t in C.TASKS] + [C.CLASS_NAME[k] for k in ("tied", "untied")]
    axa.legend(h, lab, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.3, labelspacing=0.35)
    axa.set_title("(a) Peak per configuration", loc="left", fontsize=8)

    order = ["accounting", *C.BASELINES]
    names = {"accounting": "lifetime-aware accounting", **C.BASELINE_NAME}
    ys = list(range(len(order)))[::-1]
    for y, k in zip(ys, order):
        n_right, n_pairs, a = acc[k]
        col = C.OI["blue"] if k == "accounting" else C.OI["grey"]
        axb.barh(y, a, height=0.6, color=col, zorder=2)
        axb.text(a + 0.02, y, f"{a:.3f} ({n_right}/{n_pairs})", va="center", ha="left", fontsize=7)
        rows.append(["b", k, "", "", "", "", n_right, n_pairs, repr(a)])
    axb.set_yticks(ys)
    axb.set_yticklabels([names[k] for k in order])
    axb.set_xlim(0, 1.42)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xlabel(f"Boundary-pair fit accuracy ({acc['accounting'][1]} pairs)")
    axb.set_title("(b) Fit at the caps, boundary pairs", loc="left", fontsize=8)
    for ax in (axa, axb):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    C.save(fig, NAME)
    C.write_csv(NAME, [CAPTION, "panel a rows: config_id, task, class, measured and predicted bytes; "
                       "panel b rows: method, n_right, n_pairs, fit_accuracy"],
                ["panel", "item", "task", "class", "measured_min_peak_bytes", "predicted_min_peak_bytes", "n_right",
                 "n_pairs", "fit_accuracy"], rows, ["n2_input", "judge_final"])
    C.write_pptx(NAME, TITLE, CAPTION)


if __name__ == "__main__":
    main()
