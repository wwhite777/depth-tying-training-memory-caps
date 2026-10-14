"""Figure 4 v2 (A2, S9 revision 2026-10-14): (a) ratio of each static estimate (B1-B3) to the measured
minimum-over-knobs peak for the 744 registry configurations, by class, with class medians and the accounting's exact
reproduction (ratio 1) as a vertical line; (b) false fits (non-fitting boundary pairs admitted) per method, with the
false rejections (fitting boundary pairs rejected) stated in the panel.
Everything is recomputed from research/s4/n2_input.json with the frozen boundary rule (|P - 1024B| <= 0.2 x 1024B)
and checked against judge_final's N2.1 and N2.2 values; any mismatch exits non-zero."""
import statistics
import sys

import matplotlib.pyplot as plt

import common as C

NAME = "fig4_a2_accounting_v2"
TITLE = "Figure 4. A2: static estimates against measured peaks, and false fits at the caps"
CAPTION = ("Figure 4. (a) Ratio of each static estimate to the measured minimum-over-knobs peak of requested PyTorch "
           "CPU tensor bytes, for all 744 registry configurations by class (points jittered vertically; black "
           "ticks: class medians). B1: parameter count × state slots; B2: weights + gradients + optimizer "
           "state; B3: static four-component sum. The vertical line at 1 marks the lifetime-aware accounting, whose "
           "prediction equalled the measured peak for all 744 configurations. (b) Boundary pairs are "
           "configuration–cap pairs whose measured peak lies within 20% of the cap (928 pairs: 500 fit, 428 do "
           "not). Bars: the number of the 428 non-fitting pairs that each method classifies as fitting. No method "
           "classifies any of the 500 fitting pairs as not fitting.")
SHORT = {"param_count_with_state_multiplier": "B1", "persistent_weights_gradients_optimizer": "B2",
         "static_four_component_sum": "B3"}
PHI = 0.6180339887498949
JITTER = 0.6  # full width of the vertical jitter band, in row units
CLASSES = ("tied", "untied")


def need(cond, msg):
    if not cond:
        sys.exit(f"ASSERTION FAILED ({NAME}): {msg}")


def jitter(k):
    """Deterministic low-discrepancy offset for the k-th point of a row (k = 0, 1, ...)."""
    return JITTER * ((((k + 1) * PHI) % 1.0) - 0.5)


def row_y(bi, ci):
    return -(bi * 2.6 + ci * 1.0)


def boundary_counts(n2):
    """False fits (non-fitting pair classified as fitting) and false rejections per method."""
    methods = ["accounting", *C.BASELINES]
    ff = {k: 0 for k in methods}
    fr = {k: 0 for k in methods}
    n_pairs = n_fit = n_not = 0
    for c in n2["configs"]:
        meas = c["measured_min_peak_bytes"]
        est = {"accounting": c["predicted_min_peak_bytes"], **{b: c["baselines"][b] for b in C.BASELINES}}
        for cap in n2["caps_kib"]:
            cb = cap * 1024
            if abs(meas - cb) <= 0.2 * cb:
                n_pairs += 1
                fits = meas <= cb
                n_fit += fits
                n_not += not fits
                for k in methods:
                    says_fit = est[k] <= cb
                    ff[k] += (not fits) and says_fit
                    fr[k] += fits and not says_fit
    return n_pairs, n_fit, n_not, ff, fr


def main():
    C.setup_style()
    n2 = C.load("n2_input")
    j = C.load("judge_final")
    configs = n2["configs"]
    need(len(configs) == j["n2_1"]["n_configs"] == 744, "744 configurations")
    n_exact = sum(1 for c in configs if c["measured_min_peak_bytes"] == c["predicted_min_peak_bytes"])
    need(n_exact == 744 and j["n2_1"]["max_rel_err"] == 0.0 and j["n2_1"]["n_within_5pct"] == 744,
         "accounting exact on all 744 configurations (N2.1)")

    n_pairs, n_fit, n_not, ff, fr = boundary_counts(n2)
    need(n_pairs == j["n2_1"]["n_boundary_pairs"] == 928, f"n pairs {n_pairs} == 928")
    need(n_fit == 500 and n_not == 428, f"n fit {n_fit} == 500 and n not fit {n_not} == 428")
    need(ff["accounting"] == 0 and fr["accounting"] == 0 and j["n2_1"]["n_boundary_right"] == 928,
         "accounting right on all 928 boundary pairs (N2.1)")
    expected_b = {"param_count_with_state_multiplier": 424, "persistent_weights_gradients_optimizer": 424,
                  "static_four_component_sum": 422}
    comps = {cp["baseline"]: cp for cp in j["n2_2"]["comparisons"]}
    need(set(comps) == set(C.BASELINES), "N2.2 baselines")
    for b in C.BASELINES:
        # b: accounting right, baseline wrong; c: baseline right, accounting wrong (N2.2 discordant counts)
        acc_wrong_base_right = 0  # the accounting is right on every pair (asserted above)
        need(ff[b] + fr[b] == comps[b]["b"] == expected_b[b], f"{b}: discordant b {comps[b]['b']}")
        need(comps[b]["c"] == acc_wrong_base_right == 0, f"{b}: discordant c {comps[b]['c']}")
        need(fr[b] == 0 and ff[b] == expected_b[b], f"{b}: false fits {ff[b]}, false rejections {fr[b]}")

    fig = plt.figure(figsize=(19 * C.CM, 8.0 * C.CM))
    axa = fig.add_axes([0.185, 0.15, 0.35, 0.76])
    axb = fig.add_axes([0.745, 0.15, 0.215, 0.76])
    rows = []
    all_ratios = []
    ticks, ticklabels = [], []
    for bi, b in enumerate(C.BASELINES):
        for ci, cl in enumerate(CLASSES):
            y0 = row_y(bi, ci)
            sub = [c for c in configs if C.class_of(c["config_id"]) == cl]
            ratios = [c["baselines"][b] / c["measured_min_peak_bytes"] for c in sub]
            ys = [y0 + jitter(k) for k in range(len(sub))]
            all_ratios += ratios
            axa.scatter(ratios, ys, s=3, marker="o", facecolors="none", edgecolors=C.CLASS_COLOR[cl],
                        linewidths=0.35, alpha=0.8, zorder=3)
            med = statistics.median(ratios)
            axa.plot([med, med], [y0 - 0.42, y0 + 0.42], color="black", lw=1.4, zorder=4, solid_capstyle="butt")
            axa.text(med, y0 + 0.45, f"{med:.2f}", ha="center", va="bottom", fontsize=6.5, zorder=5)
            ticks.append(y0)
            ticklabels.append(f"{SHORT[b]}, {C.CLASS_NAME[cl]}")
            for c, r, y in zip(sub, ratios, ys):
                rows.append(["a", "point", b, cl, c["config_id"], c["baselines"][b], c["measured_min_peak_bytes"], r,
                             y, "", ""])
            rows.append(["a", "median", b, cl, "", "", "", med, y0, "", len(sub)])
            rows.append(["a", "min", b, cl, "", "", "", min(ratios), "", "", len(sub)])
            rows.append(["a", "max", b, cl, "", "", "", max(ratios), "", "", len(sub)])
    lo, hi = 0.25, 1.09
    need(lo < min(all_ratios) and max(all_ratios) < 1.0 < hi, f"x range covers {min(all_ratios)}..{max(all_ratios)}")
    axa.axvline(1.0, color=C.OI["green"], lw=1.0, zorder=2)
    ymin, ymax = row_y(2, 1) - 0.7, row_y(0, 0) + 0.95
    axa.text(1.012, (ymin + ymax) / 2, f"lifetime-aware accounting:\n{n_exact} of {len(configs)} peaks exact (ratio 1)",
             rotation=90, ha="left", va="center", fontsize=6.5, color="#006b4f", linespacing=1.15)
    rows.append(["a", "accounting_exact", "accounting", "", "", "", "", 1.0, "", n_exact, len(configs)])
    axa.set_xlim(lo, hi)
    axa.set_ylim(ymin, ymax)
    axa.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axa.set_yticks(ticks)
    axa.set_yticklabels(ticklabels)
    for lab, (bi, ci) in zip(axa.get_yticklabels(), [(bi, ci) for bi in range(3) for ci in range(2)]):
        lab.set_color(C.CLASS_COLOR[CLASSES[ci]])
    axa.set_xlabel("Static estimate / measured peak")
    axa.set_title("(a) Static estimates per configuration", loc="left", fontsize=8)

    order = [*C.BASELINES, "accounting"]
    names = {"param_count_with_state_multiplier": "B1: parameter count\n× state slots",
             "persistent_weights_gradients_optimizer": "B2: weights + gradients\n+ optimizer state",
             "static_four_component_sum": "B3: static four-\ncomponent sum",
             "accounting": "lifetime-aware\naccounting"}
    yb = list(range(len(order)))[::-1]
    for y, k in zip(yb, order):
        col = C.OI["green"] if k == "accounting" else C.OI["grey"]
        axb.barh(y, ff[k], height=0.6, color=col, zorder=2)
        axb.text(ff[k] + 8, y, f"{ff[k]} of {n_not}", va="center", ha="left", fontsize=7)
        rows.append(["b", "false_fit", k, "", "", "", "", "", "", ff[k], n_not])
    for k in order:
        rows.append(["b", "false_reject", k, "", "", "", "", "", "", fr[k], n_fit])
    axb.text(10, -1.05, f"Fitting pairs rejected: 0 of {n_fit}\nfor every method", ha="left", va="center",
             fontsize=6.5)
    need(all(fr[k] == 0 for k in order), "the in-panel note states zero false rejections")
    axb.set_yticks(yb)
    axb.set_yticklabels([names[k] for k in order], linespacing=1.1)
    axb.set_ylim(-1.6, len(order) - 0.45)
    axb.set_xlim(0, 640)
    axb.set_xticks([0, 100, 200, 300, 400])
    axb.set_xlabel(f"Non-fitting pairs admitted (of {n_not})")
    axb.set_title(f"(b) False fits, {n_pairs} boundary pairs", loc="left", fontsize=8)
    for ax in (axa, axb):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axa.text(0.262, row_y(2, 1), "black ticks:\nclass medians", ha="left", va="center", fontsize=6.5,
             linespacing=1.15)
    C.save(fig, NAME)
    C.write_csv(NAME, [CAPTION, "panel a point rows: method, class, config_id, estimate and measured bytes, ratio "
                       "(value), jittered y; median/min/max rows: ratio (value), n = configurations; "
                       "accounting_exact: count of exact predictions; panel b rows: count of false fits (n = "
                       "non-fitting boundary pairs) and false rejections (n = fitting boundary pairs)"],
                ["panel", "row", "method", "class", "config_id", "estimate_bytes", "measured_min_peak_bytes", "value",
                 "y_position", "count", "n"], rows, ["n2_input", "judge_final"])
    C.write_pptx(NAME, TITLE, CAPTION)
    print(f"{NAME}: pairs {n_pairs}, fit {n_fit}, not fit {n_not}; false fits {ff}; false rejections {fr}")


if __name__ == "__main__":
    main()
