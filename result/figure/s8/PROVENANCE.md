# Provenance of Figures 1–4

The figure scripts were written with Claude Code (Anthropic; model claude-opus-5-5) on 2026-10-11 and checked by the authors. No AI image generator was used. Figure 1 is drawn by a Python script with matplotlib patches and text; Figures 2–4 are data plots.

## Figures and generating scripts (`result/figure/s8/scripts/`)
| figure | files | script | data sources read |
|---|---|---|---|
| Figure 1, study design (schematic, 19 x 9.95 cm) | fig1_design.{pdf,png,csv} | fig1_design.py | n2_input.json, selection.json, profile_registry_summary.json, judge_final_20261011T054347Z.json, judge_20261011T045319Z.json (seed set only, to separate seeds 1–10 from the top-up; nothing from it is plotted) |
| Figure 2, feasible counts (19 x 6.6 cm) | fig2_feasible_counts.{pdf,png,csv} | fig2_feasible_counts.py | selection.json |
| Figure 3, accuracy difference per cell (19 x 8.8 cm) | fig3_a1_cells.{pdf,png,csv} | fig3_a1_cells.py | judge_final_20261011T054347Z.json |
| Figure 4, accounting versus measurement (19 x 8.0 cm) | fig4_a2_accounting.{pdf,png,csv} | fig4_a2_accounting.py | n2_input.json; judge_final_20261011T054347Z.json (assertion only) |

The figures are numbered in their order of first citation in the paper.

Shared helpers are in `scripts/common.py` (style, paths, CSV and a one-slide PPTX per figure). All data sources are in `research/s4/`. Each CSV header records the sha256 of the sources it used. Every plotted number is read from these files at run time; no value is typed by hand.

## Data sources (frozen; read only)
- research/s4/judge_final_20261011T054347Z.json — sha256 e0bc089578fbff704af37098269b2caa334bc12ff65094c7eacec242476e35b1
- research/s4/judge_20261011T045319Z.json — sha256 ed6375cf8b3f837dbe0886ad345fef847b66f31803fe0199aa64b173263d6d45
- research/s4/selection.json — sha256 c50acf65dbe9e1b2f4416903c5f50fb71b99126803f8dc6d7b7f7cec13402904
- research/s4/n2_input.json — sha256 f240c26497a8a79a4f14cd565a9046ea897655cb020e938d2612b4e6b9fcc515
- research/s4/profile_registry_summary.json — sha256 429bc046221d1b33b2aeda2c4da96f63d8312dedaf41a95882441b7e117cb94c

## Software
- Python 3.12.12 with the packages of `requirements-figures.txt`: matplotlib 3.11.2, numpy 2.5.4, python-pptx 1.0.2, Pillow 12.3.0, fonttools 4.66.1, and PyMuPDF 1.28.2 for the checker.
- Font: DejaVu Sans as bundled with matplotlib (TrueType), embedded as Type 42 (pdf.fonttype 42); the PDF CreationDate is omitted.
- Colours: Okabe-Ito palette.

## Reproduce
From the repository root, with the figure environment of the README (`.venv-fig`):
```
.venv-fig/bin/python result/figure/s8/scripts/fig1_design.py
.venv-fig/bin/python result/figure/s8/scripts/fig2_feasible_counts.py
.venv-fig/bin/python result/figure/s8/scripts/fig3_a1_cells.py
.venv-fig/bin/python result/figure/s8/scripts/fig4_a2_accounting.py
.venv-fig/bin/python result/figure/s8/check_figures.py
```
check_figures.py recomputes every CSV value from the frozen sources. It also checks the source hashes, the embedded TrueType fonts of each PDF, the PNG size and dpi, and the PPTX slide and notes. It writes `logs/check_figures_<UTC>.log`.

## Revision of 2026-10-14
The revised scripts were written with Claude Code (Anthropic; model claude-opus-5-5) on 2026-10-14 and checked by the authors. No AI image generator was used. The earlier figures, scripts and outputs above are unchanged; the new files sit next to them.

| figure | files | script | data sources read |
|---|---|---|---|
| Figure 4 v2, static estimates vs measured peak and false fits (19 x 8.0 cm) | fig4_a2_accounting_v2.{pdf,png,csv} | fig4_a2_accounting_v2.py | n2_input.json; judge_final_20261011T054347Z.json (assertions against N2.1 and the N2.2 discordant counts) |
| Figure 1 v2, study design with the training microbatch of the accuracy runs (19 x 9.95 cm) | fig1_design_v2.{pdf,png,csv} | fig1_design_v2.py (a copy of fig1_design.py with label changes) | the five sources of fig1_design.py, plus research/s4/n1_input_topup.json (sha256 a329cdc8e657f747c6380d06b786e524953dbddc0d0fbc601c87577bdb7ea0a6, pinned by judge_final input_sha256.n1) and the run files result/s4/select/*/result.json (744, pinned by selection.json inputs_sha256) and result/s4/confirm/*/result.json (300, pinned by n1_input_topup.json _provenance.inputs_sha256); every pinned hash is verified |

Figure 4 v2: panel (a) plots, for B1 (parameter count x state slots), B2 (weights + gradients + optimizer state) and B3 (static four-component sum), the ratio of the estimate to the measured minimum-over-knobs peak for all 744 configurations by class, with class medians and a vertical line at 1 for the lifetime-aware accounting (744 of 744 exact). The vertical jitter is deterministic: offset = 0.6 x (frac((k + 1) x 0.6180339887498949) - 0.5) for the k-th configuration of a row. Panel (b) counts, on the boundary pairs (|P - 1024B| <= 0.2 x 1024B; 928 pairs, 500 fit, 428 do not), the non-fitting pairs each method admits (B1 424, B2 424, B3 422, accounting 0) and states that no method rejects any fitting pair (0 of 500). The script exits non-zero if any count differs from judge_final N2.1 / N2.2 or from these totals.

Figure 1 v2: the "Select on validation" and "Confirm on test" boxes add "trained at microbatch 64, no checkpointing" (the value is read from the pinned run files: all 744 selection and 300 confirmation runs record microbatch 64 and checkpointing false); the "Measured peak" box adds "(used for feasibility)"; the caption adds one sentence on the same two points. The five pipeline boxes are 0.3 cm taller (bottom edge 3.75 -> 3.45 cm in drawing units) so the added lines fit; the figure size is unchanged.

check_figures.py: both new figures are registered next to the existing ones (FIGS, FIG_NUMBER, recomputation functions exp_design_v2 and exp_a2_accounting_v2, and the source n1_input); the existing checks are unchanged.

### Reproduce
From the repository root, with the figure environment of the README (`.venv-fig`):
```
.venv-fig/bin/python result/figure/s8/scripts/fig1_design_v2.py
.venv-fig/bin/python result/figure/s8/scripts/fig4_a2_accounting_v2.py
.venv-fig/bin/python result/figure/s8/check_figures.py
```
