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
