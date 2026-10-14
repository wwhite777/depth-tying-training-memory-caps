# Cap-respecting transfer check: plan v1 (2026-10-14)

Status: NOT RUN. This is the decision basis for D1. It launches only on the PI's word. On that word its sha256 is recorded in DECISION_LOG.md before the first run, and nothing in it changes after the first result.
Labelling: exploratory, outside the frozen protocol (v4, a0453c36). The frozen A1 verdict (INCONCLUSIVE), its numbers and Table 1 are not recomputed and do not change, whatever this check shows.

## Question
The 22 configurations selected in the 21 finite-cap cells were trained for accuracy at microbatch 64 without checkpointing. At that setting their measured peak exceeds the cap of every one of the 42 finite class-cells, by 2.49 to 106.31 times. Feasibility used each configuration's minimum-memory setting. All 22 settings are at microbatch 1; 18 of the 22 use per-block checkpointing.

The frozen assumption is that the two settings train the same model up to floating-point reordering:
- checkpointing: fp32 gradients are bitwise identical;
- accumulation: identical in float64; in fp32 it differs by no more than a pure reordering of the examples.

The question: does training at the minimum-memory setting give the test accuracy measured at microbatch 64?

## Design
- **Configurations.** The 22 distinct configurations selected in the 21 finite-cap cells (research/s4/selection.json): 11 depth-tied and 11 distinct-parameter, all with Adafactor. Each is trained at its recorded argmin knob (microbatch 1; checkpointing as recorded, on for 18 and off for 4).
- **Seeds.** Design (ii), recommended: seeds 1, 2 and 3, which are archived confirmation seeds (66 runs). Design (i): seed 1 only (22 runs).
- **Recipe.** The frozen recipe, unchanged: the learning rate of research/s3/v4/lr_selected.json as recorded in selection.json, the task's epochs, effective batch 64, the pinned schedule, clipping at norm 1.0, one thread, deterministic algorithms, mode confirm. The harness guards allow microbatch 1 and checkpointing in confirm mode, and the harness verifies the code-freeze records before every run.
- **Comparator.** The archived run of the same configuration and seed (result/s4/confirm/<config>__lr<lr>__s<seed>/result.json, microbatch 64). Matching seeds give the same initialization and the same training-example order: the shuffle generator is seeded separately from the model initialization (lb/train.py fit()).
- **Environment.** ~/envs/jeongwoncheol_loopbudget_probe: Python 3.12.12, torch 2.10.0+cpu, numpy 2.5.3, scipy 1.18.1, pyyaml 6.0.3; freeze research/s9/transfer_probe/ENV_FREEZE_probe_20261014T070246Z.txt. The archived runs used torch 2.10.0+cu128 on CPU.
- **Precondition E (environment equivalence).** Before any minimum-memory run, rerun the seed-1 microbatch-64 confirmation of KWS|shared|d32|T2|adafactor and CIFAR10|shared|d24|T2|adafactor. Require bitwise-identical test_acc, final_val_acc and per-epoch train losses with the archive.
  - The HAR selection HAR|reversible_shared|d32|T4|adafactor was already reproduced bitwise on 2026-10-14 (research/s9/transfer_probe/repro_confirm_20261014T070259Z.log).
  - Fallback, fixed now: if either rerun differs, the comparator for all pairs becomes a fresh microbatch-64 rerun in this environment (about 10 more core-hours for design ii).
- **Launch.** The frozen launcher, `PYTHONPATH=src <probe venv>/bin/python -m lb.launch --specs <file> --jobs 7 --log result/s9/transfer/logs/launcher_<UTC>.log`. The spec files were written by research/s9/make_transfer_specs.py: research/s9/transfer_specs_precondE.jsonl (precondition E, run first), transfer_specs_seeds123.jsonl (design ii, 66 specs) and transfer_specs_seed1.jsonl (design i, 22 specs). The specs for the fallback reruns are generated only if precondition E fails. Each run writes to its own directory, result/s9/transfer/<config>__lr<lr>__s<seed>__m1[_ckpt]. The harness refuses to overwrite a finished run.

## Readouts (fixed before any run)
Let δ = test accuracy at the minimum-memory setting minus archived test accuracy at microbatch 64, in percentage points, for one configuration and seed.
- **R1 (primary): mean transfer shift.**
  - Take the mean of δ over the seeds of each configuration, then the mean over the 22 configurations with its 90% t interval (21 degrees of freedom).
  - Transfer of the mean is SUPPORTED if that interval lies inside ±0.5 pp: two one-sided tests at α = 0.05, margin half the smallest floor (Speech Commands, 1.00).
  - It is CONTRADICTED if the two-sided 95% interval excludes 0 and |mean| ≥ 0.5 pp.
  - Otherwise it is UNRESOLVED.
  - Per task, the same statistics are reported descriptively.
- **R2: size of single-run differences.** |δ| for every run, against that configuration's archived seed-to-seed SD of test accuracy (seeds 1–12, microbatch 64). If the two trajectories decorrelate fully, the expected |δ| is about 1.13 times that SD. Descriptive.
- **R3: the A1 cells.** For each of the 21 finite cells: Δ (tied minus distinct, mean over the seeds run) at the minimum-memory setting against the same Δ from the archived runs of the same seeds. Report the differences and any sign change. Descriptive; no event or verdict is recomputed.
- **R4: training curves.** The relative difference of the per-epoch training loss at the final epoch, and the first epoch at which the two curves differ. Descriptive.
- **Cost.** Wall time per run, against the probe's estimate.

All numbers come from one analysis script written and hashed before the first run: research/s9/transfer_analysis.py (synthetic tests research/s9/test_transfer_analysis.py, 12 of 12 pass; planted rule changes fail them). It reads only result/s9/transfer/, result/s9/transfer_env/, result/s4/confirm/, research/s4/selection.json, the final judge file (cells) and the probe's timing file (cost), and records the sha256 of every input. A failed run is rerun once under the launcher's policy; failures are reported, never dropped.

## Cost (measured 2026-10-14, research/s9/transfer_probe/timing_20261014T070439Z.json and timing_20261014T070622Z.json)
- Microbatch 1 costs 7–36 times the microbatch-64 time per optimizer step. This is per-call overhead: 64 small forward and backward passes per step.
- Design (i): 22.0–22.2 core-hours, 3.3–3.4 h wall on 7 single-thread jobs. The longest single run is CIFAR10|shared|d64|T8, at 3.4 h.
- Design (ii): 66.0–66.6 core-hours, 9.4–9.6 h wall on 7 jobs. The upper-bound "ratio" estimate is 72–75 core-hours and 10.3–10.7 h.
- Precondition E: under 0.3 core-hours.
- CPU total. The study used 148.6 core-hours (records). Design (ii) would bring it to about 215–224 core-hours. RESOURCE_CONTRACT.md caps S0–S4 at 300; this S9 check needs the PI's word to use the remaining headroom.
- Disk: under 10 MB of records. The probe venv has about 38 MiB of unique files; the rest are hardlinks into the uv cache.

## Reporting
- Whatever R1 shows, the paper gains a short section labelled exploratory: "Accuracy at the cap-respecting setting". It gives R1–R4 and the cost, and the abstract and conclusion state the result in one sentence.
- SUPPORTED: the A1 numbers stand as measured. The microbatch-64 accuracy is then backed by a direct check at the feasible setting for the selected configurations and seeds 1–3.
- CONTRADICTED or UNRESOLVED: the accuracy comparison is described as conditional on the untested transfer, and the size of the shift is reported.
- The results, scripts and records go into release v1.1 and the supplementary archive.
