# CONTRIBUTION_CONTRACT — loopbudget (I-28 + I-31) — v4, FROZEN 2026-10-10T08:40:50Z with PREREGISTERED_loopbudget_v4.yaml (S2 claim contract; amendments 1 and 2 at S3 before any outcome, confirmed by the PI; amendment 3 = the PI's S3-gate decisions; v1 to v3 unchanged)
Adopted direction: the PI's words of 2026-10-10, "proceed as all recommended" (research/CLAIM_OPTIONS_S1_v3.md). Machine-readable protocol: PREREGISTERED_loopbudget_v4.yaml. Frozen by src/gates/freeze_prereg.py after audit r001 closed S1 and before any confirmatory run; the hashes of this file and the YAML are in PREREGISTERED_loopbudget_v4.sha256. Changes only through DECISION_LOG.md and a new version.

## First line
Parameter-count arguments for depth tying (one block executed T times stores the parameter state of a single block) do not by themselves decide which architecture class trains best under a hard cap on training memory, because activations, transient buffers and memory-lean untied alternatives also set the peak; within a frozen registry of depth-tied and distinct-parameter configurations trained from scratch on three tinyML tasks, we test, across seven caps on peak requested PyTorch CPU tensor bytes from 256 KiB to 1024 KiB, whether the tied class's validation-selected configuration reaches better held-out accuracy than the untied class's (A1), and whether a lifetime-aware byte accounting frozen before training predicts which configurations fit each cap (A2).

## Claims (one primary, one supporting)
- Primary A1, conjuncts N1.1 (judged) and N1.2 (descriptive). N1.1: at least one (task, cap) cell shows a tied advantage above the task's floor (one-sided paired t over confirmation seeds, Holm over 21 cells). Falsifier (FAIL): every cell with a defined difference has a one-sided 95 percent upper bound below its floor (an intersection-union test, so no multiplicity adjustment is needed for this all-cells statement). Otherwise MISS, one top-up of two seeds (11 and 12), then INCONCLUSIVE. N1.2 reports per task whether events form no event, a band, an everywhere-tested result or multiple crossings, using the E_plus, E_minus and zero-sign events.
- Supporting A2, conjuncts N2.1 and N2.2 (both judged). N2.1: the accounting frozen at the end of S3 predicts each registry configuration's minimum-over-knobs peak within 5 percent for at least 95 percent of configurations and classifies fit on boundary pairs with accuracy at least 0.98 (FAIL below 90 percent or 0.95). N2.2: on boundary pairs it beats each of three graded baselines (parameter count with the state multiplier; persistent weights, gradients and optimizer state; the static four-component sum), exact McNemar with Holm over three. No accuracy prediction is claimed.

## Practical effect, metric and estimand
- Metric: top-1 test accuracy, percentage points. Memory metric: peak requested PyTorch CPU tensor bytes in the stated stack (measured_object in the YAML; includes parameters, gradients, optimizer state with step tensors, microbatch inputs and targets, saved tensors and transients; excludes Python objects, allocator headers, BLAS/oneDNN scratch, dataset and loader buffers; one thread; steady-state step). No device, SRAM, MCU-feasibility or GPU statement is made.
- Estimand per (task, cap): mean over confirmation seeds of the paired test-accuracy difference between the two classes' selected configurations.
- Floor per task: max(1.0, 2 x SD) points, SD = sample standard deviation over seeds 101 to 105 of the final-epoch validation accuracy of the untied reference configuration (untied, T = 4, d = 32, Adam at its selected learning rate). The rule is frozen now; a script computes the three values at the end of S3 from training and validation data only, they are written to DECISION_LOG.md and locked before any S4 run, and they are never changed in response to test accuracy or any tied-versus-untied result (a diverged floor run sets that task's floor to 1.0). Reproduction tolerances (S3 equivalence tests: relative 1e-5 for parameters after accumulation, 1e-6 for exact-checkpoint gradients) are separate from this scientific floor.

## Registry, knobs and training (frozen)
- Architecture: isotropic MLP-Mixer blocks on task tokens (HAR 16 x 72, KWS 49 x 10 MFCC, CIFAR-10 64 x 48); LayerNorm only; no dropout.
- Tied class: shared (T = 2, 4, 8), two tied groups (T = 4, 8), reversible shared (T = 2, 4, 8). Untied class: untied (T = 2, 4, 8), reversible untied (T = 2, 4, 8), one wide layer (T = 1). Widths 24, 32, 48, 64 (wide layer also 96, 128). Optimizers Adam, SGD with momentum, plain SGD, Adafactor (single-tensor implementations). 248 configurations per task.
- Memory knobs that do not change the trained model beyond floating-point reordering: microbatch 1 to 64 with accumulation to an effective batch of 64; exact per-block checkpointing (S3 equivalence tests on all three tasks; the accumulation tests run in float64 so that they test the code rather than fp32 rounding: the accumulated gradient and the parameters after three updates match the full batch for every optimizer, Adafactor's zero-gradient tensors excepted; in fp32 the same comparisons differ by rounding of the size that reordering the examples of a batch produces, and are reported). A configuration is feasible at a cap if its minimum-over-knobs measured peak fits.
- Training from scratch with fixed epochs (HAR 30, KWS 20, CIFAR-10 30), cosine schedule after one epoch of linear warmup, no weight decay, gradient clipping 1.0; learning rates chosen per (task, optimizer, class) from a five-value grid on validation with one reference configuration per class at S3 (equal search effort; final-epoch validation accuracy at seed 0, ties to the smaller learning rate; amendment v4 extended each frozen three-value grid by one half-decade step at both ends after 20 of 24 S3 selections fell at an edge; a selection still at an edge is reported).

## Mandatory baselines and controls
- Classical and trivial bars per task: majority class; a linear softmax classifier on the flattened tokens (feasible at every cap). A cell where neither selected configuration beats the linear bar is flagged.
- The untied class is the standard comparator; the pointer-shared stack is a sanity check whose equality with the looped block is expected (S3 test), never a falsifier.
- Controls in the pilot: graded feasibility baselines (T1); knob-wise measurement of saved activations and transients (T2); the optimizer menu for both classes (T3); the pointer-shared sanity test (T4); the unbounded-cap control cell outside the multiplicity family (T7); equal learning-rate search, validation-only selection and fresh confirmation seeds (T9); two tied groups and reversible structures (T10); training from scratch with disjoint splits (T11). Deferred to S6 if GO: compute-limited sensitivity (T5), the alternative implementation rerun (T6), effective-batch sensitivity (T8), the approximate-gradient stratum, the ConvMixer family.

## Splits, seeds and inference units
- HAR: official volunteer split; validation = the four highest-numbered training volunteers. KWS: official validation list; official 12-class test archive; unknown and silence 10 percent each. CIFAR-10: last 5,000 training images as validation. Test data are read only by confirmation runs (code-enforced and tested at S3).
- Seeds: selection 0; confirmation 1 to 10; top-up 11 and 12 once; floor seeds 101 to 105 (validation only). Inference unit: the confirmation seed, paired across classes. Ten confirmation seeds because the Holm first step over 21 cells needs a one-sided t of about 3.7 at df 9 (about 6.0 at df 4 with five seeds).

## Uncertainty, multiplicity and interim looks
- Paired t intervals (df = seeds minus 1), alpha 0.05; Holm over the 21 cells for each event family (E_plus, E_minus, zero sign), which covers union statements such as N1.1 PASS; the all-cells FAIL statement is an intersection-union test at the unadjusted level. One analysis after all confirmation runs, the single top-up if MISS, no other look at test accuracy.

## Budget and kill rules
- Ratified CPU cap for S0 to S4: 300 core-hours (raised from 250 by the PI on 2026-10-10 after the S3 re-estimate in research/s3/resize/; research/s2/pilot_sizing.json gave the S2 estimate). Stopping policy adopted by the PI on 2026-10-10: unless N1.1 is PASS, the line does not expand beyond S4 and is written as a short bounded report or the accounting note; the stopping policy carries no scientific conclusion by itself. Outcome language: FAIL means no practically important tied gain within the tested scope (every one-sided upper bound below its floor) and is the only outcome that supports removing tying from the tested menu; INCONCLUSIVE means no gain above the floor was detected and the intervals are too wide to exclude one. GO (N1.1 PASS): S5 freezes the full study. No RE-SCOPE on feasibility alone. A dated G2 concurrent-work re-sweep precedes the first S4 run.

## Differences from the closest work (NOVELTY_MATRIX.md v2)
- Seed paper M1 (Jeong and Oh 2026, JSA; R1 manuscript read, research/s1/sources/M1_R1_notes.json): suffix fine-tuning of one pretrained ResNet-8 with SGD-momentum at batch 32; a memory predictor with transient buffers left to an overhead term; a single-batch assumption; no weight sharing. Here: training from scratch; depth tying as the lever; transients predicted explicitly and validated on held-out configurations; microbatch accumulation and exact checkpointing as memory-only knobs.
- RevGNN (M11): tied versus untied reversible GNNs at nearly equal measured GPU training peaks, gigabyte scale. Here: a frozen registry and a common cap at sub-megabyte budgets on tinyML tasks with a lean-training menu for both classes.

## Forbidden wording
REJECT_MEMO.md v2, section "Forbidden wording", applies in full; in particular: no "first" claims about comparing tied and untied models under training memory; no "at equal measured peak bytes" (say "under the same cap on peak requested CPU tensor bytes" and report each selected configuration's actual peak and slack); no SRAM, MCU, on-device or GPU statements; no "sharing never helps"; no accounting-predicts-accuracy statement.

## Failures and code identity
Every planned run is reported. A diverged run (non-finite loss) is kept, with non-finite logits counted as wrong predictions; an execution error (crash, killed process, out-of-memory) is rerun once with the same seed and configuration, both attempts logged; nothing else is repeated or dropped. The training harness and profiler are hashed before the S3 learning-rate and floor runs (which train only the two reference configurations and record no memory); the feasibility predictor, calibrated on the disjoint calibration grid only, is hashed before any registry configuration is profiled; confirmatory runners check the YAML hash and these code hashes and stop with ERROR on a mismatch.

## Fallback and changes
Deviations from this contract are recorded in DECISION_LOG.md before any outcome they could depend on; after exposure, plausible readings are disclosed and outcome-informed choices labelled exploratory. Tasks, caps, registry and floor rule are not changed after outcomes.

## Amendment 1 (v2; DECISION_LOG.md 2026-10-10T06:20:53Z; before any outcome)
The microbatch-accumulation equivalence gate moves from the parameters after three updates to the accumulated gradient, for every optimizer, because for Adam and Adafactor the frozen parameter check fails even between two full-batch computations that differ only in the order of the examples (research/s3/amendment_v2/); the parameter check stays for SGD and SGD with momentum. The relative-error definition, the code-freeze reading, the schedule formula, the learning-rate tie rule and the torchaudio environment amendment are folded in from DECISION_LOG entries. No task, cap, registry entry, seed, floor rule, decision rule or claim changed.

## Amendment 2 (v3; DECISION_LOG.md 2026-10-10T06:48:03Z; before any outcome)
The fp32 gradient gate of v2 failed on the Speech Commands and CIFAR-10 shapes (1.03e-5 to 2.05e-5) because the microbatch-64 fp32 gradient itself carries rounding of that size; the accumulation gates therefore run in float64 at the unchanged tolerances, where microbatch 8 and 1 equal microbatch 64 within 7e-14, and the parameter gate covers every optimizer again (Adafactor's zero-gradient tensors excepted). fp32 comparisons are reported. The checkpoint, reversible and pointer-shared gates stay in fp32, and every equivalence test covers all three tasks. No task, cap, registry entry, seed, floor rule, decision rule or claim changed.

## Amendment 3 (v4; DECISION_LOG.md 2026-10-10T08:39:22Z; the PI's S3-gate decisions, before any S4 run)
The PI adopted the five-value learning-rate grids (each frozen grid extended by one half-decade step at both ends, the same for both classes), raised the CPU cap for S0 to S4 to 300 core-hours, and confirmed amendments 1 and 2. All S3 artifacts that depend on the protocol hash are made again under v4 in versioned locations; the v3 artifacts stay in place. No task, cap, registry entry, seed, floor rule, decision rule or claim changed.

## FREEZE-KEYS (must equal PREREGISTERED_loopbudget_v4.yaml; checked by src/gates/freeze_prereg.py)
```yaml
protocol: loopbudget
version: 4
caps_kib: [256, 323, 406, 512, 645, 813, 1024]
effective_batch: 64
microbatch: [1, 2, 4, 8, 16, 32, 64]
widths: [24, 32, 48, 64]
wide_widths: [96, 128]
depths: [1, 2, 4, 8]
optimizers: [adam, sgd_momentum, sgd, adafactor]
epochs: {HAR: 30, KWS: 20, CIFAR10: 30}
seeds_selection: [0]
seeds_confirmation: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds_top_up: [11, 12]
alpha: 0.05
configurations_per_task: 248
```
