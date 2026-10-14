"""S9-T timing probe: per-optimizer-step wall time of the frozen train_step at the feasible knob (argmin_knob from
research/s4/selection.json: microbatch 1, checkpointing as recorded) and at microbatch 64 without checkpointing, for
the 22 distinct configurations selected in the 21 finite-cap cells. No accuracy, no data files: one effective batch of
64 random inputs (torch.randn) and random labels of the task's shape. Imports only from src/lb (read only).

Run from the project root:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
    <venv>/bin/python research/s9/transfer_probe/timing_probe.py --out research/s9/transfer_probe/timing_<UTC>.json

Estimate per configuration (steps K = epochs x ceil(N_train / 64), from the YAML in force and data_manifest.json;
archived wall W = wall_seconds of result/s4/confirm/<run>__s1/result.json, trained at microbatch 64):
  pure     = K x t1                         (training steps only)
  additive = W + K x (t1 - t64)             (archived run with every step replaced; overhead kept as is)
  ratio    = K x t1 x W / (K x t64)         (card's ratio method: overhead scaled with the step time; upper bound)
t1, t64 = median of the timed steps.
"""
import argparse
import heapq
import json
import math
import pathlib
import statistics
import sys
import time

import torch

from lb import prereg
from lb.train import build_from_config, make_optimizer, parse_config_id, set_threads, train_step, utc_now

ROOT = pathlib.Path(__file__).resolve().parents[3]
WARMUP, TIMED, BATCH, SEED = 3, 10, 64, 1


def selected_configs():
    s = json.loads((ROOT / "research/s4/selection.json").read_text())
    out, n_cells = {}, 0
    for c in s["cells"]:
        if c["cap_kib"] == "unbounded":
            continue
        n_cells += 1
        for cls in ("tied", "untied"):
            e = c.get(cls)
            if e:
                out.setdefault(e["config_id"], e)
    return n_cells, out


def time_steps(cfg, task_cfg, opt_name, lr, microbatch, checkpointing):
    torch.manual_seed(SEED)
    model = build_from_config(cfg, task_cfg, checkpointing)
    opt = make_optimizer(opt_name, model.parameters(), lr)
    g = torch.Generator().manual_seed(SEED)
    X = torch.randn(BATCH, task_cfg["n_tokens"], task_cfg["token_dim"], generator=g)
    y = torch.randint(task_cfg["classes"], (BATCH,), generator=g)
    idx = torch.arange(BATCH)
    model.train()
    times, losses = [], []
    for i in range(WARMUP + TIMED):
        t0 = time.perf_counter()
        loss = train_step(model, opt, X, y, idx, microbatch, lr)
        dt = time.perf_counter() - t0
        losses.append(loss)
        if i >= WARMUP:
            times.append(dt)
    return {"microbatch": microbatch, "checkpointing": bool(checkpointing), "step_seconds": times,
            "median_s": statistics.median(times), "mean_s": statistics.fmean(times), "min_s": min(times),
            "max_s": max(times), "all_losses_finite": all(math.isfinite(v) for v in losses)}


def lpt_makespan(durations, workers):
    """Longest-processing-time-first greedy schedule on identical workers."""
    loads = [0.0] * workers
    heapq.heapify(loads)
    for d in sorted(durations, reverse=True):
        heapq.heappush(loads, heapq.heappop(loads) + d)
    return max(loads)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args(argv)
    threads = set_threads()
    p = prereg.load()
    man = json.loads((ROOT / "research/s3/data_manifest.json").read_text())
    n_cells, cfgs = selected_configs()
    print(f"{utc_now()} cells {n_cells} configs {len(cfgs)} threads {threads} torch {torch.__version__}", flush=True)
    rows = []
    for cid, e in cfgs.items():
        cfg = parse_config_id(cid, p)
        task = cfg["task"]
        task_cfg = prereg.task_cfg(task, p)
        knob = e["argmin_knob"]
        lr = float(e["lr"])
        epochs = int(p["training"]["epochs"][task])
        n_train = int(man["tasks"][task]["outputs"]["train.pt"]["X_shape"][0])
        K = epochs * math.ceil(n_train / BATCH)
        run = pathlib.Path(e["result"]).parent.name.rsplit("__s", 1)[0]
        arch = json.loads((ROOT / "result/s4/confirm" / f"{run}__s1" / "result.json").read_text())
        assert arch["config_id"] == cid and arch["microbatch"] == 64 and not arch["checkpointing"] and arch["steps"] == K
        conf_walls = [json.loads(f.read_text())["wall_seconds"]
                      for f in sorted((ROOT / "result/s4/confirm").glob(f"{run}__s*/result.json"))]
        feas = time_steps(cfg, task_cfg, cfg["optimizer"], lr, int(knob["microbatch"]), bool(knob["checkpointing"]))
        m64 = time_steps(cfg, task_cfg, cfg["optimizer"], lr, 64, False)
        t1, t64, W = feas["median_s"], m64["median_s"], float(arch["wall_seconds"])
        est = {"pure_s": K * t1, "additive_s": W + K * (t1 - t64), "ratio_s": K * t1 * W / (K * t64)}
        row = {"config_id": cid, "task": task, "class": e["class"], "argmin_knob": knob, "lr": lr,
               "n_params": e["n_params"], "min_peak_bytes": e["min_peak_bytes"], "epochs": epochs, "n_train": n_train,
               "steps_per_run": K, "archived_run": str(pathlib.Path("result/s4/confirm") / f"{run}__s1"),
               "archived_wall_s": W, "archived_confirm_wall_s_median": statistics.median(conf_walls),
               "archived_confirm_runs": len(conf_walls), "archived_overhead_s": W - K * t64,
               "feasible": feas, "m64": m64, "slowdown_t1_over_t64": t1 / t64, "estimate": est}
        rows.append(row)
        print(f"{utc_now()} {cid} knob mb{knob['microbatch']} ckpt={knob['checkpointing']} t1={t1*1e3:.2f}ms "
              f"t64={t64*1e3:.2f}ms x{t1/t64:.1f} K={K} W={W:.1f}s est_add={est['additive_s']/3600:.3f}h "
              f"est_ratio={est['ratio_s']/3600:.3f}h", flush=True)
    totals = {}
    for method in ("pure_s", "additive_s", "ratio_s"):
        d = [r["estimate"][method] for r in rows]
        for design, seeds in (("i_22_runs_seed1", 1), ("ii_66_runs_seeds1to3", 3)):
            jobs = d * seeds
            totals[f"{design}__{method[:-2]}"] = {
                "runs": len(jobs), "core_hours": sum(jobs) / 3600,
                "wall_hours_lpt": lpt_makespan(jobs, a.workers) / 3600,
                "wall_hours_lower_bound": max(sum(jobs) / a.workers, max(jobs)) / 3600,
                "longest_run_hours": max(jobs) / 3600}
    out = {"kind": "s9_transfer_probe_timing", "torch": torch.__version__, "threads": threads,
           "warmup_steps": WARMUP, "timed_steps": TIMED, "effective_batch": BATCH, "seed": SEED, "workers": a.workers,
           "inputs": "torch.randn(64, n_tokens, token_dim), torch.randint(classes, (64,)), generator seed 1",
           "estimate_methods": {"pure": "K x t1", "additive": "W + K x (t1 - t64)", "ratio": "K x t1 x W / (K x t64)"},
           "prereg_sha256": prereg.yaml_sha256(), "argv": sys.argv, "end_utc": utc_now(),
           "rows": rows, "totals": totals}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps(totals, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
