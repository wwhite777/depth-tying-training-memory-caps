"""Run launcher: at most N concurrent single-thread subprocesses (N <= 8), resumable.

  cd /home/wjeong/loopbudget && PYTHONPATH=src python -m lb.launch --specs RUNS.jsonl --jobs N --log LAUNCHER.log

Each JSONL line: {"run_id": str, "args": [CLI arguments without --out], "out": dir, "module": "lb.train" (default)
or "lb.profiler"}. The launcher appends "--out <out>" and runs `python -m <module> ...` from the project root with
CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, PYTHONPATH=<root>/src.
Per attempt k the out dir holds launch_attempt<k>.log (stdout+stderr) and launch_attempt<k>.json (exit record).
Rules: a spec whose out dir has DONE.json is skipped. An attempt that exits with code 2 is a refusal (guard violation
or code-freeze ERROR of lb.train / lb.profiler, an argparse usage error, the CUDA refusal): it is recorded as
"refused", is not an execution error, and the spec is never rerun (also not on resume). Any other attempt that ends
without DONE.json (non-zero exit, or exit 0 without DONE.json) is an execution error and the spec is rerun once
(failures rule of the YAML); after two counted attempts the spec is reported as failed and never rerun. An attempt the launcher itself terminated on SIGINT/SIGTERM
is recorded as "interrupted" and is not counted; an attempt log without an exit record (host crash, launcher killed
with SIGKILL) is counted unless its process is still alive, in which case the spec is skipped as still running.
The launcher log (JSON lines) records launcher start/end and per-attempt start, end, exit status and log path.
Launcher exit code: 0 if every spec is done, 1 if any failed, was refused or is still running elsewhere, 130 if
interrupted.
"""
import argparse
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_JOBS = 8
MODULES = ("lb.train", "lb.profiler")
REFUSAL_EXIT = 2


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_specs(path):
    specs, ids = [], set()
    for n, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        s = json.loads(line)
        for k in ("run_id", "args", "out"):
            if k not in s:
                raise ValueError(f"{path}:{n}: missing {k!r}")
        s.setdefault("module", "lb.train")
        if s["module"] not in MODULES:
            raise ValueError(f"{path}:{n}: module {s['module']!r} not in {MODULES}")
        if "--out" in s["args"]:
            raise ValueError(f"{path}:{n}: args must not contain --out (the launcher appends it)")
        if s["run_id"] in ids:
            raise ValueError(f"{path}:{n}: duplicate run_id {s['run_id']!r}")
        ids.add(s["run_id"])
        out = pathlib.Path(s["out"])
        s["out"] = out if out.is_absolute() else ROOT / out
        specs.append(s)
    return specs


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refused_attempt(out):
    """The attempt number of a recorded refusal (exit 2) in the out dir, or None."""
    for rec in sorted(out.glob("launch_attempt*.json")):
        if json.loads(rec.read_text()).get("status") == "refused":
            return int(rec.stem.replace("launch_attempt", ""))
    return None


def attempt_state(out):
    """(counted attempts, next attempt number, live pid or None) from the out dir's attempt files."""
    counted, k, live = 0, 0, None
    for log in sorted(out.glob("launch_attempt*.log")):
        n = int(log.stem.replace("launch_attempt", ""))
        k = max(k, n)
        rec = out / f"launch_attempt{n}.json"
        if rec.exists():
            r = json.loads(rec.read_text())
            if r.get("status") in ("interrupted", "refused"):
                continue
            if r.get("status") == "running":
                if _pid_alive(r["pid"]):
                    live = r["pid"]
                    continue
            counted += 1
        else:
            counted += 1
    return counted, k + 1, live


class Launcher:
    def __init__(self, specs, jobs, log_path):
        if not 1 <= jobs <= MAX_JOBS:
            raise ValueError(f"--jobs must be in 1..{MAX_JOBS}")
        self.specs, self.jobs, self.log_path = specs, jobs, pathlib.Path(log_path)
        self.running = {}            # run_id -> (Popen, spec, attempt, log file handle, start)
        self.interrupted = False

    def log(self, **kw):
        kw = {"utc": utc_now(), **kw}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(kw, default=str) + "\n")

    def _env(self):
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1"})
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src")] + [x for x in [env.get("PYTHONPATH")] if x])
        return env

    def _start(self, s, attempt):
        out = s["out"]
        out.mkdir(parents=True, exist_ok=True)
        logp = out / f"launch_attempt{attempt}.log"
        fh = open(logp, "w")
        cmd = [sys.executable, "-m", s["module"], *[str(a) for a in s["args"]], "--out", str(out)]
        p = subprocess.Popen(cmd, cwd=ROOT, env=self._env(), stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        start = utc_now()
        (out / f"launch_attempt{attempt}.json").write_text(json.dumps(
            {"status": "running", "pid": p.pid, "cmd": cmd, "start_utc": start}))
        self.running[s["run_id"]] = (p, s, attempt, fh, start)
        self.log(event="start", run_id=s["run_id"], attempt=attempt, pid=p.pid, log=logp, cmd=cmd)

    def _finish(self, run_id, rc, status):
        p, s, attempt, fh, start = self.running.pop(run_id)
        fh.close()
        done = (s["out"] / "DONE.json").exists()
        if status is None:
            status = "done" if done else ("refused" if rc == REFUSAL_EXIT else "error")
        (s["out"] / f"launch_attempt{attempt}.json").write_text(json.dumps(
            {"status": status, "pid": p.pid, "exit": rc, "done_json": done, "start_utc": start,
             "end_utc": utc_now()}))
        self.log(event="end", run_id=run_id, attempt=attempt, pid=p.pid, exit=rc, status=status, done_json=done)
        return status

    def _terminate_all(self, signum, frame):
        self.interrupted = True

    def run(self):
        signal.signal(signal.SIGTERM, self._terminate_all)
        signal.signal(signal.SIGINT, self._terminate_all)
        self.log(event="launcher_start", n_specs=len(self.specs), jobs=self.jobs, pid=os.getpid(), argv=sys.argv)
        queue, summary = [], {}
        for s in self.specs:
            if (s["out"] / "DONE.json").exists():
                summary[s["run_id"]] = "skipped_done"
                self.log(event="skip", run_id=s["run_id"], reason="DONE.json exists")
                continue
            counted, nxt, live = attempt_state(s["out"])
            refused = refused_attempt(s["out"])
            if refused is not None:
                summary[s["run_id"]] = "refused"
                self.log(event="skip", run_id=s["run_id"], reason=f"refused (exit {REFUSAL_EXIT}) in attempt {refused}")
            elif live is not None:
                summary[s["run_id"]] = "running_elsewhere"
                self.log(event="skip", run_id=s["run_id"], reason=f"attempt process {live} still alive")
            elif counted >= 2:
                summary[s["run_id"]] = "failed"
                self.log(event="skip", run_id=s["run_id"], reason="two counted attempts without DONE.json")
            else:
                queue.append([s, nxt, counted])
        while (queue or self.running) and not self.interrupted:
            while queue and len(self.running) < self.jobs and not self.interrupted:
                s, nxt, counted = queue.pop(0)
                self._start(s, nxt)
                s["_counted"] = counted
            for run_id in list(self.running):
                p = self.running[run_id][0]
                rc = p.poll()
                if rc is None:
                    continue
                s, attempt = self.running[run_id][1], self.running[run_id][2]
                status = self._finish(run_id, rc, None)
                if status == "done":
                    summary[run_id] = "done"
                elif status == "refused":
                    summary[run_id] = "refused"
                    self.log(event="refused", run_id=run_id, attempt=attempt, exit=rc, reason="exit 2: not rerun")
                elif s["_counted"] + 1 < 2:
                    self.log(event="rerun", run_id=run_id, after_attempt=attempt, reason="execution error")
                    queue.append([s, attempt + 1, s["_counted"] + 1])
                else:
                    summary[run_id] = "failed"
            time.sleep(0.2)
        if self.interrupted:
            for run_id in list(self.running):
                p = self.running[run_id][0]
                p.terminate()
                try:
                    rc = p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    rc = p.wait()
                self._finish(run_id, rc, "interrupted")
                summary[run_id] = "interrupted"
        code = 130 if self.interrupted else (0 if all(v in ("done", "skipped_done") for v in summary.values())
                                             and len(summary) == len(self.specs) else 1)
        self.log(event="launcher_end", exit=code, summary=summary)
        return code


def main(argv=None):
    ap = argparse.ArgumentParser(description="resumable launcher of single-thread runs")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--jobs", type=int, required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    if not 1 <= a.jobs <= MAX_JOBS:
        ap.error(f"--jobs must be in 1..{MAX_JOBS}")
    return Launcher(read_specs(a.specs), a.jobs, a.log).run()


if __name__ == "__main__":
    sys.exit(main())
