"""Load the frozen preregistration in force (PREREGISTERED_loopbudget_v4.yaml: amendment 3 of v1, the PI's S3-gate
decisions, DECISION_LOG.md 2026-10-10T08:39:22Z; v1 to v3 kept unchanged) and verify its sha256 and CONTRIBUTION_CONTRACT_v4.md's against PREREGISTERED_loopbudget_v4.sha256; any mismatch raises PreregMismatch (callers stop with ERROR)."""
import hashlib
import pathlib
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
YAML_PATH = ROOT / "PREREGISTERED_loopbudget_v4.yaml"
SHA_PATH = ROOT / "PREREGISTERED_loopbudget_v4.sha256"
CONTRACT_PATH = ROOT / "CONTRIBUTION_CONTRACT_v4.md"


class PreregMismatch(RuntimeError):
    pass


def _expected():
    out = {}
    for line in SHA_PATH.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = h
    return out


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    exp = _expected()
    for path in (YAML_PATH, CONTRACT_PATH):
        got = _sha(path)
        if exp.get(path.name) != got:
            raise PreregMismatch(f"{path.name}: sha256 {got} != frozen {exp.get(path.name)}")
    return exp


def load(verify_hash=True):
    if verify_hash:
        verify()
    d = yaml.safe_load(YAML_PATH.read_bytes())
    if d.get("status") != "FROZEN":
        raise PreregMismatch(f"status {d.get('status')} is not FROZEN")
    return d


def yaml_sha256():
    return _sha(YAML_PATH)


def task_cfg(task, prereg=None):
    p = prereg or load()
    t = p["tasks"][task]
    return {"task": task, "n_tokens": int(t["tokens"]["n_tokens"]), "token_dim": int(t["tokens"]["token_dim"]), "classes": int(t["classes"])}
