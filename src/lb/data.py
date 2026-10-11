"""Loading API for the S3 task tensors (data/<TASK>/{train,val,test}.pt written by src/lb/prepare_data.py).

load_task(task, split, allow_test=False) -> (X float32 [N, n_tokens, token_dim], y int64 [N]).
The test split is gated: requesting it without allow_test=True raises TestSplitAccess before any file is opened
(development runs never pass allow_test; only confirmation runs do). Every file is checked against the sha256 in
research/s3/data_manifest.json before it is deserialized; a mismatch raises DataIntegrityError. Nothing is downloaded.
"""
import hashlib
import json
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research" / "s3" / "data_manifest.json"
TASKS = ("HAR", "KWS", "CIFAR10")
SPLITS = ("train", "val", "test")


class TestSplitAccess(PermissionError):
    """Raised when the test split is requested without allow_test=True."""


class DataIntegrityError(RuntimeError):
    """Raised when a data file does not match the manifest."""


def _check_request(task, split, allow_test):
    if task not in TASKS:
        raise ValueError(f"task {task!r} not in {TASKS}")
    if split not in SPLITS:
        raise ValueError(f"split {split!r} not in {SPLITS}")
    if split == "test" and allow_test is not True:
        raise TestSplitAccess(f"{task}/test requested without allow_test=True; development code never reads the test split")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _verified_bytes(task, fname, data_root, manifest_path):
    man = json.loads(pathlib.Path(manifest_path).read_text())
    try:
        rec = man["tasks"][task]["outputs"][fname]
    except KeyError as e:
        raise DataIntegrityError(f"{task}/{fname} not in manifest {manifest_path}") from e
    path = pathlib.Path(data_root) / task / fname
    got = _sha256(path)
    if got != rec["sha256"]:
        raise DataIntegrityError(f"{path}: sha256 {got} != manifest {rec['sha256']}")
    return path, rec


def load_task(task, split, *, allow_test=False, data_root=None, manifest_path=None):
    """Return (X, y) for task in TASKS and split in SPLITS. data_root / manifest_path override the defaults (tests)."""
    _check_request(task, split, allow_test)
    data_root = ROOT / "data" if data_root is None else data_root
    manifest_path = MANIFEST if manifest_path is None else manifest_path
    path, rec = _verified_bytes(task, f"{split}.pt", data_root, manifest_path)
    d = torch.load(path, map_location="cpu", weights_only=True)
    X, y = d["X"], d["y"]
    if list(X.shape) != rec["X_shape"] or list(y.shape) != rec["y_shape"] or X.dtype != torch.float32 or y.dtype != torch.int64:
        raise DataIntegrityError(f"{path}: shape/dtype {tuple(X.shape)} {X.dtype} {tuple(y.shape)} {y.dtype} != manifest")
    return X, y


def load_meta(task, split, *, allow_test=False, data_root=None, manifest_path=None):
    """Per-sample IDs (volunteer, clip source and speaker, source index) for a split; same gating and sha256 check."""
    _check_request(task, split, allow_test)
    data_root = ROOT / "data" if data_root is None else data_root
    manifest_path = MANIFEST if manifest_path is None else manifest_path
    path, _ = _verified_bytes(task, f"{split}_meta.json", data_root, manifest_path)
    return json.loads(path.read_text())
