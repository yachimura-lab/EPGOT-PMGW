"""Machine-readable provenance for the paper figures."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PACKAGE_NAMES = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "POT": "pot",
    "scikit-learn": "scikit-learn",
    "scipy": "scipy",
}


def array_sha256(value):
    """Hash an array together with its dtype and shape."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def solve_id(solver, coupling, parameters=None):
    """Stable identifier for one solver output and its parameters."""
    digest = hashlib.sha256()
    digest.update(str(solver).encode())
    digest.update(array_sha256(coupling).encode())
    digest.update(
        json.dumps(_jsonify(parameters or {}), sort_keys=True, separators=(",", ":")).encode()
    )
    return digest.hexdigest()


def coupling_marginal_diagnostics(coupling, source_weights, target_weights, partial=False):
    """Residuals for balanced marginals or partial sub-marginal constraints."""
    coupling = np.asarray(coupling, dtype=float)
    source_weights = np.asarray(source_weights, dtype=float)
    target_weights = np.asarray(target_weights, dtype=float)
    row = coupling.sum(axis=1)
    column = coupling.sum(axis=0)
    if partial:
        row_residual = float(np.max(np.maximum(row - source_weights, 0.0)))
        column_residual = float(np.max(np.maximum(column - target_weights, 0.0)))
    else:
        row_residual = float(np.max(np.abs(row - source_weights)))
        column_residual = float(np.max(np.abs(column - target_weights)))
    return {
        "row_residual": row_residual,
        "column_residual": column_residual,
        "minimum_entry": float(coupling.min()),
        "matched_mass": float(coupling.sum()),
        "constraint": "sub-marginal" if partial else "balanced marginal",
    }


def runtime_environment():
    """Versions needed to reproduce the numerical environment."""
    packages = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            packages[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[label] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_revision(repo_root=None):
    """Current source revision and whether it has uncommitted changes."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])

    def git(*args):
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        # Generated notebooks, sidecars and rendered figures are expected to
        # change during a clean regeneration. "dirty" here describes the
        # numerical source/lock inputs, not those generated artifacts.
        dirty = bool(
            git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                "src",
                "tools",
                "pyproject.toml",
                "uv.lock",
                ".python-version",
            )
        )
        submodule = git("submodule", "status", "vendor/PGW_Metric").lstrip("-+ ").split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return {"commit": None, "dirty": None, "pgw_metric_commit": None}
    return {
        "commit": commit,
        "dirty": dirty,
        "pgw_metric_commit": submodule,
    }


def write_figure_sidecar(
    path,
    *,
    figure,
    outputs,
    panels,
    inputs,
    seeds,
    notes=None,
):
    """Write one provenance sidecar for a canonical paper figure.

    NumPy arrays are represented by shape, dtype and SHA-256 rather than
    expanded into JSON. This keeps point-level couplings compact while still
    making the exact solve traceable. ``generated_at`` records when the
    sidecar was actually written, while ``source_date_epoch`` records the
    fixed timestamp used to make rendered figure bytes reproducible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output_records = []
    for output in outputs:
        output = Path(output)
        output_records.append(
            {
                "path": str(output),
                "sha256": file_sha256(output),
                "bytes": output.stat().st_size,
            }
        )

    source_epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "946684800"))
    payload = {
        "schema_version": 2,
        "figure": int(figure),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_date_epoch": source_epoch,
        "source": git_revision(),
        "environment": runtime_environment(),
        "seeds": _jsonify(seeds),
        "inputs": _jsonify(inputs),
        "panels": _jsonify(panels),
        "outputs": output_records,
        "notes": notes,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": array_sha256(value),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value
