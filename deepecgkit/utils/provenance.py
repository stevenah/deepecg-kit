"""What a run cannot reproduce from its own settings: the code, the machine and the data.

These values are written into ``run_metadata.json`` by ``deepecg train`` so
that a result can be traced to a software revision, a host and device, and
the dataset files that were on disk, without relying on notes kept outside
the run directory.
"""

import hashlib
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

__all__ = ["dataset_fingerprint", "environment_summary", "git_commit"]

#: More files than this are not fingerprinted file by file; the count alone is kept.
FINGERPRINT_FILE_LIMIT = 200_000


def git_commit(path: str | Path | None = None) -> dict[str, Any] | None:
    """Return the commit the package's working tree is at, if it is a git checkout.

    Args:
        path: A directory inside the checkout; defaults to the package directory.

    Returns:
        ``{"commit": <full hash>, "dirty": <bool>}``, or None when the package
        was not run from a git checkout (an installed wheel, for instance).
    """
    start = Path(path) if path is not None else Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if head.returncode != 0:
            return None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=start,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def environment_summary(device: Any = None) -> dict[str, Any]:
    """Describe the interpreter, the libraries, the host and the compute device.

    Args:
        device: The resolved torch device the run uses, if known.

    Returns:
        A JSON-serialisable mapping.
    """
    summary: dict[str, Any] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": str(device) if device is not None else None,
    }
    for name in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            module = __import__(name)
            summary[f"{name}_version"] = getattr(module, "__version__", None)
        except ImportError:
            summary[f"{name}_version"] = None
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            summary["device_name"] = torch.cuda.get_device_name(device)
            summary["cuda_version"] = torch.version.cuda
            summary["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:  # pragma: no cover - driver-specific failures
            pass
    summary["executable"] = sys.executable
    return summary


def dataset_fingerprint(data_dir: str | Path | None) -> dict[str, Any] | None:
    """Fingerprint the files of a dataset directory by name and size.

    Hashing the contents of a multi-gigabyte corpus on every run is too slow,
    so the fingerprint covers the sorted relative paths and byte sizes of
    every regular file. It changes when a file is added, removed or
    truncated, which is what a reader of a run needs to know; it does not
    detect a byte flip inside a file of unchanged size.

    Args:
        data_dir: The dataset directory.

    Returns:
        ``{"path", "n_files", "total_bytes", "sha256"}``, or None when the
        directory does not exist.
    """
    if data_dir is None:
        return None
    root = Path(data_dir)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    n_files = 0
    total = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        n_files += 1
        if n_files > FINGERPRINT_FILE_LIMIT:
            continue
        size = path.stat().st_size
        total += size
        digest.update(f"{path.relative_to(root).as_posix()}\0{size}\n".encode())
    return {
        "path": str(root),
        "n_files": n_files,
        "total_bytes": total,
        "sha256": digest.hexdigest() if n_files <= FINGERPRINT_FILE_LIMIT else None,
    }
