from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def lockfile_checksum(lock_path: Path) -> str:
    if not lock_path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(lock_path.read_bytes())
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def save_run_metadata(output_dir: Path, config_dict: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = Path("uv.lock")
    payload = {
        "python_version": os.sys.version,
        "torch_version": torch.__version__,
        "git_sha": git_sha(),
        "uv_lock_sha256": lockfile_checksum(lock_path),
        "config": config_dict,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2))
