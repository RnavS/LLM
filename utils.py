from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    import torch
except ImportError:  # pragma: no cover - allows non-training utilities to work before dependencies are installed
    torch = None


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_console_output() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def save_text(path: str | Path, content: str) -> None:
    Path(path).write_text(content, encoding="utf-8")


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def set_seed(seed: int) -> None:
    if torch is None:
        raise ImportError("torch is required to set the random seed for training.")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if torch is None:
        raise ImportError("torch is required for training or inference.")
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> int:
    if torch is None:
        raise ImportError("torch is required to inspect model parameters.")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def format_seconds(total_seconds: float) -> str:
    total_seconds = max(total_seconds, 0.0)
    minutes, seconds = divmod(int(total_seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def lr_with_warmup_and_cosine_decay(
    step: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return max_lr * float(step + 1) / float(max(1, warmup_steps))
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine * (max_lr - min_lr)


def resolve_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        latest = candidate / "latest.pt"
        if latest.exists():
            return latest
        raise FileNotFoundError(f"No latest.pt checkpoint found in {candidate}")
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")
    return candidate


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[Dict[str, Any], Path]:
    if torch is None:
        raise ImportError("torch is required to load checkpoints.")
    resolved = resolve_checkpoint_path(path)
    checkpoint = torch.load(resolved, map_location=device)
    return checkpoint, resolved


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    if torch is None:
        raise ImportError("torch is required to save checkpoints.")
    torch.save(payload, Path(path))


def elapsed_since(start_time: float) -> str:
    return format_seconds(time.time() - start_time)
