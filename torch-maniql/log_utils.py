from __future__ import annotations

import json
import logging
import os
import socket
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def setup_logging(save_dir: str, *, name: str = "torch-maniql", level: str = "INFO") -> logging.Logger:
    """Configure console + file logging.

    Writes to <save_dir>/logs/<timestamp>.log and also logs to stdout.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Avoid duplicate handlers when scripts are imported/re-run.
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(sh)

    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    logger.info("Logging to %s", logfile)
    return logger


def _coerce_config(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    # argparse.Namespace or other object with __dict__
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {"config": str(config)}


def init_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    name: str | None,
    group: str | None,
    tags: list[str] | None,
    mode: str,
    save_dir: str,
    config: Any = None,
) -> Optional[Any]:
    """Initialize a W&B run. Returns wandb module if enabled else None."""
    if not enabled:
        return None

    try:
        import wandb  # type: ignore
    except Exception as e:
        raise RuntimeError("wandb is not installed but --wandb was provided") from e

    os.makedirs(save_dir, exist_ok=True)
    cfg = _coerce_config(config)
    cfg.setdefault("host", socket.gethostname())
    cfg.setdefault("save_dir", os.path.abspath(save_dir))

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags,
        mode=mode,  # "online" | "offline" | "disabled"
        dir=os.path.abspath(save_dir),
        config=cfg,
    )
    return wandb


def wandb_log(wandb_mod: Any | None, metrics: Dict[str, Any], *, step: int) -> None:
    if wandb_mod is None:
        return
    # Keep it simple: wandb will ignore non-serializable values.
    wandb_mod.log(dict(metrics), step=int(step))


def write_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
