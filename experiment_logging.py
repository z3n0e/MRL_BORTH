from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from wandb_utils import wandb_log


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


class MetricLogger:
    def __init__(self, path: Path | None = None, wandb_run=None):
        self.path = path
        self.wandb_run = wandb_run
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, source: str, metrics: Dict[str, Any]) -> None:
        payload = {"source": source, "timestamp": time.time(), **metrics}
        if self.path is not None:
            with self.path.open("a") as handle:
                handle.write(json.dumps(payload) + "\n")
        wandb_log(self.wandb_run, metrics)
