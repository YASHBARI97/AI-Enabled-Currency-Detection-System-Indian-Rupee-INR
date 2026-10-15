"""Config loading and environment settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class Settings:
    model_path: str = "artifacts/best_model.pt"
    model_meta: str = "artifacts/model_meta.json"
    max_upload_mb: float = 5.0
    log_level: str = "INFO"
    history_path: str = "reports/prediction_history.csv"
    enable_history: bool = False
    cors_origins: tuple = ()

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        return cls(
            model_path=g("CURRENCY_MODEL_PATH", cls.model_path),
            model_meta=g("CURRENCY_MODEL_META", cls.model_meta),
            max_upload_mb=float(g("CURRENCY_MAX_UPLOAD_MB", cls.max_upload_mb)),
            log_level=g("CURRENCY_LOG_LEVEL", cls.log_level),
            history_path=g("CURRENCY_HISTORY_PATH", cls.history_path),
            enable_history=g("CURRENCY_ENABLE_HISTORY", "false").lower() == "true",
            cors_origins=tuple(o.strip() for o in g("CURRENCY_CORS_ORIGINS", "").split(",") if o.strip()),
        )
