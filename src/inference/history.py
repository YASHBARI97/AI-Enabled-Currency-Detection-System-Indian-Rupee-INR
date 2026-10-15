"""Optional prediction history. Stores metadata only -- never the uploaded image."""
import csv
from datetime import datetime, timezone
from pathlib import Path


def record(path: str, pred: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp_utc", "status", "prediction", "confidence", "inference_ms"])
        w.writerow([datetime.now(timezone.utc).isoformat(), pred["status"], pred["predicted_denomination"],
                    pred["confidence"], pred["inference_ms"]])
