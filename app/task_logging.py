from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TaskLogger:
    def __init__(self, log_path: Path, enabled: bool = True) -> None:
        self.log_path = log_path
        self.enabled = enabled
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **payload,
        }
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
