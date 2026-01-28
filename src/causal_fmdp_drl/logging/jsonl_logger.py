"""Simple JSONL metrics logger."""

import json
from pathlib import Path
from typing import Any, Dict


class JSONLLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a")

    def log(self, data: Dict[str, Any]) -> None:
        self._file.write(json.dumps(data) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
