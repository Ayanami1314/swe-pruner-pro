"""Resume & append-safe JSONL writing for benchmark runners.

`load_done_ids(path, id_key="id")` — read an append-only jsonl (possibly
partially written across runs) and return the set of values seen at
`id_key`. Skips malformed lines rather than aborting, since a mid-write
crash can leave a truncated final line.

`SafeJsonlWriter(path)` — thread-safe append with shared lock across
ThreadPool workers. Each benchmark currently rolls its own `write_lock +
jsonlines.open("a")`; this consolidates it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


def load_done_ids(path: Path | str, id_key: str = "id") -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    done: set[str] = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            val = obj.get(id_key)
            if val is not None:
                done.add(val)
    return done


class SafeJsonlWriter:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
