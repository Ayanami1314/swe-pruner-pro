"""Sandbox abstractions for executing bash commands.

Two concrete implementations:

* `DockerSandbox` — default for aa-lcr / longcodeqa / oolong / trail-benchmark.
  Runs each command in a one-shot `docker run --rm --network=none` container
  with the workspace mounted read-only. Output is truncated head+tail when it
  exceeds `truncate_at`.

* `WhitelistSandbox` — sweqa only. No docker; commands run via `bash -c` in
  the workspace dir, but every head command in each pipeline segment has to
  match an explicit whitelist and `>` / `>>` redirects are blocked. This is
  the cheap local substitute the sweqa runner has always used against cloned
  git repos.

Both expose the same `.exec(command, cwd) -> str` interface.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Protocol


class Sandbox(Protocol):
    def exec(self, command: str, cwd: str) -> str: ...


@dataclass
class DockerSandbox:
    image: str
    timeout: int = 30
    truncate_at: int = 12000
    truncate_head: int = 6000
    truncate_tail: int = 3000
    pull_timeout: int = 60

    _pulled: bool = field(default=False, init=False, repr=False)
    _pull_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _ensure_image(self) -> None:
        if self._pulled:
            return
        with self._pull_lock:
            if self._pulled:
                return
            subprocess.run(
                ["docker", "pull", self.image],
                capture_output=True, timeout=self.pull_timeout,
            )
            self._pulled = True

    def exec(self, command: str, cwd: str) -> str:
        self._ensure_image()
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm", "--network=none",
                    "-v", f"{cwd}:/data:ro", "-w", "/data",
                    self.image,
                    "sh", "-c", command,
                ],
                capture_output=True, text=True, timeout=self.timeout + 5,
            )
        except subprocess.TimeoutExpired:
            return "(command timed out)"
        output = result.stdout
        if result.returncode != 0 and result.stderr:
            output += f"\n[stderr] {result.stderr}"
        if len(output) > self.truncate_at:
            output = (output[:self.truncate_head]
                      + "\n... [truncated] ...\n"
                      + output[-self.truncate_tail:])
        return output.strip() or "(no output)"


_REDIRECT_RE = re.compile(r"[^12&]>(?!&)")


@dataclass
class WhitelistSandbox:
    allowed_commands: frozenset[str]
    timeout: int = 30
    truncate_at: int = 8000
    truncate_head: int = 4000
    truncate_tail: int = 2000

    def _check_whitelist(self, command: str) -> str | None:
        """Return an error message if any head command isn't in the allow-list."""
        parts = re.split(r"\|+|&&|;", command)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            tokens = part.split()
            cmd_idx = 0
            for i, tok in enumerate(tokens):
                if "=" in tok and not tok.startswith("-"):
                    continue
                cmd_idx = i
                break
            if cmd_idx >= len(tokens):
                continue
            cmd_name = os.path.basename(tokens[cmd_idx])
            if cmd_name not in self.allowed_commands:
                return (f"(blocked: '{cmd_name}' not in read-only whitelist. "
                        f"Allowed: cat, grep, find, ls, head, tail, awk, sed, wc, sort, ...)")
        return None

    def exec(self, command: str, cwd: str) -> str:
        if _REDIRECT_RE.search(command) or ">>" in command:
            return "(blocked: output redirection not allowed in read-only mode)"
        blocked = self._check_whitelist(command)
        if blocked:
            return blocked
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return "(command timed out)"

        output = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        if result.returncode != 0 and stderr:
            output += f"\n[stderr] {stderr}"
        if len(output) > self.truncate_at:
            output = (output[:self.truncate_head]
                      + "\n... [truncated] ...\n"
                      + output[-self.truncate_tail:])
        return output.strip() or "(no output)"
