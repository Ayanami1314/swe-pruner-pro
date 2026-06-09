"""SWE-Pruner reference baseline (forwards to an external service running
the GitHub-published swe-pruner Phase-1 implementation).

Architecture:
  * The reference repo (github.com/Ayanami1314/swe-pruner) ships a
    compression-head HTTP service. Either pre-launch it externally and
    pass ``ref_url`` (or set ``SWE_PRUNER_REF_URL``), or let this class
    spawn it as a subprocess.
  * Configuration via env vars or constructor args:
      SWE_PRUNER_REF_DIR       — checkout location (no default; required for subprocess mode)
      SWE_PRUNER_REF_MODEL     — model path (no default)
      SWE_PRUNER_REF_DEVICE    — cuda:N for the ref service
      SWE_PRUNER_REF_PORT      — port the ref service listens on
      SWE_PRUNER_REF_URL       — pre-launched URL (skip subprocess if set)
      SWE_PRUNER_REF_LOG       — subprocess log file (default /tmp/swe-pruner-ref.log)
      SWE_PRUNER_REF_LAUNCH_CMD — full launch command (override default)
  * On failure, returns passthrough + ``error_msg`` (NEVER falls back to
    another baseline, per the paper's protocol).
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

import requests

from .types import BaselineResult

logger = logging.getLogger(__name__)


class SWEPrunerBackend:
    name = "swe_pruner"

    def __init__(
        self,
        ref_url: str | None = None,
        ref_dir: str | None = None,
        model_path: str | None = None,
        device: str = "cuda:0",
        port: int = 8101,
        boot_timeout_s: int = 180,
        request_timeout_s: float = 120.0,
    ):
        self.model_path = model_path or os.environ.get("SWE_PRUNER_REF_MODEL", "")
        self.device = device
        self.port = port
        self.request_timeout_s = request_timeout_s
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        env_url = ref_url or os.environ.get("SWE_PRUNER_REF_URL")
        if env_url:
            self.ref_url = env_url.rstrip("/")
            self._proc: subprocess.Popen | None = None
            logger.info(f"[swe_pruner] using pre-launched ref service at {self.ref_url}")
            return

        ref_dir_str = ref_dir or os.environ.get("SWE_PRUNER_REF_DIR")
        if not ref_dir_str:
            raise RuntimeError(
                "swe_pruner: set SWE_PRUNER_REF_URL to a running service, "
                "or SWE_PRUNER_REF_DIR to a local clone of github.com/Ayanami1314/swe-pruner"
            )
        self.ref_dir = Path(ref_dir_str)
        if not self.ref_dir.exists():
            raise RuntimeError(
                f"swe-pruner reference checkout not found at {self.ref_dir}"
            )
        if not self.model_path:
            raise RuntimeError(
                "swe_pruner: set SWE_PRUNER_REF_MODEL (or pass model_path)"
            )

        self.ref_url = f"http://127.0.0.1:{self.port}"
        self._proc = self._launch_subprocess()
        if not self._wait_for_health(boot_timeout_s):
            raise RuntimeError(
                f"swe_pruner ref service failed to come up at {self.ref_url} "
                f"within {boot_timeout_s}s"
            )
        logger.info(f"[swe_pruner] ref service healthy at {self.ref_url}")

    def _launch_subprocess(self) -> subprocess.Popen:
        # Default cmd invokes the ref repo's Typer CLI module directly.
        # Override SWE_PRUNER_REF_LAUNCH_CMD to use a different entrypoint.
        cmd_str = os.environ.get(
            "SWE_PRUNER_REF_LAUNCH_CMD",
            f"python -m swe_pruner.online_serving "
            f"--model-path {shlex.quote(self.model_path)} "
            f"--port {self.port} --host 127.0.0.1",
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.device.split(":")[-1]
        log_path = Path(os.environ.get("SWE_PRUNER_REF_LOG", "/tmp/swe-pruner-ref.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "ab", buffering=0)
        logger.info(f"[swe_pruner] launching subprocess: {cmd_str} "
                    f"(cwd={self.ref_dir}, CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
        return subprocess.Popen(
            shlex.split(cmd_str),
            cwd=str(self.ref_dir),
            env=env,
            stdout=log_f, stderr=log_f,
            close_fds=True,
        )

    def _wait_for_health(self, timeout_s: int) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                logger.error(f"[swe_pruner] subprocess exited "
                             f"(rc={self._proc.returncode}) before becoming healthy")
                return False
            try:
                r = self.session.get(f"{self.ref_url}/health", timeout=5)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def _invoke_ref_service(
        self, history: list[dict], tool_call: dict,
        tool_response: str, threshold: float, query: str,
    ) -> dict:
        # Reference repo's /prune is the original Phase-1 API: takes flat
        # (query, code) — no history / tool_call. We feed our (history,
        # tool_call, tool_response) into that schema by passing
        # context_focus_question as ``query`` and tool_response as ``code``.
        payload = {
            "query": query,
            "code": tool_response,
            "threshold": threshold,
        }
        r = self.session.post(
            f"{self.ref_url}/prune", json=payload,
            timeout=self.request_timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def prune(
        self,
        *,
        history: list[dict],
        tool_call: dict,
        tool_response: str,
        threshold: float,
        query: str,
    ) -> BaselineResult:
        t0 = time.time()
        # Phase-1 API requires non-empty query — passthrough rather than 422.
        if not query or not query.strip():
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg="swe_pruner: empty query (agent omitted context_focus_question)",
            )
        try:
            data = self._invoke_ref_service(history, tool_call,
                                            tool_response, threshold, query)
            pruned = data.get("pruned_code")
            if not isinstance(pruned, str):
                raise ValueError(f"ref service returned no 'pruned_code' "
                                 f"(keys={list(data.keys())})")
            return BaselineResult(
                pruned_code=pruned,
                kept_lines=list(data.get("kept_frags") or []),
                original_lines=len(tool_response.splitlines()),
                kept_line_count=len(pruned.splitlines()),
                original_chars=len(tool_response),
                pruned_chars=len(pruned),
                latency_ms=(time.time() - t0) * 1000,
                error_msg=data.get("error_msg"),
            )
        except Exception as exc:
            logger.exception("[swe_pruner] prune failed")
            return BaselineResult.passthrough(
                tool_response,
                latency_ms=(time.time() - t0) * 1000,
                error_msg=f"swe_pruner: {type(exc).__name__}: {exc}",
            )
