"""Smoke tests for the installed CLI, without loading a model or checkpoint."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import unittest


class PrunerCLITest(unittest.TestCase):
    def run_cli(self, *args: str, module: bool = False) -> subprocess.CompletedProcess:
        if module:
            command = [sys.executable, "-m", "swe_pruner_pro.serving.pruner_server"]
        else:
            name = "swe-pruner-server.exe" if os.name == "nt" else "swe-pruner-server"
            command = [str(Path(sysconfig.get_path("scripts")) / name)]
        env = {key: value for key, value in os.environ.items() if not key.startswith("PRUNER_")}
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_COLOR="1", TERM="dumb")
        return subprocess.run(
            [*command, *args], capture_output=True, text=True, env=env, timeout=60,
        )

    def assert_help(self, result: subprocess.CompletedProcess) -> None:
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for option in ("--checkpoint", "--backbone", "--sglang-url", "--port"):
            self.assertIn(option, result.stdout)

    def test_installed_console_help(self) -> None:
        self.assert_help(self.run_cli("--help"))

    def test_module_help(self) -> None:
        self.assert_help(self.run_cli("--help", module=True))

    def test_installed_console_validates_arguments(self) -> None:
        result = self.run_cli("--checkpoint", "unused-checkpoint")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("--backbone is required", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
