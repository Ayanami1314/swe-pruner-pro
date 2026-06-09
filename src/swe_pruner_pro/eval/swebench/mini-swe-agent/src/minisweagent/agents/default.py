"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from jinja2 import StrictUndefined, Template

from minisweagent import Environment, Model

from typing import Any
from minisweagent.utils.pruner import PrunerClient, PrunerConfig, PruneResponse
from minisweagent.utils.prune_hooks import PruneContext, PrunePostContext, run_post_hooks, run_pre_hooks
from minisweagent.run.utils.save import save_traj


def _resolve_env_placeholders(value: Any):
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.getenv(key, "")
    if isinstance(value, dict):
        return {k: _resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(v) for v in value]
    return value

from minisweagent.utils.log import logger

_KEEP_ALL_MARKER = "KEEP_ALL_IN_THIS_COMMAND"
# In-image tmp fix: strip the line range from "(filtered N lines: X-Y)" markers
# so they don't visually clash with `cat -n`-style line numbers in the output.
_FILTER_RANGE_RE = re.compile(r"\(filtered (\d+) lines:\s*\d+-\d+\)")
_CFQ_RE = re.compile(r'<context_focus_question>(.*?)</context_focus_question>',
                     re.DOTALL)

ABLATION_PROMPT_SUFFIX = """

## Ablation: Context Focus Question
Every response MUST include a `<context_focus_question>...</context_focus_question>` \
XML tag BEFORE the bash block, holding a short natural-language phrase describing \
what specific information you are trying to extract from this command's output. \
Examples:
  - <context_focus_question>definition of class FooBar</context_focus_question>
  - <context_focus_question>where is X imported and used</context_focus_question>
  - <context_focus_question>callers of method baz()</context_focus_question>

The pruning module uses this question to keep only the relevant lines in the \
command output. A vague or missing focus question disables pruning for that turn.
"""

@dataclass
class AgentConfig:
    # The default settings are the bare minimum to run the agent. Take a look at the config files for improved settings.
    system_template: str = "You are a helpful assistant that can do anything."
    instance_template: str = (
        "Your task: {{task}}. Please reply with a single shell command in triple backticks. "
        "To finish, the first line of the output of the shell command must be 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'."
    )
    timeout_template: str = (
        "The last command <command>{{action['action']}}</command> timed out and has been killed.\n"
        "The output of the command was:\n <output>\n{{output}}\n</output>\n"
        "Please try another command and make sure to avoid those requiring interactive input."
    )
    format_error_template: str = "Please always provide EXACTLY ONE action in triple backticks."
    action_observation_template: str = "Observation: {{output}}"
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    step_limit: int = 0
    cost_limit: float = 3.0
    pruner: dict[str, Any] | None = None
    # If set, the current trajectory is written here after every message.
    # Makes partial trajectories survive SIGKILL / timeout / crash.
    trajectory_path: str | None = None


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.pruner_client: PrunerClient | None = None
        if self.config.pruner:
            resolved = _resolve_env_placeholders(self.config.pruner)
            # Skip pruner if URL is empty (baseline mode)
            if resolved.get("url"):
                print(f"Using Pruner Config: {resolved}")
                pruner_cfg = PrunerConfig(**{k: v for k, v in resolved.items()})
                print(f"Loaded Pruner Config: {pruner_cfg}")
                self.pruner_client = PrunerClient(pruner_cfg)
            else:
                print("Pruner URL not set, running without pruner")

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = asdict(self.config) | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, **kwargs})
        self._save_partial_trajectory()

    def _save_partial_trajectory(self) -> None:
        """Write the current trajectory to disk after every message.

        Uses the same schema as ``save_traj`` (minisweagent.run.utils.save) so
        an interrupted run leaves a usable trajectory behind. Failures here
        are swallowed — partial save is best-effort and must never break the
        agent loop.
        """
        path_str = getattr(self.config, "trajectory_path", None)
        if not path_str:
            return
        try:
            save_traj(self, Path(path_str), print_path=False, exit_status="in_progress", result=None)
        except Exception:
            pass

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        system_text = self.render_template(self.config.system_template)
        if self.pruner_client and self.pruner_client.config.backend:
            system_text = system_text + ABLATION_PROMPT_SUFFIX
        self.add_message("system", system_text)
        self.add_message("user", self.render_template(self.config.instance_template))
        unparsed_err_cnt = 0
        while True:
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            except Exception as e:
                self.add_message("user", f"Error: {e}")
                unparsed_err_cnt += 1
                if unparsed_err_cnt >= 3:
                    return "Error", f"Unparsed error occurred {unparsed_err_cnt} times: {e}"

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        message_kwargs: dict[str, Any] = {}
        if output.get("pruned_stats"):
            message_kwargs["pruned_stats"] = output["pruned_stats"]
        self.add_message("user", observation, **message_kwargs)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message."""
        content = response["content"]
        actions = re.findall(self.config.action_regex, content, re.DOTALL)
        if len(actions) != 1:
            raise FormatError(self.render_template(self.config.format_error_template, actions=actions))
        return {"action": actions[0].strip(), **response}

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except subprocess.TimeoutExpired as e:
            output = e.output.decode("utf-8", errors="replace") if e.output else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        except TimeoutError:
            raise ExecutionTimeoutError(self.render_template(self.config.timeout_template, action=action, output=""))
        self.has_finished(output)
        self._apply_pruner(action, output)
        return output

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
            raise Submitted("".join(lines[1:]))

    def _convert_messages_to_chat_format(self) -> list[dict]:
        """Convert mini-swe-agent messages to OpenAI chat format for Phase 2 pruner.

        Mini-swe-agent format:
          - system/user/assistant roles with content string
          - assistant messages contain bash commands in markdown code blocks
          - user messages after an assistant with a valid bash action are
            tool observations (rendered via ``action_observation_template``,
            which can be customized — don't rely on a content prefix like
            "Observation: ").

        Converted to:
          - system/user → keep role+content
          - assistant with bash command → add tool_calls array (no content)
          - user that follows a converted assistant-with-tool_calls → role=tool
        """
        chat_messages: list[dict] = []

        for i, msg in enumerate(self.messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                chat_messages.append({"role": "system", "content": content})
            elif role == "assistant":
                actions = re.findall(self.config.action_regex, content, re.DOTALL)
                if actions:
                    command = actions[0].strip()
                    chat_messages.append({
                        "role": "assistant",
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": {"command": command},
                            },
                        }],
                    })
                else:
                    chat_messages.append({"role": "assistant", "content": content})
            elif role == "user":
                # It's a tool observation iff the previous *converted* message
                # is an assistant with tool_calls. This catches observations
                # regardless of the configured action_observation_template.
                # Format-error replies (NonTerminatingException in run loop)
                # follow an assistant WITHOUT a valid bash block, so the
                # preceding converted message has content instead of
                # tool_calls and we correctly keep role=user.
                prev = chat_messages[-1] if chat_messages else None
                if prev and prev.get("role") == "assistant" and prev.get("tool_calls"):
                    chat_messages.append({"role": "tool", "content": content})
                else:
                    chat_messages.append({"role": "user", "content": content})

        return chat_messages

    def _get_output_threshold(self, action: dict) -> float | None:
        """Return pruner threshold for this action.

        If the command contains the KEEP_ALL_IN_THIS_COMMAND marker, the agent
        is opting out of pruning for this call (returns None). Otherwise the
        configured default is used.
        """
        cmd = action.get("action", "") if isinstance(action, dict) else ""
        if _KEEP_ALL_MARKER in cmd:
            return None
        return self.pruner_client.config.threshold if self.pruner_client else None

    def _get_context_focus_question(self) -> str:
        """Extract <context_focus_question>...</context_focus_question> from the
        last assistant message. Used by ablation backends (llmlingua2, rerank,
        ...) as the relevance query. Empty when missing — server passthroughs."""
        for msg in reversed(self.messages):
            if msg.get("role") == "assistant":
                m = _CFQ_RE.search(msg.get("content", ""))
                if m:
                    return m.group(1).strip()
                break
        return ""

    def _apply_pruner(self, action: dict, output: dict[str, str]) -> None:
        if not self.pruner_client:
            return
        text = output.get("output")
        if not text:
            return

        threshold = self._get_output_threshold(action)
        if threshold is None:
            return

        # Convert message history to OpenAI chat format for Phase 2 pruner
        history = self._convert_messages_to_chat_format()
        cfq = self._get_context_focus_question()
        tool_call = {"name": "bash", "arguments": {"command": action.get("action", "")}}
        if cfq:
            tool_call["arguments"]["context_focus_question"] = cfq

        ctx = PruneContext(
            messages=history,
            tool_call=tool_call,
            tool_response=text,
        )
        decision = run_pre_hooks(ctx)
        if decision and decision.skip:
            output["pruned_stats"] = {
                "skipped": True,
                "skip_reason": decision.reason,
                "skip_metadata": decision.metadata,
            }
            return

        try:
            pruned_result: PruneResponse = self.pruner_client.prune(
                history=history,
                tool_call=tool_call,
                tool_response=text,
                threshold=threshold,
                context_focus_question=cfq,
            )
        except Exception as exc:
            logger.error("Pruner request FAILED (using raw output): %s", exc)
            output["pruned_stats"] = {"error": str(exc)}
            return

        if pruned_result.pruned_chars < pruned_result.original_chars:
            stripped_pruned_code = _FILTER_RANGE_RE.sub(r"(filtered \1 lines)", pruned_result.pruned_code)
            post_ctx = PrunePostContext(
                ctx=ctx,
                pruned_code=stripped_pruned_code,
                original_chars=pruned_result.original_chars,
                pruned_chars=pruned_result.pruned_chars,
                original_lines=pruned_result.original_lines,
                kept_line_count=pruned_result.kept_line_count,
            )
            output["output"] = run_post_hooks(post_ctx)

        output["pruned_stats"] = {
            "original_chars": pruned_result.original_chars,
            "pruned_chars": pruned_result.pruned_chars,
            "original_lines": pruned_result.original_lines,
            "kept_line_count": pruned_result.kept_line_count,
            "latency_ms": pruned_result.latency_ms,
        }