"""Shared runtime for the multi-turn agent benchmarks (SWE-QA, Oolong).

Public surface mirrors the upstream ``eval_lib`` package:

* Agent loop: ``run_agent_loop``
* Sandboxes: ``Sandbox``, ``DockerSandbox``, ``WhitelistSandbox``
* Pruner HTTP client: ``call_pruner``, ``sanitize_history_for_prune``
* Prune hooks (pre / post): ``PruneContext``, ``PrunePostContext``,
  ``run_pre_hooks``, ``run_post_hooks``, ``DEFAULT_PRE_HOOKS``,
  ``DEFAULT_POST_HOOKS``, ``hook_repeat_read``, ``hook_command_whitelist``,
  ``hook_early_history``, ``hook_fully_pruned_hint``, ``resolve_pre_hooks_spec``
* Tool-call XML fallback parser (mimo): ``parse_reasoning_tool_calls``,
  ``maybe_mimo_parser``
* Sampling/system-prefix helpers: ``sampling_params``, ``system_prefix``
* Chat retry wrapper: ``chat_completion_with_retry``
* Resume helpers: ``load_done_ids``, ``SafeJsonlWriter``
* Stats: ``RunStats``
* Ablation-mode helpers: ``ABLATION_BACKENDS``, ``ABLATION_PROMPT_SUFFIX``,
  ``derive_query``, ``wrap_bash_tool_with_focus_question``
"""
from .ablation_tool import (
    ABLATION_BACKENDS,
    ABLATION_PROMPT_SUFFIX,
    derive_query,
    wrap_bash_tool_with_focus_question,
)
from .agent_loop import run_agent_loop
from .chat import chat_completion_with_retry
from .mimo import maybe_mimo_parser, parse_reasoning_tool_calls
from .models import sampling_params, system_prefix
from .prune_hooks import (
    DEFAULT_POST_HOOKS,
    DEFAULT_PRE_HOOKS,
    PruneContext,
    PruneDecision,
    PrunePostContext,
    hook_command_whitelist,
    hook_early_history,
    hook_fully_pruned_hint,
    hook_repeat_read,
    resolve_pre_hooks_spec,
    run_post_hooks,
    run_pre_hooks,
)
from .pruner_client import call_pruner, sanitize_history_for_prune
from .resume import SafeJsonlWriter, load_done_ids
from .sandbox import DockerSandbox, Sandbox, WhitelistSandbox
from .stats import RunStats

__all__ = [
    "ABLATION_BACKENDS",
    "ABLATION_PROMPT_SUFFIX",
    "derive_query",
    "wrap_bash_tool_with_focus_question",
    "run_agent_loop",
    "chat_completion_with_retry",
    "maybe_mimo_parser",
    "parse_reasoning_tool_calls",
    "sampling_params",
    "system_prefix",
    "PruneContext",
    "PruneDecision",
    "PrunePostContext",
    "DEFAULT_PRE_HOOKS",
    "DEFAULT_POST_HOOKS",
    "hook_repeat_read",
    "hook_command_whitelist",
    "hook_early_history",
    "hook_fully_pruned_hint",
    "run_pre_hooks",
    "run_post_hooks",
    "resolve_pre_hooks_spec",
    "call_pruner",
    "sanitize_history_for_prune",
    "load_done_ids",
    "SafeJsonlWriter",
    "Sandbox",
    "DockerSandbox",
    "WhitelistSandbox",
    "RunStats",
]
