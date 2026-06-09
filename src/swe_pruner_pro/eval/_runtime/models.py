"""Per-model sampling params and system-prompt prefixes.

Different model families have published "recommended" sampling configurations
that materially change quality. The old code just hardcoded `temperature=0.2`
everywhere, which was noticeably off for Qwen3 (official rec: T=0.7, top_p=0.8,
top_k=20, rep_penalty=1.05) and mimo (T=0.3, top_p=0.95, thinking mode on).

This module centralizes those:

* ``sampling_params(model)`` returns a dict suitable for ``**kwargs`` into
  ``chat.completions.create``. Contains native OpenAI fields (``temperature``,
  ``top_p``) plus an ``extra_body`` entry for non-OpenAI params (``top_k``,
  ``repetition_penalty``, ``chat_template_kwargs``).

* ``system_prefix(model, date=None)`` returns the provider-specific system
  prologue (empty for most models). MiMo's serving rec is to prepend
  ``"You are MiMo, an AI assistant developed by Xiaomi. Today's date: ...
  Your knowledge cutoff date is December 2024."`` before the task system
  prompt; omitting it degrades answer quality noticeably.
"""

from __future__ import annotations

import datetime as _dt


def _model_family(model: str | None) -> str:
    if not model:
        return "generic"
    m = model.lower().lstrip("/").split("/")[-1]
    if m.startswith("mimo"):
        return "mimo"
    if m.startswith("qwen3") or "qwen3-coder" in m or "qwen2.5" in m:
        return "qwen3"
    return "generic"


_SAMPLING: dict[str, dict] = {
    "mimo": {
        "temperature": 0.3,
        "top_p": 0.95,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    },
    # Qwen3-family recommended: T=0.7, top_p=0.8, top_k=20, rep_penalty=1.05
    "qwen3": {
        "temperature": 0.7,
        "top_p": 0.8,
        "extra_body": {
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    },
    # Unknown/default: leave server defaults; caller's explicit `temperature`
    # (if any) still wins via merge order.
    "generic": {},
}


def sampling_params(model: str | None) -> dict:
    """Return chat.completions sampling kwargs for the given model. Safe to
    pass straight into ``client.chat.completions.create(**kwargs, ...)``.
    Callers that set `temperature` explicitly can still override by passing
    their value after ``**sampling_params(model)``."""
    return {k: (v.copy() if isinstance(v, dict) else v)
            for k, v in _SAMPLING.get(_model_family(model), {}).items()}


def system_prefix(model: str | None, date: _dt.date | None = None) -> str:
    """Model-specific system-prompt prologue. Empty string for models that
    don't require one. Caller concatenates this to the task system prompt
    (newline-separated) before the API call."""
    if _model_family(model) != "mimo":
        return ""
    d = date or _dt.date.today()
    return (
        f"You are MiMo, an AI assistant developed by Xiaomi.\n\n"
        f"Today's date: {d.isoformat()} {d.strftime('%A')}. "
        f"Your knowledge cutoff date is December 2024."
    )
