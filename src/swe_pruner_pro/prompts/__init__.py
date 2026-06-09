"""Prompt templates used across SWE-Pruner Pro."""
from __future__ import annotations

from swe_pruner_pro.prompts.label import LABEL_TEMPLATE, SYSTEM_PROMPT
from swe_pruner_pro.prompts.ablation_judge import JUDGE_SYSTEM
from swe_pruner_pro.prompts.sweqa_judge import SCORING_PROMPT, JUDGE_DIMS

__all__ = [
    "LABEL_TEMPLATE",
    "SYSTEM_PROMPT",
    "JUDGE_SYSTEM",
    "SCORING_PROMPT",
    "JUDGE_DIMS",
]
