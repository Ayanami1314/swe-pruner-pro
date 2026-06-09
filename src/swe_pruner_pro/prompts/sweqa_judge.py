"""SWE-QA family answer-quality judge prompt.

Used by GPT-5.4-mini to score the agent's final answer against a reference
answer on a 1-10 rubric across five dimensions (correctness, completeness,
relevance, clarity, reasoning). Verbatim from paper appendix ``app:prompts``
("SWE-QA family answer-quality judge prompt").
"""
from __future__ import annotations

SCORING_PROMPT = """\
You are an expert evaluator. Compare a candidate answer against a reference answer for a code repository question.

Question: {question}

Reference Answer:
{reference}

Candidate Answer:
{candidate}

Score the candidate on these 5 dimensions (1-10 each):
1. correctness: Are the core facts and details accurate?
2. completeness: Does it cover all key points from the reference?
3. relevance: Is it focused on the question without irrelevant information?
4. clarity: Is the language clear and precise?
5. reasoning: Is the reasoning logical and well-structured?

Respond with ONLY a JSON object (no explanation):
{{"correctness": N, "completeness": N, "relevance": N, "clarity": N, "reasoning": N}}"""

JUDGE_DIMS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


def format(*, question: str, reference: str, candidate: str) -> str:
    return SCORING_PROMPT.format(question=question, reference=reference, candidate=candidate)
