"""LLM-as-judge for SWE-QA family answers.

Calls an OpenAI-compatible chat endpoint (default: OpenRouter
``openai/gpt-5.4-mini``) to score each candidate answer against a
reference on five dimensions (1-10 each). The scoring prompt itself
lives in :mod:`swe_pruner_pro.prompts.sweqa_judge`.

API key resolution: ``OPENROUTER_API_KEY`` first, falling back to
``OPENAI_API_KEY``. Override the endpoint with ``--base-url`` /
``OPENAI_BASE_URL``.

Three subcommands:

- ``evaluate``  — score a single ``*_answers.jsonl`` file.
- ``judge-all`` — walk a results dir and score every ``*_answers.jsonl``
  found under it (resumes via written ``*_scores.jsonl``).
- ``summary``   — print a markdown table of per-experiment means.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
from typing import Optional

import typer
from openai import OpenAI

from ...prompts.sweqa_judge import JUDGE_DIMS, SCORING_PROMPT

app = typer.Typer(help="LLM-as-a-Judge for SWE-QA family evaluation")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-5.4-mini"


def _make_client(base_url: str = "", api_key: str = "") -> OpenAI:
    base = base_url or os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    key = (api_key
           or os.environ.get("OPENROUTER_API_KEY")
           or os.environ.get("OPENAI_API_KEY", ""))
    if not key:
        raise RuntimeError(
            "No API key found — set OPENROUTER_API_KEY or OPENAI_API_KEY, "
            "or pass --api-key."
        )
    return OpenAI(base_url=base, api_key=key)


def score_answer(
    question: str, reference: str, candidate: str,
    client: OpenAI, model: str,
) -> Optional[dict]:
    if not reference or not candidate or candidate.strip() == "No answer found":
        return None
    prompt = SCORING_PROMPT.format(question=question, reference=reference, candidate=candidate)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        scores = json.loads(text.strip())
        for k in JUDGE_DIMS:
            if k not in scores or not (1 <= int(scores[k]) <= 10):
                return None
        return {k: int(scores[k]) for k in JUDGE_DIMS}
    except Exception as e:
        print(f"  [error] {e}")
        return None


def _judge_one(record: dict, ref_dict: dict, client: OpenAI, model: str) -> Optional[dict]:
    question = record.get("question", "")
    candidate = record.get("answer", "")
    reference = ref_dict.get(question, record.get("ground_truth", ""))
    scores = score_answer(question, reference, candidate, client, model)
    if scores is None:
        return None
    total = sum(scores.values())
    return {
        "id": record.get("id", ""),
        "question": question,
        "experiment": record.get("experiment", ""),
        "scores": scores,
        "total": total,
        "mean": round(total / 5, 2),
    }


def _load_done_ids(scores_path: Path) -> set:
    done = set()
    if scores_path.exists():
        for line in open(scores_path):
            try:
                done.add(json.loads(line.strip()).get("id", ""))
            except json.JSONDecodeError:
                continue
    return done


@app.command()
def evaluate(
    candidate: str = typer.Option(..., "-c", "--candidate", help="Candidate answers JSONL"),
    output: str = typer.Option("", "-o", "--output", help="Output scores JSONL (default: <candidate>_scores.jsonl)"),
    reference: str = typer.Option("", "-r", "--reference", help="Reference JSONL (default: ground_truth field of each candidate row)"),
    model: str = typer.Option(DEFAULT_MODEL, "-m", "--model"),
    workers: int = typer.Option(32, "-w", "--workers"),
    base_url: str = typer.Option("", "--base-url", help="Override OpenAI-compatible base URL"),
    api_key: str = typer.Option("", "--api-key", help="API key (else OPENROUTER_API_KEY / OPENAI_API_KEY)"),
):
    """Judge a single candidate answers file."""
    client = _make_client(base_url, api_key)
    ref_dict: dict = {}
    if reference and Path(reference).exists():
        for line in open(reference):
            d = json.loads(line.strip())
            if d.get("question") and (d.get("answer") or d.get("ground_truth")):
                ref_dict[d["question"]] = d.get("answer") or d.get("ground_truth")
    records = [json.loads(l) for l in open(candidate) if l.strip()]
    print(f"Judging {len(records)} answers with {model} ({workers} workers)...")

    out_path = Path(output or candidate.replace(".jsonl", "_scores.jsonl"))
    done_ids = _load_done_ids(out_path)
    remaining = [r for r in records if r.get("id", "") not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} already done, {len(remaining)} remaining")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_judge_one, r, ref_dict, client, model): r for r in remaining}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if out_path.exists():
        means = [json.loads(l)["mean"] for l in open(out_path) if l.strip()]
        if means:
            print(f"Done: {len(means)} scored, mean={sum(means) / len(means):.2f}")


@app.command("judge-all")
def judge_all(
    results_dir: str = typer.Option("results", "-d", "--results-dir"),
    model: str = typer.Option(DEFAULT_MODEL, "-m", "--model"),
    workers: int = typer.Option(8, "-w", "--workers"),
    base_url: str = typer.Option("", "--base-url"),
    api_key: str = typer.Option("", "--api-key"),
):
    """Judge all experiment results under results_dir."""
    client = _make_client(base_url, api_key)
    rdir = Path(results_dir)

    for exp_dir in sorted(rdir.iterdir()):
        if not exp_dir.is_dir():
            continue
        for ans_file in sorted(exp_dir.glob("*_answers.jsonl")):
            scores_file = ans_file.with_name(ans_file.stem.replace("_answers", "_scores") + ".jsonl")
            records = [json.loads(l) for l in open(ans_file) if l.strip()]
            done_ids = _load_done_ids(scores_file)
            remaining = [r for r in records if r.get("id") not in done_ids]
            if not remaining:
                print(f"[skip] {exp_dir.name}/{ans_file.name}: all {len(done_ids)} done")
                continue
            print(f"[judge] {exp_dir.name}/{ans_file.name}: {len(remaining)} remaining ({len(done_ids)} done)")
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_judge_one, r, {}, client, model): r for r in remaining}
                for fut in concurrent.futures.as_completed(futures):
                    res = fut.result()
                    if res:
                        results.append(res)
            with open(scores_file, "a") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            means = [json.loads(l)["mean"] for l in open(scores_file) if l.strip()]
            if means:
                print(f"  -> {len(means)} scored, mean={sum(means) / len(means):.2f}")


@app.command()
def summary(
    results_dir: str = typer.Option("results", "-d", "--results-dir"),
):
    """Print summary table of all judged experiments."""
    rdir = Path(results_dir)
    rows = []
    for exp_dir in sorted(rdir.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_scores: list[float] = []
        exp_details = {k: [] for k in JUDGE_DIMS}
        for scores_file in sorted(exp_dir.glob("*_scores.jsonl")):
            for line in open(scores_file):
                d = json.loads(line.strip())
                exp_scores.append(d["mean"])
                for k in exp_details:
                    exp_details[k].append(d["scores"][k])
        if not exp_scores:
            continue
        n = len(exp_scores)
        avg = sum(exp_scores) / n
        detail_avgs = {k: sum(v) / len(v) for k, v in exp_details.items() if v}
        good = sum(1 for s in exp_scores if s >= 7) / n * 100
        bad = sum(1 for s in exp_scores if s <= 2) / n * 100
        rows.append((exp_dir.name, n, avg, good, bad, detail_avgs))

    if not rows:
        print("No scores found")
        return
    print(f"\n{'Experiment':<50s} {'N':>4s} {'Mean':>5s} {'Good%':>6s} {'Bad%':>5s}  Corr  Comp  Relv  Clar  Reas")
    print("-" * 110)
    for name, n, avg, good, bad, details in sorted(rows, key=lambda x: -x[2]):
        print(f"{name:<50s} {n:>4d} {avg:>5.2f} {good:>5.1f}% {bad:>4.1f}%"
              f"  {details.get('correctness', 0):4.1f}  {details.get('completeness', 0):4.1f}"
              f"  {details.get('relevance', 0):4.1f}  {details.get('clarity', 0):4.1f}"
              f"  {details.get('reasoning', 0):4.1f}")


if __name__ == "__main__":
    app()
