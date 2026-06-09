"""Oolong agent-based eval (multi-turn re-cast in Docker sandbox).

Agent loop:
  1. Pre-generated workspace (containing ``data.txt``) mounted read-only in docker.
  2. Agent uses the ``bash`` tool to explore the workspace.
  3. With ``--pruner-url`` (or ``--baseline`` for client-side baselines):
     whenever a tool_response exceeds ``--prune-min-chars``, the response is
     compressed before the LLM sees it.
  4. Scoring uses Oolong's official ``synth_process_response`` /
     ``dnd_process_response`` (you must install the upstream ``oolong``
     package — see README).
"""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import jsonlines
import openai
import typer
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm

from .._runtime import (
    DockerSandbox,
    derive_query,
    maybe_mimo_parser,
    resolve_pre_hooks_spec,
    run_agent_loop,
    wrap_bash_tool_with_focus_question,
)
from ..baselines import BaselinePruner

app = typer.Typer(help="Oolong agent-based eval (baseline / pruner / ablation)")

WORKSPACES_DIR = Path(os.environ.get("OOLONG_WORKSPACES_DIR", "workspaces"))

_DOCKER_IMAGE = os.getenv("OOLONG_DOCKER_IMAGE", "python:3.12-slim")

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the workspace directory. Use grep/awk/sort/uniq/wc/etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"},
            },
            "required": ["command"],
        },
    },
}

SYSTEM_PROMPT_BASE = """\
You are a data analyst agent. You can run bash commands (via the bash tool) on a read-only directory containing data.txt.
Use grep, awk, sort, uniq -c, wc, etc. to explore and answer the question precisely.

IMPORTANT: When you have the answer, state it clearly as "Answer: <your answer>".
For counts give exact numbers. For labels give exact text. Do NOT re-verify an answer you already have."""

FILTER_SUFFIX = """

Long command outputs are automatically filtered to show the most relevant lines. \
Filtered sections appear as "(filtered N lines)" markers showing how many lines were removed. \
This saves context window space — you do NOT need to worry about managing output length. \
To skip filtering for a single command, prepend `echo "KEEP_ALL_IN_THIS_COMMAND" && ` to your \
command to receive its full unfiltered output."""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + FILTER_SUFFIX


def _build_baseline(name: str, *, sglang_url: str = "", device: str = "cuda:0") -> BaselinePruner:
    if name == "llmlingua2":
        from ..baselines.llmlingua2 import LLMLingua2Pruner
        return LLMLingua2Pruner(device=device)
    if name == "selective_context":
        from ..baselines.selective_context import SelectiveContextPruner
        return SelectiveContextPruner(device=device)
    if name == "rerank":
        from ..baselines.rerank import RerankPruner
        return RerankPruner(device=device)
    if name == "self_prune":
        from ..baselines.self_prune import SelfPrunePruner
        if not sglang_url:
            raise typer.BadParameter("--baseline self_prune requires --self-prune-url")
        return SelfPrunePruner(sglang_url=sglang_url)
    if name == "longcodezip":
        from ..baselines.longcodezip import LongCodeZipPruner
        return LongCodeZipPruner(device=device)
    if name == "swe_pruner":
        from ..baselines.swe_pruner import SWEPrunerBackend
        return SWEPrunerBackend(device=device)
    raise typer.BadParameter(f"unknown --baseline '{name}'")


def _get_workspace(datapoint: dict) -> str:
    ws_dir = WORKSPACES_DIR / datapoint["id"]
    ws_dir.mkdir(parents=True, exist_ok=True)
    data_file = ws_dir / "data.txt"
    if not data_file.exists():
        data_file.write_text(datapoint.get("context_window_text", ""), encoding="utf-8")
    return os.path.realpath(str(ws_dir))


def _run_with_pruner_server(
    *, question: str, workspace: str, model: str, client: openai.OpenAI,
    max_iterations: int, pruner_url: str, prune_min_chars: int,
    prune_threshold: float, ablation_backend: str, pre_hooks_resolved, sandbox,
) -> tuple[str, dict, list[dict]]:
    if ablation_backend:
        from .._runtime import ABLATION_PROMPT_SUFFIX
        bash_tool = wrap_bash_tool_with_focus_question(BASH_TOOL)
        system_prompt = SYSTEM_PROMPT + ABLATION_PROMPT_SUFFIX
        experiment = "pruner"
    else:
        bash_tool = BASH_TOOL
        experiment = "pruner" if pruner_url else "baseline"
        system_prompt = SYSTEM_PROMPT if experiment == "pruner" else SYSTEM_PROMPT_BASE
    return run_agent_loop(
        client=client, model=model,
        system_prompt=system_prompt,
        user_prompt=(
            "Please answer the following question by exploring the data files "
            f"in the current directory:\n\n{question}"
        ),
        tools=[bash_tool], sandbox=sandbox, workspace=workspace,
        max_iterations=max_iterations, experiment=experiment,
        pruner_url=pruner_url, prune_strategy="threshold",
        prune_min_chars=prune_min_chars, prune_threshold=prune_threshold,
        pruner_backend=ablation_backend, pre_hooks=pre_hooks_resolved,
        tool_call_fallback_parser=maybe_mimo_parser(model),
        include_per_turn_usage=True,
    )


def _run_with_local_baseline(
    *, question: str, workspace: str, model: str, client: openai.OpenAI,
    max_iterations: int, baseline: BaselinePruner, prune_min_chars: int,
    prune_threshold: float, pre_hooks_resolved, sandbox,
) -> tuple[str, dict, list[dict]]:
    from .._runtime import pruner_client as pc_mod
    from .._runtime import ABLATION_PROMPT_SUFFIX

    orig = pc_mod.call_pruner

    def _local_call(url, history, tool_call, tool_response, *, threshold=0.5,
                    tools=None, pruner_backend="", context_focus_question="",
                    timeout=120):
        try:
            res = baseline.prune(
                history=history, tool_call=tool_call,
                tool_response=tool_response, threshold=threshold,
                query=context_focus_question or derive_query(tool_call),
            )
            return res.pruned_code, {
                "original_chars": res.original_chars,
                "pruned_chars": res.pruned_chars,
                "original_lines": res.original_lines,
                "kept_line_count": res.kept_line_count,
                "latency_ms": res.latency_ms,
                "error_msg": res.error_msg,
                "backend": baseline.name,
            }
        except Exception as exc:
            return tool_response, {"error": str(exc), "backend": baseline.name}

    pc_mod.call_pruner = _local_call
    try:
        bash_tool = wrap_bash_tool_with_focus_question(BASH_TOOL)
        return run_agent_loop(
            client=client, model=model,
            system_prompt=SYSTEM_PROMPT + ABLATION_PROMPT_SUFFIX,
            user_prompt=(
                "Please answer the following question by exploring the data files "
                f"in the current directory:\n\n{question}"
            ),
            tools=[bash_tool], sandbox=sandbox, workspace=workspace,
            max_iterations=max_iterations, experiment="pruner",
            pruner_url="local-baseline", prune_strategy="threshold",
            prune_min_chars=prune_min_chars, prune_threshold=prune_threshold,
            pre_hooks=pre_hooks_resolved,
            tool_call_fallback_parser=maybe_mimo_parser(model),
            include_per_turn_usage=True,
        )
    finally:
        pc_mod.call_pruner = orig


@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m"),
    base_url: str = typer.Option("", "--base-url", envvar="OPENAI_BASE_URL"),
    api_key: str = typer.Option("EMPTY", "--api-key", envvar="OPENAI_API_KEY"),
    pruner_url: str = typer.Option("", "--pruner-url", envvar="PRUNER_URL"),
    dataset: str = typer.Option("synth", "--dataset", "-d", help="synth|dnd"),
    experiment: str = typer.Option("baseline", "--experiment", "-e",
                                    help="baseline | pruner"),
    data_file: str = typer.Option("", "--data-file", "-f",
                                   help="Local JSONL file (overrides --dataset HF download)"),
    max_context_len: int = typer.Option(65536, "--max-context-len"),
    min_context_len: int = typer.Option(0, "--min-context-len"),
    max_iterations: int = typer.Option(50, "--max-iterations"),
    prune_min_chars: int = typer.Option(2000, "--prune-min-chars"),
    prune_threshold: float = typer.Option(0.5, "--prune-threshold"),
    ablation_backend: str = typer.Option(
        "", "--ablation-backend",
        help="Server-side ablation backend (llmlingua2|longcodezip|selective_context|self_prune|rerank|swe_pruner)",
    ),
    baseline: str = typer.Option(
        "", "--baseline",
        help="Client-side baseline (mutually exclusive with --pruner-url).",
    ),
    self_prune_url: str = typer.Option("", "--self-prune-url"),
    baseline_device: str = typer.Option("cuda:0", "--baseline-device"),
    concurrency: int = typer.Option(16, "--concurrency", "-j"),
    output_dir: str = typer.Option("results", "--output-dir", "-o"),
    max_examples: int = typer.Option(0, "--max-examples", help="0 = all"),
    pre_hooks: str = typer.Option("default", "--pre-hooks"),
) -> None:
    """Run agent-based eval on Oolong (multi-turn re-cast)."""
    if baseline and pruner_url:
        raise typer.BadParameter("--baseline and --pruner-url are mutually exclusive")
    assert experiment in ("baseline", "pruner")

    try:
        pre_hooks_resolved = resolve_pre_hooks_spec(pre_hooks)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    # Oolong scorers (upstream package). Imported lazily so users only need
    # the package when they actually run Oolong eval.
    try:
        from oolong.eval_helpers import (  # type: ignore
            synth_process_response,
            dnd_process_response,
        )
    except ImportError:
        try:  # alternative layout
            from eval_helpers import (  # type: ignore
                synth_process_response,
                dnd_process_response,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Oolong scorers not found. Install the upstream `oolong` "
                "package (https://github.com/oolongbench/oolong) so that "
                "`from oolong.eval_helpers import synth_process_response` works."
            ) from exc

    if data_file and Path(data_file).exists():
        rprint(f"[bold]Loading from local file:[/bold] {data_file}")
        data = list(jsonlines.open(data_file, "r"))
        process_response = synth_process_response if dataset == "synth" else dnd_process_response
    elif dataset == "synth":
        from datasets import load_dataset
        data = list(load_dataset("oolongbench/oolong-synth")["test"])
        process_response = synth_process_response
    else:
        from datasets import load_dataset
        data = list(load_dataset("oolongbench/oolong-real", "dnd")["test"])
        process_response = dnd_process_response

    if max_context_len > 0:
        data = [d for d in data if d.get("context_len", 0) <= max_context_len]
    if min_context_len > 0:
        data = [d for d in data if d.get("context_len", 0) > min_context_len]

    rprint(f"[bold]Dataset:[/bold] {dataset}, {len(data)} examples")
    rprint(f"[bold]Experiment:[/bold] {experiment}, model={model}")
    if pruner_url:
        rprint(f"[bold]Pruner:[/bold] {pruner_url} min_chars={prune_min_chars} thr={prune_threshold}")
    if baseline:
        rprint(f"[bold]Baseline:[/bold] {baseline} (in-process)")

    safemodelname = model.replace("/", "-").strip("-")
    results_path = Path(output_dir) / dataset / f"{safemodelname}-{experiment}"
    results_path.mkdir(parents=True, exist_ok=True)
    output_file = results_path / "full_output.jsonl"

    completed_ids: set = set()
    if output_file.exists():
        for obj in jsonlines.open(str(output_file), "r"):
            completed_ids.add(obj["id"])
        rprint(f"[yellow]Resuming: {len(completed_ids)} done[/yellow]")

    remaining = [dict(d) for d in data if d["id"] not in completed_ids]
    if max_examples > 0:
        remaining = remaining[:max_examples]

    sandbox = DockerSandbox(image=_DOCKER_IMAGE, truncate_at=8000,
                            truncate_head=4000, truncate_tail=2000)

    baseline_obj: Optional[BaselinePruner] = None
    if baseline:
        baseline_obj = _build_baseline(
            baseline, sglang_url=self_prune_url, device=baseline_device,
        )

    correct = 0.0
    total = 0
    write_lock = threading.Lock()
    traj_dir = results_path / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(datapoint: dict) -> dict:
        cli = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
        ws = _get_workspace(datapoint)
        start_time = time.time()
        if baseline_obj is not None:
            answer_text, agent_stats, trajectory = _run_with_local_baseline(
                question=datapoint["question"], workspace=ws, model=model,
                client=cli, max_iterations=max_iterations,
                baseline=baseline_obj, prune_min_chars=prune_min_chars,
                prune_threshold=prune_threshold,
                pre_hooks_resolved=pre_hooks_resolved, sandbox=sandbox,
            )
        else:
            answer_text, agent_stats, trajectory = _run_with_pruner_server(
                question=datapoint["question"], workspace=ws, model=model,
                client=cli, max_iterations=max_iterations,
                pruner_url=pruner_url, prune_min_chars=prune_min_chars,
                prune_threshold=prune_threshold,
                ablation_backend=ablation_backend,
                pre_hooks_resolved=pre_hooks_resolved, sandbox=sandbox,
            )
        elapsed = time.time() - start_time

        traj_file = traj_dir / f"{datapoint['id']}.json"
        traj_file.write_text(json.dumps({
            "id": datapoint["id"], "question": datapoint["question"],
            "experiment": experiment, "model": model,
            "messages": trajectory, "agent_stats": agent_stats,
            "answer": answer_text,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        result = process_response(datapoint, answer_text, model)
        result["experiment"] = experiment
        result["time_cost"] = round(elapsed, 2)
        result["agent_stats"] = agent_stats
        for key in ("context_len", "task_group", "task", "answer_type"):
            if key not in result:
                result[key] = datapoint.get(key, "")
        return result

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_one, dp): dp for dp in remaining}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"{dataset}/{experiment}"):
            try:
                result = fut.result()
            except Exception as exc:
                row = futures[fut]
                rprint(f"  [red]row {row.get('id', '?')} failed: "
                       f"{type(exc).__name__}: {str(exc)[:140]}[/red]")
                continue
            correct += result.get("score", 0)
            total += 1
            with write_lock:
                with jsonlines.open(str(output_file), "a") as f:
                    f.write(result)

    if total > 0:
        avg = correct / total
        summary = f"Final: {correct:.2f}/{total} = {avg:.4f}"
        (results_path / "overall.txt").write_text(summary)
        rprint(f"[bold green]{summary}[/bold green]")


def _load_results(results_dir: str) -> list[dict]:
    results = []
    for f in sorted(Path(results_dir).glob("*output*.jsonl")):
        for obj in jsonlines.open(str(f), "r"):
            results.append(obj)
    return results


@app.command()
def compare(
    baseline_dir: str = typer.Option(..., "--baseline-dir", "-b"),
    pruner_dir: str = typer.Option(..., "--pruner-dir", "-p"),
    output: str = typer.Option("", "--output", "-o"),
) -> None:
    """Compare baseline vs pruner Oolong results."""
    b_results = _load_results(baseline_dir)
    r_results = _load_results(pruner_dir)
    b_by_id = {r["id"]: r for r in b_results}
    r_by_id = {r["id"]: r for r in r_results}
    common_ids = sorted(set(b_by_id) & set(r_by_id))
    if not common_ids:
        rprint("[red]No common examples[/red]")
        return

    b_scores = [b_by_id[i]["score"] for i in common_ids]
    r_scores = [r_by_id[i]["score"] for i in common_ids]
    b_avg = sum(b_scores) / len(b_scores)
    r_avg = sum(r_scores) / len(r_scores)
    b_pt = sum(b_by_id[i].get("agent_stats", {}).get("total_prompt_tokens", 0) for i in common_ids)
    r_pt = sum(r_by_id[i].get("agent_stats", {}).get("total_prompt_tokens", 0) for i in common_ids)
    b_ct = sum(b_by_id[i].get("agent_stats", {}).get("total_completion_tokens", 0) for i in common_ids)
    r_ct = sum(r_by_id[i].get("agent_stats", {}).get("total_completion_tokens", 0) for i in common_ids)

    table = Table(title="Oolong: Baseline vs Pruner")
    table.add_column("Metric"); table.add_column("Baseline")
    table.add_column("Pruner");  table.add_column("Delta")
    table.add_row("Common examples", str(len(common_ids)), str(len(common_ids)), "")
    table.add_row("Avg Score", f"{b_avg:.4f}", f"{r_avg:.4f}", f"{r_avg - b_avg:+.4f}")
    table.add_row("Prompt tokens", f"{b_pt:,}", f"{r_pt:,}", f"{r_pt - b_pt:+,}")
    table.add_row("Completion tokens", f"{b_ct:,}", f"{r_ct:,}", f"{r_ct - b_ct:+,}")
    rprint(table)

    if output:
        Path(output).write_text(json.dumps({
            "common_count": len(common_ids),
            "baseline_avg_score": b_avg, "pruner_avg_score": r_avg,
            "delta": r_avg - b_avg,
            "baseline_total_tokens": b_pt + b_ct,
            "pruner_total_tokens": r_pt + r_ct,
        }, indent=2))


if __name__ == "__main__":
    app()
