"""SWE-QA / SWE-QA-Pro agent evaluation with optional pruning.

Lightweight bash agent: explores cloned code repositories via shell,
optionally prunes long tool responses through an external pruner server
or one of the six client-side baselines. Produces per-repo
``*_answers.jsonl`` ready for ``judge.py``.

Two ``--variant`` choices:

* ``sweqa``     — original 144-question / 3-repo SWE-QA set.
* ``sweqa-pro`` — TIGER-Lab/SWE-QA-Pro-Bench (260 Qs / 26 repos).
                  Run ``prepare-pro`` first to materialize the per-repo
                  ``questions_pro/<repo>.jsonl`` + ``pro_repos.json`` files.

Pruning options:

* ``--pruner-url``       — call an external pruner server's ``/prune``.
* ``--baseline NAME``    — invoke a client-side ablation baseline directly
                            (``llmlingua2``, ``selective_context``, ``rerank``,
                            ``self_prune``, ``longcodezip``, ``swe_pruner``).
                            Requires ``--prune-strategy threshold|always``.

Bash sandboxing: ``DockerSandbox`` (default) or ``WhitelistSandbox`` (when
``--no-docker`` is set). Both come from
:mod:`swe_pruner_pro.eval._runtime`.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import openai
import typer
from rich import print as rprint
from tqdm import tqdm

from .._runtime import (
    DockerSandbox,
    Sandbox,
    WhitelistSandbox,
    call_pruner,
    derive_query,
    maybe_mimo_parser,
    resolve_pre_hooks_spec,
    run_agent_loop,
    sampling_params,
    sanitize_history_for_prune,
    wrap_bash_tool_with_focus_question,
)
from ..baselines import BaselinePruner

app = typer.Typer(help="SWE-QA agent eval with pruning")

# Default sandboxing image. Override with SWEQA_DOCKER_IMAGE.
_DOCKER_IMAGE = os.getenv("SWEQA_DOCKER_IMAGE", "python:3.12-slim")
_DEFAULT_WHITELIST = frozenset({
    "cat", "head", "tail", "grep", "egrep", "fgrep", "rg",
    "find", "fd", "ls", "tree", "wc", "sort", "uniq",
    "awk", "sed", "cut", "tr", "diff", "echo", "pwd",
    "python", "python3", "git",
})

# ──────────────────────────────────────────────────────────────────────
# Repo metadata
# ──────────────────────────────────────────────────────────────────────

V2_REPOS: dict[str, tuple[str, str]] = {
    "conan": ("https://github.com/conan-io/conan", "52f43d9"),
    "reflex": ("https://github.com/reflex-dev/reflex", "fe0f946"),
    "streamlink": ("https://github.com/streamlink/streamlink", "ab1f365"),
}

BASE_DIR = Path(os.environ.get("SWEQA_DATA_DIR", Path.cwd()))


def _bench_paths(variant: str) -> tuple[Path, Path, dict[str, tuple[str, str]]]:
    if variant == "sweqa":
        return BASE_DIR / "questions", BASE_DIR / "swe-repos", V2_REPOS
    if variant == "sweqa-pro":
        repos_json = BASE_DIR / "pro_repos.json"
        if not repos_json.exists():
            return BASE_DIR / "questions_pro", BASE_DIR / "swe-repos-pro", {}
        meta = {k: tuple(v) for k, v in json.loads(repos_json.read_text()).items()}
        return BASE_DIR / "questions_pro", BASE_DIR / "swe-repos-pro", meta
    raise typer.BadParameter(f"unknown variant '{variant}' (expected sweqa or sweqa-pro)")


# ──────────────────────────────────────────────────────────────────────
# Prompts & tools
# ──────────────────────────────────────────────────────────────────────

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the code repository (read-only). Use grep/find/cat/head/tail/awk/wc/ls/etc. to explore.",
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
You are a code repository exploration agent. You can run bash commands (via the bash tool) \
on a read-only copy of a source code repository.
Use grep, find, cat, head, tail, awk, wc, ls, etc. to explore the codebase and answer the question precisely.

IMPORTANT: When you have the answer, state it clearly as "Answer: <your answer>".
Provide detailed, comprehensive answers based on what you find in the code."""

FILTER_SUFFIX = """

## Automatic Output Filtering
Long command outputs are automatically filtered to show the most relevant lines.
Filtered sections appear as "(filtered N lines)" markers showing how many lines were removed.
This saves context window space — you do NOT need to worry about managing output length.

### Recommended workflow
- **Read broadly**: Use `cat -n`, `grep -rn`, `find` freely. Long outputs are automatically
  compressed, so there is no penalty for reading whole files.
- **If filtered content matters**: Re-run with a more targeted command
  (e.g. `sed -n 'A,Bp' file`) to bring it back into context.
- **Skip filtering for one command**: Prepend `echo "KEEP_ALL_IN_THIS_COMMAND" && ` to your
  command to receive its full unfiltered output."""

SYSTEM_PROMPT = SYSTEM_PROMPT_BASE + FILTER_SUFFIX


# ──────────────────────────────────────────────────────────────────────
# Baseline registry — instantiated on demand
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Pruner shim: when --baseline is set, the agent loop should call the
# baseline directly. We mimic the run_agent_loop pruner contract by
# monkey-patching ``call_pruner``-like behavior into a wrapper. The
# cleanest approach is a thin local prune call done BEFORE delegating
# to run_agent_loop — but run_agent_loop expects a URL. We instead pass
# pruner_url="" and run a private agent loop here when a baseline is
# in play. Simplest path: keep using the shared loop and just point
# to a local in-process HTTP shim. We avoid that complexity by giving
# baselines their own local agent driver below.
# ──────────────────────────────────────────────────────────────────────

def _run_with_pruner_server(
    *, question: str, repo_path: str, model: str, client: openai.OpenAI,
    max_iterations: int, pruner_url: str, prune_strategy: str,
    prune_min_chars: int, prune_threshold: float, ablation_backend: str,
    pre_hooks_resolved, sandbox: Sandbox,
) -> tuple[str, dict, list[dict]]:
    if ablation_backend:
        bash_tool = wrap_bash_tool_with_focus_question(BASH_TOOL)
        from .._runtime import ABLATION_PROMPT_SUFFIX
        system_prompt = SYSTEM_PROMPT + ABLATION_PROMPT_SUFFIX
        experiment = "pruner"
    else:
        bash_tool = BASH_TOOL
        experiment = "pruner" if pruner_url and prune_strategy != "none" else "baseline"
        system_prompt = SYSTEM_PROMPT if experiment == "pruner" else SYSTEM_PROMPT_BASE
    return run_agent_loop(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=(
            "Please explore the code repository and answer the following question.\n"
            "Use bash commands to navigate and search the codebase.\n\n"
            f"Question: {question}"
        ),
        tools=[bash_tool],
        sandbox=sandbox,
        workspace=repo_path,
        max_iterations=max_iterations,
        experiment=experiment,
        pruner_url=pruner_url,
        prune_strategy=prune_strategy,
        prune_min_chars=prune_min_chars,
        prune_threshold=prune_threshold,
        pruner_backend=ablation_backend,
        pre_hooks=pre_hooks_resolved,
        tool_call_fallback_parser=maybe_mimo_parser(model),
    )


# ──────────────────────────────────────────────────────────────────────
# Direct in-process baseline driver
# Mirrors run_agent_loop but routes prune calls to a local BaselinePruner
# instead of the HTTP /prune endpoint. We keep this separate to avoid
# tangling baseline plumbing into the shared loop.
# ──────────────────────────────────────────────────────────────────────

def _run_with_local_baseline(
    *, question: str, repo_path: str, model: str, client: openai.OpenAI,
    max_iterations: int, baseline: BaselinePruner, prune_strategy: str,
    prune_min_chars: int, prune_threshold: float, sandbox: Sandbox,
    pre_hooks_resolved,
) -> tuple[str, dict, list[dict]]:
    # Reuse run_agent_loop; intercept the HTTP path by stubbing call_pruner
    # via a thread-local override. Simpler: monkey-patch the runtime module
    # at the call site. We do that here, scoped to this call.
    from .._runtime import pruner_client as pc_mod

    orig = pc_mod.call_pruner

    def _local_call(url, history, tool_call, tool_response, *, threshold=0.5,
                    tools=None, pruner_backend="", context_focus_question="",
                    timeout=120):
        try:
            res = baseline.prune(
                history=history,
                tool_call=tool_call,
                tool_response=tool_response,
                threshold=threshold,
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
        from .._runtime import ABLATION_PROMPT_SUFFIX
        return run_agent_loop(
            client=client,
            model=model,
            system_prompt=SYSTEM_PROMPT + ABLATION_PROMPT_SUFFIX,
            user_prompt=(
                "Please explore the code repository and answer the following question.\n"
                "Use bash commands to navigate and search the codebase.\n\n"
                f"Question: {question}"
            ),
            tools=[bash_tool],
            sandbox=sandbox,
            workspace=repo_path,
            max_iterations=max_iterations,
            experiment="pruner",
            pruner_url="local-baseline",  # non-empty so loop dispatches to pruner path
            prune_strategy=prune_strategy,
            prune_min_chars=prune_min_chars,
            prune_threshold=prune_threshold,
            pre_hooks=pre_hooks_resolved,
            tool_call_fallback_parser=maybe_mimo_parser(model),
        )
    finally:
        pc_mod.call_pruner = orig


# ──────────────────────────────────────────────────────────────────────
# Question loading
# ──────────────────────────────────────────────────────────────────────

def _load_questions(repo_name: str, questions_dir: Path) -> list[dict]:
    path = questions_dir / f"{repo_name}.jsonl"
    questions: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("question"):
                qid = hashlib.md5(data["question"].encode()).hexdigest()[:12]
                data["id"] = f"{repo_name}_{qid}"
                data["repo"] = repo_name
                questions.append(data)
    return questions


# ──────────────────────────────────────────────────────────────────────
# CLI: setup
# ──────────────────────────────────────────────────────────────────────

@app.command()
def setup(
    variant: str = typer.Option("sweqa", "--variant", help="sweqa | sweqa-pro"),
) -> None:
    """Clone repos and check out specified commits for the given variant."""
    _, repos_dir, repo_meta = _bench_paths(variant)
    if not repo_meta:
        rprint(f"[red]No repo metadata for variant '{variant}'. "
               f"For sweqa-pro, run 'prepare-pro' first.[/red]")
        raise typer.Exit(1)
    repos_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, commit) in repo_meta.items():
        repo_dir = repos_dir / name
        if repo_dir.exists():
            rprint(f"[green]{name}[/green] already exists, skipping clone")
        else:
            rprint(f"Cloning {name} ({url})...")
            subprocess.run(["git", "clone", url, str(repo_dir)], check=True)
        subprocess.run(["git", "fetch", "origin", commit], cwd=str(repo_dir), capture_output=True)
        subprocess.run(["git", "checkout", commit], cwd=str(repo_dir), capture_output=True)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True,
        )
        rprint(f"[green]{name}[/green] @ {result.stdout.strip()}")
    rprint(f"[bold green]Setup complete for variant={variant}[/bold green]")


@app.command("prepare-pro")
def prepare_pro(
    hf_repo: str = typer.Option("TIGER-Lab/SWE-QA-Pro-Bench", "--hf-repo"),
    hf_local_dir: str = typer.Option("/tmp/swe-qa-pro-bench", "--hf-local-dir"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Download SWE-QA-Pro-Bench from HF and convert into per-repo jsonl."""
    from huggingface_hub import snapshot_download

    questions_dir = BASE_DIR / "questions_pro"
    repos_json = BASE_DIR / "pro_repos.json"
    if questions_dir.exists() and any(questions_dir.glob("*.jsonl")) and not force:
        rprint(f"[yellow]{questions_dir} already populated; pass --force[/yellow]")
        return

    rprint(f"Downloading {hf_repo} -> {hf_local_dir}")
    snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=hf_local_dir)
    src = Path(hf_local_dir) / "data" / "test.jsonl"
    if not src.exists():
        rprint(f"[red]Expected {src} after download, not found[/red]")
        raise typer.Exit(1)

    questions_dir.mkdir(parents=True, exist_ok=True)
    repo_meta: dict[str, tuple[str, str]] = {}
    by_repo: dict[str, list[dict]] = {}
    with open(src, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            full = d["repo"]
            short = full.split("/")[-1]
            commit = d["commit_id"]
            repo_meta[short] = (f"https://github.com/{full}", commit)
            by_repo.setdefault(short, []).append({
                "question": d["question"],
                "answer": d["answer"],
                "repo_full": full,
                "commit_id": commit,
                "category_type": d.get("qa_type", {}).get("class_name", ""),
                "category": d.get("qa_type", {}).get("sub_class_name", ""),
            })
    for short, rows in sorted(by_repo.items()):
        out = questions_dir / f"{short}.jsonl"
        with open(out, "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    repos_json.write_text(json.dumps(repo_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[bold green]Prepared {len(by_repo)} repos / "
           f"{sum(len(v) for v in by_repo.values())} questions[/bold green]")


# ──────────────────────────────────────────────────────────────────────
# CLI: run
# ──────────────────────────────────────────────────────────────────────

@app.command()
def run(
    model: str = typer.Option(..., "--model", "-m"),
    experiment: str = typer.Option("baseline", "--experiment", "-e"),
    variant: str = typer.Option("sweqa", "--variant", help="sweqa | sweqa-pro"),
    repos: str = typer.Option("", "--repos", "-r", help="Comma-separated repo names; empty = all"),
    max_iterations: int = typer.Option(50, "--max-iterations"),
    concurrency: int = typer.Option(4, "--concurrency", "-j"),
    output_dir: str = typer.Option("results", "--output-dir", "-o"),
    max_examples: int = typer.Option(0, "--max-examples"),
    repos_dir: str = typer.Option("", "--repos-dir", help="Override repos directory"),
    # LLM connection
    openai_base_url: str = typer.Option("", "--openai-base-url",
                                         envvar="OPENAI_BASE_URL"),
    openai_api_key: str = typer.Option("", "--openai-api-key",
                                        envvar="OPENAI_API_KEY"),
    # Pruner (HTTP server mode)
    pruner_url: str = typer.Option("", "--pruner-url", envvar="PRUNER_URL"),
    prune_strategy: str = typer.Option("none", "--prune-strategy",
                                        help="none|always|threshold"),
    prune_min_chars: int = typer.Option(2000, "--prune-min-chars"),
    prune_threshold: float = typer.Option(0.5, "--prune-threshold"),
    ablation_backend: str = typer.Option(
        "", "--ablation-backend",
        help="Route /prune to a server-side ablation backend "
             "(llmlingua2|longcodezip|selective_context|self_prune|rerank|swe_pruner)",
    ),
    # In-process baseline mode (no pruner server needed)
    baseline: str = typer.Option(
        "", "--baseline",
        help="Use a client-side baseline pruner directly (mutually exclusive with --pruner-url). "
             "One of: llmlingua2, selective_context, rerank, self_prune, longcodezip, swe_pruner.",
    ),
    self_prune_url: str = typer.Option("", "--self-prune-url",
                                        help="OpenAI-compatible URL for the self_prune baseline"),
    baseline_device: str = typer.Option("cuda:0", "--baseline-device"),
    pre_hooks: str = typer.Option("default", "--pre-hooks",
                                   help="default|none|all|comma-list of "
                                        "early_history,command_whitelist,repeat_read"),
    no_docker: bool = typer.Option(False, "--no-docker",
                                    help="Use a whitelist subprocess sandbox instead of Docker"),
) -> None:
    """Run agent-based eval on SWE-QA / SWE-QA-Pro repos."""
    if baseline and pruner_url:
        raise typer.BadParameter("--baseline and --pruner-url are mutually exclusive")

    questions_dir, default_repos_dir, _ = _bench_paths(variant)
    if repos.strip():
        repo_names = [r.strip() for r in repos.split(",") if r.strip()]
    else:
        repo_names = sorted(p.stem for p in questions_dir.glob("*.jsonl"))
    if not repo_names:
        rprint(f"[red]No questions found in {questions_dir}.[/red]")
        raise typer.Exit(1)

    try:
        pre_hooks_resolved = resolve_pre_hooks_spec(pre_hooks)
    except ValueError as e:
        raise typer.BadParameter(str(e))

    sandbox: Sandbox = (
        WhitelistSandbox(allowed_commands=_DEFAULT_WHITELIST)
        if no_docker else
        DockerSandbox(image=_DOCKER_IMAGE, truncate_at=8000,
                      truncate_head=4000, truncate_tail=2000)
    )

    base_url = openai_base_url
    api_key = openai_api_key or "EMPTY"

    safemodelname = model.split("/")[-1]
    results_path = Path(output_dir) / f"{safemodelname}-{experiment}"
    results_path.mkdir(parents=True, exist_ok=True)

    all_questions: list[dict] = []
    for repo_name in repo_names:
        all_questions.extend(_load_questions(repo_name, questions_dir))

    rprint(f"[bold]Variant:[/bold] {variant}, repos: {len(repo_names)}, Qs: {len(all_questions)}")
    rprint(f"[bold]Model:[/bold] {model} @ {base_url or '<default>'}")
    if pruner_url:
        rprint(f"[bold]Pruner:[/bold] {pruner_url} strategy={prune_strategy} min_chars={prune_min_chars}")
    if baseline:
        rprint(f"[bold]Baseline:[/bold] {baseline} (in-process)")

    baseline_obj: Optional[BaselinePruner] = None
    if baseline:
        baseline_obj = _build_baseline(
            baseline, sglang_url=self_prune_url, device=baseline_device,
        )

    write_lock = threading.Lock()
    flat_jobs: list[dict] = []
    for repo_name in repo_names:
        rqs = [q for q in all_questions if q["repo"] == repo_name]
        if max_examples > 0:
            rqs = rqs[:max_examples]

        _repos_base = Path(repos_dir) if repos_dir else default_repos_dir
        repo_path = os.path.realpath(str(_repos_base / repo_name))
        if not os.path.isdir(repo_path):
            rprint(f"[red]{repo_name}: repo not found at {repo_path}, run 'setup' first[/red]")
            continue

        output_file = results_path / f"{repo_name}_answers.jsonl"
        completed_ids: set = set()
        if output_file.exists():
            with open(output_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        completed_ids.add(json.loads(line.strip())["id"])
        remaining = [q for q in rqs if q["id"] not in completed_ids]
        for q in remaining:
            flat_jobs.append({
                "datapoint": q, "repo_name": repo_name,
                "repo_path": repo_path, "output_file": output_file,
            })

    if not flat_jobs:
        rprint("[green]Nothing to do (everything already in results)[/green]")
        return
    rprint(f"\n[bold]Cross-repo parallel:[/bold] {len(flat_jobs)} questions, {concurrency} workers")

    def _process_one(job: dict) -> dict:
        dp = job["datapoint"]
        cli = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)
        start = time.time()
        if baseline_obj is not None:
            answer_text, agent_stats, trajectory = _run_with_local_baseline(
                question=dp["question"], repo_path=job["repo_path"],
                model=model, client=cli, max_iterations=max_iterations,
                baseline=baseline_obj, prune_strategy=prune_strategy or "threshold",
                prune_min_chars=prune_min_chars, prune_threshold=prune_threshold,
                sandbox=sandbox, pre_hooks_resolved=pre_hooks_resolved,
            )
        else:
            answer_text, agent_stats, trajectory = _run_with_pruner_server(
                question=dp["question"], repo_path=job["repo_path"],
                model=model, client=cli, max_iterations=max_iterations,
                pruner_url=pruner_url, prune_strategy=prune_strategy,
                prune_min_chars=prune_min_chars, prune_threshold=prune_threshold,
                ablation_backend=ablation_backend,
                pre_hooks_resolved=pre_hooks_resolved, sandbox=sandbox,
            )
        elapsed = time.time() - start

        traj_dir = results_path / "trajectories" / job["repo_name"]
        traj_dir.mkdir(parents=True, exist_ok=True)
        (traj_dir / f"{dp['id']}.json").write_text(json.dumps({
            "id": dp["id"], "repo": job["repo_name"],
            "question": dp["question"], "experiment": experiment, "model": model,
            "messages": trajectory, "agent_stats": agent_stats,
            "answer": answer_text,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "id": dp["id"], "repo": job["repo_name"],
            "question": dp["question"], "answer": answer_text,
            "ground_truth": dp.get("ground_truth") or dp.get("answer", ""),
            "experiment": experiment, "time_cost": round(elapsed, 2),
            "agent_stats": agent_stats,
            "category_type": dp.get("category_type", ""),
            "category": dp.get("category", ""),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process_one, j): j for j in flat_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=variant):
            job = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                rprint(f"  [red]row {job['datapoint'].get('id', '?')} ({job['repo_name']}) "
                       f"failed: {type(exc).__name__}: {str(exc)[:140]}[/red]")
                continue
            with write_lock:
                with open(job["output_file"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

    rprint(f"\n[bold green]Done. Results in {results_path}[/bold green]")
    rprint("Run the judge separately:")
    rprint(f"  python -m swe_pruner_pro.eval.sweqa.judge judge-all -d {output_dir}")


if __name__ == "__main__":
    app()
