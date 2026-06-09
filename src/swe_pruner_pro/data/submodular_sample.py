"""Diversity-maximizing subset selection via facility-location submodular sampling.

Lazy greedy with a priority queue — O(n log n) per pick, total O(n k log n)
with stale-entry re-evaluation. Provides the (1 - 1/e) approximation guarantee.

The kernel uses instance-level metadata (language, repo, response-size bucket,
log-step-count) — see the paper Section 3 / Table 5 for context.
"""
from __future__ import annotations

import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, help="Submodular diversity sampling for training data")
console = Console()


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def infer_language(instance_id: str) -> str:
    if instance_id.startswith("mswe_"):
        parts = instance_id.split("_", 2)
        if len(parts) >= 2:
            lang = parts[1].lower()
            return "c++" if lang in ("c++", "cpp") else lang
    return "python"


def infer_repo(instance_id: str) -> str:
    if instance_id.startswith("mswe_"):
        parts = instance_id.split("_", 2)
        if len(parts) >= 3:
            repo_part = parts[2]
            idx = repo_part.rfind("-")
            if idx > 0 and repo_part[idx + 1:].isdigit():
                return repo_part[:idx]
            return repo_part
    elif instance_id.startswith("claude-opus"):
        if "_" in instance_id:
            repo_part = instance_id.split("_", 1)[1]
            idx = repo_part.rfind("-")
            if idx > 0 and repo_part[idx + 1:].isdigit():
                return repo_part[:idx]
            return repo_part
    else:
        parts = instance_id.rsplit("_", 1)
        if len(parts) == 2 and (parts[1].startswith("pr") or parts[1].isdigit()):
            return parts[0]
    return instance_id


def size_bucket(avg_lines: float) -> str:
    if avg_lines < 30:
        return "short"
    if avg_lines < 80:
        return "medium"
    if avg_lines < 200:
        return "long"
    return "very_long"


def step_count_bucket(n_steps: int) -> str:
    if n_steps <= 3:
        return "few"
    if n_steps <= 10:
        return "some"
    return "many"


@dataclass
class InstanceMeta:
    instance_id: str
    language: str
    repo: str
    size_bucket: str
    step_bucket: str
    n_steps: int
    step_indices: list[int] = field(default_factory=list)


def extract_metadata(steps: list[dict]) -> list[InstanceMeta]:
    groups: dict[str, list] = defaultdict(list)
    for i, s in enumerate(steps):
        groups[s["instance_id"]].append((i, s))
    out = []
    for iid, items in groups.items():
        avg_lines = np.mean([
            s.get("response_lines", len(s["tool_response"].splitlines()))
            for _, s in items
        ])
        out.append(InstanceMeta(
            instance_id=iid,
            language=infer_language(iid),
            repo=infer_repo(iid),
            size_bucket=size_bucket(avg_lines),
            step_bucket=step_count_bucket(len(items)),
            n_steps=len(items),
            step_indices=[i for i, _ in items],
        ))
    return out


# ---------------------------------------------------------------------------
# Similarity & lazy-greedy facility location
# ---------------------------------------------------------------------------

def build_similarity_matrix(
    instances: list[InstanceMeta],
    w_lang: float = 3.0,
    w_repo: float = 2.0,
    w_size: float = 0.5,
    w_continuous: float = 1.0,
) -> np.ndarray:
    n = len(instances)
    sim = np.zeros((n, n), dtype=np.float32)
    langs = [m.language for m in instances]
    repos = [m.repo for m in instances]
    sizes = [m.size_bucket for m in instances]
    log_steps = np.log1p([m.n_steps for m in instances]).astype(np.float32)
    if log_steps.max() > log_steps.min():
        log_steps = (log_steps - log_steps.min()) / (log_steps.max() - log_steps.min())
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            if langs[i] == langs[j]: s += w_lang
            if repos[i] == repos[j]: s += w_repo
            if sizes[i] == sizes[j]: s += w_size
            diff = abs(log_steps[i] - log_steps[j])
            s += w_continuous * float(np.exp(-diff * diff * 8.0))
            sim[i, j] = s
            sim[j, i] = s
    return sim


def lazy_greedy_facility_location(
    sim: np.ndarray, step_counts: np.ndarray, budget: int,
    max_steps_per_instance: int = 0,
) -> list[int]:
    n = sim.shape[0]
    costs = step_counts.copy()
    if max_steps_per_instance > 0:
        costs = np.minimum(costs, max_steps_per_instance)
    current_max = np.zeros(n, dtype=np.float32)
    selected: list[int] = []
    selected_set: set[int] = set()
    total_cost = 0
    pq: list[tuple[float, int, int]] = []
    for j in range(n):
        if costs[j] <= 0:
            continue
        gain = float(sim[:, j].sum())
        heapq.heappush(pq, (-gain / costs[j], 0, j))
    while pq and total_cost < budget:
        _neg, ts, j = heapq.heappop(pq)
        if j in selected_set:
            continue
        if costs[j] + total_cost > budget * 1.5:
            continue
        if ts < len(selected):
            marginal = float(np.maximum(sim[:, j] - current_max, 0).sum())
            if marginal < 1e-8:
                marginal = 1e-6
            heapq.heappush(pq, (-marginal / costs[j], len(selected), j))
            continue
        if costs[j] + total_cost > budget:
            continue
        selected.append(j)
        selected_set.add(j)
        total_cost += int(costs[j])
        np.maximum(current_max, sim[:, j], out=current_max)
    return selected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_steps(path: Path) -> list[dict]:
    console.print(f"Loading {path} ...")
    return [json.loads(line) for line in open(path) if line.strip()]


def _print_distribution(instances: list[InstanceMeta], title: str = "Dataset"):
    console.print(f"\n[bold]{title}[/bold]: {len(instances)} instances, "
                  f"{sum(m.n_steps for m in instances)} steps")
    t = Table(title="Languages")
    t.add_column("Language"); t.add_column("Instances", justify="right"); t.add_column("Steps", justify="right")
    groups: dict[str, list[InstanceMeta]] = defaultdict(list)
    for m in instances:
        groups[m.language].append(m)
    for lang, ms in sorted(groups.items(), key=lambda x: -len(x[1])):
        t.add_row(lang, str(len(ms)), str(sum(m.n_steps for m in ms)))
    console.print(t)
    sb = Counter(m.size_bucket for m in instances)
    console.print(f"Sizes: {dict(sb)}")


@app.command()
def stats(input_jsonl: Path = typer.Argument(...)):
    """Show dataset distribution statistics."""
    steps = _load_steps(input_jsonl)
    _print_distribution(extract_metadata(steps))


@app.command()
def sample(
    input_jsonl: Path = typer.Argument(..., help="Filtered-steps JSONL"),
    output: Path = typer.Option("sampled.jsonl", "-o"),
    budget: int = typer.Option(50000, "-n", help="Target number of steps"),
    max_steps_per_instance: int = typer.Option(50, "--max-per-instance"),
    w_lang: float = typer.Option(3.0, "--w-lang"),
    w_repo: float = typer.Option(2.0, "--w-repo"),
    w_size: float = typer.Option(0.5, "--w-size"),
    seed: int = typer.Option(42),
):
    """Select a diverse subset using facility-location submodular maximization."""
    steps = _load_steps(input_jsonl)
    instances = extract_metadata(steps)
    _print_distribution(instances, "Before sampling")

    console.print(f"\nBuilding {len(instances)}x{len(instances)} similarity ...")
    sim = build_similarity_matrix(instances, w_lang=w_lang, w_repo=w_repo, w_size=w_size)
    step_counts = np.array([m.n_steps for m in instances], dtype=np.int64)

    console.print("Running lazy-greedy facility location ...")
    selected = lazy_greedy_facility_location(sim, step_counts, budget,
                                              max_steps_per_instance=max_steps_per_instance)
    console.print(f"  Selected {len(selected)} instances")

    rng = np.random.RandomState(seed)
    chosen: list[int] = []
    for idx in selected:
        meta = instances[idx]
        ids = meta.step_indices
        if max_steps_per_instance > 0 and len(ids) > max_steps_per_instance:
            ids = rng.choice(ids, max_steps_per_instance, replace=False).tolist()
        chosen.extend(ids)
    if len(chosen) > budget:
        rng.shuffle(chosen)
        chosen = chosen[:budget]
    chosen.sort()

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for i in chosen:
            f.write(json.dumps(steps[i], ensure_ascii=False) + "\n")
    console.print(f"\n[green]Wrote {len(chosen)} steps to {output}[/green]")
    _print_distribution([instances[i] for i in selected], "After sampling")


if __name__ == "__main__":
    app()
