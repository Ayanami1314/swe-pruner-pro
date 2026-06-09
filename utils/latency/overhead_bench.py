"""Overhead bench against sweqa-format MiMo trajectories (decode-heavy).

Schema differs from mini-swe-agent: structured tool_calls + reasoning_content.
At each assistant turn p:
  - /generate input: msgs[:p]; max_tokens = tokens(content) + tokens(reasoning_content)
    of msgs[p] so the replay simulates the original decode load (reasoning tokens
    drive the bulk of decode here).
  - /prune input: history = msgs[:p-2], tool_call = msgs[p-2].tool_calls[0],
    tool_response = msgs[p-1].content. Skipped at p<2 or when msgs[p-1]
    isn't a tool message.

ratio = Σ prune_i / Σ gen_i, same as overhead_bench.py. The only difference
is the trajectory loader and per-turn output-token accounting.
"""
from __future__ import annotations
import concurrent.futures
import json
import os
import queue
import random
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
import typer

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore


SERVER = os.environ.get("PRUNER_URL", "http://localhost:8000")
TRAJ_ROOT = Path(os.environ.get("TRAJ_ROOT", "trajectories"))
TOKENIZER_DIR = os.environ.get("PRUNER_BACKBONE", "Qwen/Qwen3-Coder-Next")
MODEL_ARG = TOKENIZER_DIR


class _TokCounter:
    def __init__(self, path: str):
        self.tok = None
        if AutoTokenizer is not None:
            try:
                self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            except Exception as e:
                typer.echo(f"[bench] tokenizer load failed ({e}); using char/4 fallback", err=True)

    def count(self, s: str) -> int:
        if not s:
            return 1
        if self.tok is not None:
            try:
                return max(1, len(self.tok.encode(s, add_special_tokens=False)))
            except Exception:
                pass
        return max(1, len(s) // 4)


@dataclass
class Trajectory:
    instance: str
    messages: list[dict]
    asst_positions: list[int]
    out_token_counts: list[int]
    n_turns: int


def _strip_keys(m: dict) -> dict:
    """Keep only OpenAI-API-recognized keys; drop reasoning_content from
    history (server doesn't accept it as input). Leave tool_calls / tool_call_id
    intact so the chat template renders correctly."""
    keep = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return {k: v for k, v in m.items() if k in keep and v is not None}


def load_trajectories(traj_root: Path, n: int, seed: int, tok: _TokCounter,
                      max_turns: Optional[int] = None) -> list[Trajectory]:
    if not traj_root.exists():
        raise SystemExit(f"trajectory root not found: {traj_root}")
    files: list[Path] = []
    for sub in sorted(traj_root.iterdir()):
        if sub.is_dir():
            files.extend(sorted(sub.glob("*.json")))
    rng = random.Random(seed)
    rng.shuffle(files)
    out: list[Trajectory] = []
    for f in files:
        if len(out) >= n:
            break
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        msgs = data.get("messages") or []
        if not msgs:
            continue
        asst_pos = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
        if not asst_pos:
            continue
        out_counts: list[int] = []
        for p in asst_pos:
            m = msgs[p]
            n_tok = tok.count(m.get("content") or "") + tok.count(m.get("reasoning_content") or "")
            out_counts.append(max(1, n_tok))
        if max_turns is not None and len(asst_pos) > max_turns:
            asst_pos = asst_pos[:max_turns]
            out_counts = out_counts[:max_turns]
            msgs = msgs[: asst_pos[-1] + 1]
        # Strip reasoning_content from history so /generate doesn't reject it
        clean_msgs = [_strip_keys(m) for m in msgs]
        instance = f.parent.name + "/" + f.stem
        out.append(Trajectory(
            instance=instance, messages=clean_msgs, asst_positions=asst_pos,
            out_token_counts=out_counts, n_turns=len(asst_pos),
        ))
    if len(out) < n:
        typer.echo(f"[bench] WARNING: only loaded {len(out)}/{n} trajectories", err=True)
    return out


def call_generate(sess: requests.Session, base_url: str, model: str,
                  messages: list[dict], max_tokens: int, timeout: float,
                  api_key: Optional[str]) -> tuple[bool, float, Optional[str]]:
    body = {
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": False, "ignore_eos": True,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.time()
    try:
        r = sess.post(f"{base_url}/v1/chat/completions", json=body,
                      headers=headers, timeout=timeout)
        r.raise_for_status()
        r.json()
        return True, (time.time() - t0) * 1000, None
    except Exception as e:
        return False, (time.time() - t0) * 1000, f"{type(e).__name__}: {e}"


def call_prune(sess: requests.Session, pruner_url: str, history: list[dict],
               tool_call: dict, tool_response: str, threshold: float,
               timeout: float) -> tuple[bool, float, Optional[str]]:
    payload = {
        "history": history,
        "tool_call": tool_call,
        "tool_response": tool_response,
        "threshold": threshold,
        "pruner_backend": "ours",
    }
    t0 = time.time()
    try:
        r = sess.post(f"{pruner_url}/prune", json=payload, timeout=timeout)
        r.raise_for_status()
        r.json()
        return True, (time.time() - t0) * 1000, None
    except Exception as e:
        return False, (time.time() - t0) * 1000, f"{type(e).__name__}: {e}"


@dataclass
class TrajResult:
    instance: str
    n_turns: int
    gen_ms: list[Optional[float]] = field(default_factory=list)
    prune_ms: list[Optional[float]] = field(default_factory=list)
    gen_errors: list[Optional[str]] = field(default_factory=list)
    prune_errors: list[Optional[str]] = field(default_factory=list)


def replay_baseline(sess: requests.Session, traj: Trajectory, *,
                    generate_url: str, model: str, api_key: Optional[str],
                    timeout: float) -> TrajResult:
    res = TrajResult(instance=traj.instance, n_turns=traj.n_turns)
    for i, p in enumerate(traj.asst_positions):
        prompt = traj.messages[:p]
        n_out = traj.out_token_counts[i]
        ok, ms, err = call_generate(sess, generate_url, model, prompt,
                                     n_out, timeout, api_key)
        res.gen_ms.append(ms if ok else None)
        res.gen_errors.append(err)
    return res


def replay_pruner(sess: requests.Session, traj: Trajectory, *,
                  pruner_url: str, threshold: float,
                  timeout: float) -> TrajResult:
    res = TrajResult(instance=traj.instance, n_turns=traj.n_turns)
    for i, p in enumerate(traj.asst_positions):
        # Need msgs[p-2] = assistant(tool_calls), msgs[p-1] = tool(content).
        if p < 2 or traj.messages[p - 1].get("role") != "tool":
            res.prune_ms.append(None); res.prune_errors.append(None); continue
        prev_asst = traj.messages[p - 2]
        if prev_asst.get("role") != "assistant" or not prev_asst.get("tool_calls"):
            res.prune_ms.append(None); res.prune_errors.append(None); continue
        tc = prev_asst["tool_calls"][0]
        fn = tc.get("function", {})
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"command": args}
        tool_call = {"name": fn.get("name", "bash"), "arguments": args or {}}
        tool_response = traj.messages[p - 1].get("content") or ""
        history = traj.messages[:p - 2]
        ok, ms, err = call_prune(sess, pruner_url, history, tool_call,
                                  tool_response, threshold, timeout)
        res.prune_ms.append(ms if ok else None)
        res.prune_errors.append(err)
    return res


def run_concurrent(trajs: list[Trajectory], replay_fn, concurrency: int,
                   session_factory) -> list[TrajResult]:
    q: queue.Queue = queue.Queue()
    for t in trajs:
        q.put(t)
    results: dict[str, TrajResult] = {}
    lock = threading.Lock()

    def worker():
        sess = session_factory()
        while True:
            try:
                t = q.get_nowait()
            except queue.Empty:
                return
            r = replay_fn(sess, t)
            with lock:
                results[t.instance] = r
            q.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        for _ in range(concurrency):
            ex.submit(worker)
    return [results[t.instance] for t in trajs]


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    return {"n": len(values), "p50": s[len(s) // 2],
            "p95": s[int(0.95 * len(s))], "mean": statistics.mean(values)}


def make_plot(per_traj: list[dict], out_path: Path) -> None:
    if plt is None:
        typer.echo("[bench] matplotlib unavailable, skipping plot", err=True)
        return
    Ks = [r["K"] for r in per_traj]
    ratios = [r["ratio_pct"] for r in per_traj]
    plt.figure(figsize=(7, 4))
    plt.scatter(Ks, ratios)
    plt.axhline(100, color="grey", linestyle="--", linewidth=0.5)
    plt.xlabel("turns"); plt.ylabel("Σprune / Σgen (%)")
    plt.title("Pruner overhead vs trajectory length (MiMo, decode-heavy)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    typer.echo(f"[bench] plot -> {out_path}")


cli = typer.Typer(help=__doc__, pretty_exceptions_enable=False, add_completion=False)


@cli.command()
def main(
    server: str = typer.Option(SERVER, "--server"),
    pruner_url: str = typer.Option("", "--pruner-url"),
    generate_url: str = typer.Option("", "--generate-url"),
    model: str = typer.Option(MODEL_ARG, "--model"),
    api_key: str = typer.Option("EMPTY", "--api-key"),
    traj_root: Path = typer.Option(TRAJ_ROOT, "--traj-root"),
    n_tasks: int = typer.Option(16, "--n-tasks"),
    concurrencies: str = typer.Option("16", "--concurrencies"),
    threshold: float = typer.Option(0.5, "--threshold"),
    timeout_s: float = typer.Option(600.0, "--timeout"),
    max_turns: int = typer.Option(0, "--max-turns",
                                   help="Cap turns per trajectory (0 = no cap)"),
    output: Path = typer.Option(Path("results/overhead_bench_sweqa.json"), "--output", "-o"),
    plot_path: Path = typer.Option(Path("results/overhead_bench_sweqa.png"), "--plot"),
    seed: int = typer.Option(42, "--seed"),
    skip_proxy: bool = typer.Option(True, "--skip-proxy/--use-proxy"),
):
    p_url = pruner_url or server
    g_url = generate_url or f"{server}/model-raw"
    if skip_proxy:
        os.environ["no_proxy"] = (os.environ.get("no_proxy", "") + ",.xiaomi.srv").lstrip(",")
        os.environ["NO_PROXY"] = os.environ["no_proxy"]

    conc_list = [int(c) for c in concurrencies.split(",") if c.strip()]
    typer.echo(f"[bench] loading tokenizer from {TOKENIZER_DIR} ...")
    tok = _TokCounter(TOKENIZER_DIR)

    trajs = load_trajectories(traj_root, n=n_tasks, seed=seed, tok=tok,
                              max_turns=max_turns or None)
    Ks = [t.n_turns for t in trajs]
    total_out_toks = sum(sum(t.out_token_counts) for t in trajs)
    typer.echo(f"[bench] {len(trajs)} trajectories  K range=[{min(Ks)}..{max(Ks)}]  "
               f"total assistant turns = {sum(Ks)}  total decode tokens = {total_out_toks}")

    def sess_factory():
        return requests.Session()

    runs = []
    for c in conc_list:
        typer.echo(f"\n[bench] === concurrency={c} ===")
        t0 = time.time()
        base = run_concurrent(trajs,
                              lambda s, t: replay_baseline(s, t, generate_url=g_url,
                                                             model=model, api_key=api_key,
                                                             timeout=timeout_s),
                              c, sess_factory)
        t1 = time.time()
        prune = run_concurrent(trajs,
                               lambda s, t: replay_pruner(s, t, pruner_url=p_url,
                                                             threshold=threshold,
                                                             timeout=timeout_s),
                               c, sess_factory)
        t2 = time.time()
        typer.echo(f"[bench]  baseline pass  : {t1 - t0:.1f}s for {len(trajs)} trajectories")
        typer.echo(f"[bench]  pruner pass    : {t2 - t1:.1f}s for {len(trajs)} trajectories")

        per_traj = []
        for b, p in zip(base, prune):
            gen_lats = [x for x in b.gen_ms if x is not None]
            pr_lats = [x for x in p.prune_ms if x is not None]
            tg = sum(gen_lats); tp = sum(pr_lats)
            per_traj.append({
                "instance": b.instance, "K": b.n_turns,
                "T_base_ms": tg, "T_prune_ms": tp,
                "ratio_pct": (tp / tg * 100) if tg > 0 else float("nan"),
            })
        sum_g = sum(r["T_base_ms"] for r in per_traj)
        sum_p = sum(r["T_prune_ms"] for r in per_traj)
        ratios = [r["ratio_pct"] for r in per_traj if r["T_base_ms"] > 0]
        runs.append({
            "concurrency": c,
            "sum_ratio_pct": (sum_p / sum_g * 100) if sum_g > 0 else float("nan"),
            "per_traj_ratio_summary": _summary(ratios),
            "per_traj": per_traj,
        })

    out = {"server": server, "model": model, "n_tasks": len(trajs),
           "traj_root": str(traj_root), "runs": runs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2))
    typer.echo(f"[bench]  saved -> {output}")

    last = runs[-1]
    typer.echo(f"\n  Σ prune / Σ gen (sum-of-sums)  = {last['sum_ratio_pct']:.2f}%")
    s = last["per_traj_ratio_summary"]
    typer.echo(f"  per-traj ratio  p50={s.get('p50',0):.2f}%  "
               f"p95={s.get('p95',0):.2f}%  mean={s.get('mean',0):.2f}%")
    typer.echo(f"  {'instance':<40} {'K':>4} {'T_base_s':>10} {'T_prune_s':>10} {'ratio%':>8}")
    for r in sorted(last["per_traj"], key=lambda x: x["K"]):
        typer.echo(f"  {r['instance'][:40]:<40} {r['K']:>4} "
                   f"{r['T_base_ms']/1000:>10.2f} {r['T_prune_ms']/1000:>10.2f} "
                   f"{r['ratio_pct']:>8.2f}")

    make_plot(last["per_traj"], plot_path)


if __name__ == "__main__":
    cli()
