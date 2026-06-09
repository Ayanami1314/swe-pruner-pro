"""Pack per-sample ``.npz`` shards from :mod:`extract_features` into three
memmap ``.bin`` files plus an ``index.json`` describing per-sample byte offsets.

Output layout::

    <output_dir>/
        hidden_states.bin    float16, total_tokens x hidden_dim
        token_labels.bin     int16,  total_tokens
        token_line_ids.bin   int16,  total_tokens
        index.json           per-sample offsets + hidden_dim

The trainer mmaps these three files for zero-copy random access.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from tqdm import tqdm

console = Console()


def main(
    feature_dir: Path = typer.Argument(..., help="Directory of .npz shards"),
    output_dir: Path = typer.Option(None, "-o",
        help="Output directory (defaults to feature_dir)."),
) -> None:
    out = Path(output_dir or feature_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards = sorted(feature_dir.glob("*.npz"))
    if not shards:
        console.print("[red]No .npz shards found[/red]")
        raise typer.Exit(1)

    hs_path = out / "hidden_states.bin"
    lbl_path = out / "token_labels.bin"
    lid_path = out / "token_line_ids.bin"
    index: dict = {"samples": [], "hidden_dim": None}

    hs_off = lbl_off = lid_off = 0
    with open(hs_path, "wb") as fh, open(lbl_path, "wb") as fl, open(lid_path, "wb") as fi:
        for shard in tqdm(shards, desc="Packing"):
            d = np.load(shard)
            hs = d["hidden_states"].astype(np.float16)
            lbl = d["token_labels"].astype(np.int16)
            lid = d["token_line_ids"].astype(np.int16)
            n_tok, dim = hs.shape
            if index["hidden_dim"] is None:
                index["hidden_dim"] = int(dim)
            elif int(dim) != index["hidden_dim"]:
                raise RuntimeError(f"Inconsistent hidden_dim in {shard}: {dim} vs {index['hidden_dim']}")
            fh.write(hs.tobytes()); fl.write(lbl.tobytes()); fi.write(lid.tobytes())
            index["samples"].append({
                "id": shard.stem,
                "n_tokens": int(n_tok),
                "hs_offset": hs_off,
                "label_offset": lbl_off,
                "line_offset": lid_off,
            })
            hs_off += hs.nbytes
            lbl_off += lbl.nbytes
            lid_off += lid.nbytes

    (out / "index.json").write_text(json.dumps(index, indent=2))
    console.print(f"[green]Packed {len(shards)} shards into {out}[/green]")


if __name__ == "__main__":
    typer.run(main)
