"""Evaluate the linear-probe LR (trained on training features) on the held-out
SWE-QA eval set, and compare to the bilinear K=4 ckpt's predictions.

Pipeline:
  1. Build per-line mean(HS) features for both training (features/0424_coder_next_patched_noquax)
     and held-out (features/sweqa_holdout_coder_next) sets.
  2. Train a single logistic regression (1 weight vector, 2049 params).
  3. Apply LR to the held-out set; group token logits by line; line score =
     mean of token logits in that line.
  4. Sweep thresholds, report best line-level F1 + matching keep_ratio.
  5. Compare to the bilinear ckpt's pred_kept_lines (already in
     logs/0507m_q_bilinear/holdout_eval/eval_predictions.jsonl).

Optionally writes an LR predictions jsonl in the same schema as the bilinear
eval_predictions, so the existing llm-as-judge pipeline can score it.
"""

import json
from pathlib import Path

import numpy as np
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score


def _load_packed(features_dir: Path):
    with open(features_dir / "index.json") as f:
        idx = json.load(f)
    hidden_dim: int = idx["hidden_dim"]
    samples = idx["samples"]
    end_token = max(int(s["offset"]) + int(s["length"]) for s in samples)
    hs_path = features_dir / "hidden_states.bin"
    fp_size = hs_path.stat().st_size
    if fp_size == end_token * hidden_dim * 2:
        hs_dtype = np.float16
    elif fp_size == end_token * hidden_dim * 4:
        hs_dtype = np.float32
    else:
        raise ValueError(f"Unexpected hidden_states.bin size {fp_size} vs end_token={end_token}, dim={hidden_dim}")
    labels_path = features_dir / "token_labels.bin"
    lab_size = labels_path.stat().st_size
    lab_dtype = np.int64 if lab_size == end_token * 8 else (np.int32 if lab_size == end_token * 4 else np.int16)
    lid_path = features_dir / "token_line_ids.bin"
    lid_size = lid_path.stat().st_size
    lid_dtype = np.int64 if lid_size == end_token * 8 else (np.int32 if lid_size == end_token * 4 else np.int16)
    hs = np.memmap(hs_path, dtype=hs_dtype, mode="r", shape=(end_token, hidden_dim))
    labels = np.memmap(labels_path, dtype=lab_dtype, mode="r", shape=(end_token,))
    line_ids = np.memmap(lid_path, dtype=lid_dtype, mode="r", shape=(end_token,))
    return idx, samples, hs, labels, line_ids, hs_dtype


def _per_sample_lines(samples, hs, labels, line_ids, sample_indices, max_lines_per_sample, rng):
    """Return list of (sample_idx, line_id, mean_hs, line_label) for every line."""
    out = []
    for si in sample_indices:
        s = samples[int(si)]
        offset, length = int(s["offset"]), int(s["length"])
        seg_lab = np.asarray(labels[offset:offset + length])
        seg_lid = np.asarray(line_ids[offset:offset + length])
        valid = ((seg_lab == 0) | (seg_lab == 1)) & (seg_lid > 0)
        if not valid.any():
            continue
        unique_lids = np.unique(seg_lid[valid])
        unique_lids = unique_lids[unique_lids > 0]
        if max_lines_per_sample > 0 and unique_lids.size > max_lines_per_sample:
            unique_lids = rng.choice(unique_lids, size=max_lines_per_sample, replace=False)
        seg_hs = np.asarray(hs[offset:offset + length], dtype=np.float32)
        for lid in unique_lids:
            mask = (seg_lid == int(lid)) & valid
            if not mask.any():
                continue
            mean_hs = seg_hs[mask].mean(axis=0)
            line_lab = int(np.round(seg_lab[mask].mean()))
            out.append((int(si), int(lid), mean_hs, line_lab))
    return out


def _frags_to_set(frags) -> set[int]:
    return {int(x) for x in frags}


def _f1_from_sets(pred: set[int], gt: set[int]) -> tuple[float, int, int, int]:
    if not pred and not gt:
        return 1.0, 0, 0, 0
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return f1, tp, fp, fn


def main(
    train_features: Path = typer.Option(Path("features/0424_coder_next_patched_noquax")),
    holdout_features: Path = typer.Option(Path("features/sweqa_holdout_coder_next")),
    holdout_jsonl: Path = typer.Option(Path("datasets/sweqa_holdout.jsonl")),
    bilinear_eval: Path = typer.Option(Path("logs/0507m_q_bilinear/holdout_eval/eval_predictions.jsonl")),
    n_train_samples: int = typer.Option(8000, help="Train LR on N training samples"),
    max_lines_per_train_sample: int = typer.Option(60),
    seed: int = typer.Option(0),
    out_predictions: Path = typer.Option(Path("logs/0507m_q_bilinear/holdout_eval/lr_eval_predictions.jsonl"),
                                          help="Write LR predictions in the same schema as bilinear eval_predictions"),
) -> None:
    rng = np.random.default_rng(seed)

    print(f"[train] loading {train_features}")
    tr_idx, tr_samples, tr_hs, tr_lab, tr_lid, tr_dtype = _load_packed(train_features)
    print(f"  {len(tr_samples)} samples, dtype={tr_dtype}")

    pick = rng.choice(len(tr_samples), size=min(n_train_samples, len(tr_samples)), replace=False)
    train_lines = _per_sample_lines(tr_samples, tr_hs, tr_lab, tr_lid, pick, max_lines_per_train_sample, rng)
    H_train = np.stack([h for _, _, h, _ in train_lines]).astype(np.float32)
    Y_train = np.array([y for _, _, _, y in train_lines], dtype=np.int64)
    print(f"[train] {len(Y_train):,} lines, pos_ratio={Y_train.mean():.3f}")

    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(H_train, Y_train)
    print(f"  ||w||={np.linalg.norm(clf.coef_):.3f}, b={clf.intercept_[0]:.3f}")

    print(f"[holdout] loading {holdout_features}")
    ho_idx, ho_samples, ho_hs, ho_lab, ho_lid, ho_dtype = _load_packed(holdout_features)
    print(f"  {len(ho_samples)} samples, dtype={ho_dtype}")

    with open(holdout_features.parent / holdout_features.name / "index_to_row.json") as f:
        stem_to_row = {k: int(v) for k, v in json.load(f).items()}
    with open(holdout_jsonl) as f:
        src_rows = [json.loads(l) for l in f]
    with open(bilinear_eval) as f:
        bil_eval = [json.loads(l) for l in f]
    bil_by_key = {(r["instance_id"], r["step_idx"]): r for r in bil_eval}

    n_test_lines = 0
    n_test_lines_pos = 0
    all_test_logits: list[float] = []
    all_test_labels: list[int] = []
    per_sample = []
    skipped = 0
    bil_pred_kept_total: dict[tuple[str, int], set[int]] = {}

    for s in ho_samples:
        stem = s["stem"]
        offset, length = int(s["offset"]), int(s["length"])
        seg_lid = np.asarray(ho_lid[offset:offset + length])
        valid_tok = seg_lid > 0
        if not valid_tok.any():
            skipped += 1
            continue
        seg_hs = np.asarray(ho_hs[offset:offset + length], dtype=np.float32)

        if stem not in stem_to_row:
            skipped += 1
            continue
        src_row = src_rows[stem_to_row[stem]]
        kept_frags = set(int(x) for x in src_row.get("kept_frags") or [])
        key = (src_row["instance_id"], src_row["step_idx"])
        if key not in bil_by_key:
            skipped += 1
            continue
        bil_pred_kept_total[key] = set(int(x) for x in bil_by_key[key]["pred_kept_lines"])

        unique_lids = np.unique(seg_lid[valid_tok])
        unique_lids = unique_lids[unique_lids > 0]
        line_hs = []
        line_labs = []
        for lid in unique_lids:
            mask = (seg_lid == int(lid)) & valid_tok
            if not mask.any():
                continue
            line_hs.append(seg_hs[mask].mean(axis=0))
            line_labs.append(1 if int(lid) in kept_frags else 0)
        H = np.stack(line_hs)
        Y = np.array(line_labs, dtype=np.int64)

        logit = H @ clf.coef_.flatten() + clf.intercept_[0]
        all_test_logits.extend(logit.tolist())
        all_test_labels.extend(Y.tolist())
        n_test_lines += len(Y)
        n_test_lines_pos += int(Y.sum())

        per_sample.append({
            "key": key,
            "instance_id": src_row["instance_id"],
            "step_idx": src_row["step_idx"],
            "stem": stem,
            "lids": unique_lids.astype(int).tolist(),
            "lr_logit": logit.astype(np.float32),
            "lr_prob": (1.0 / (1.0 + np.exp(-logit))).astype(np.float32),
            "kept_frags": kept_frags,
            "n_lines": int(seg_lid.max()),
        })

    print(f"[holdout] matched {len(per_sample)}/{len(ho_samples)} samples (skipped {skipped})")
    print(f"[holdout] {n_test_lines:,} valid lines, pos_ratio={n_test_lines_pos/n_test_lines:.3f}  "
          f"(holdout has empty kept_frags — judge-based eval, no line F1)")

    logits = np.array(all_test_logits)
    yarr = np.array(all_test_labels)
    if yarr.sum() > 0 and yarr.sum() < len(yarr):
        auc_token = roc_auc_score(yarr, logits)
        print(f"[LR line-token] held-out micro AUC={auc_token:.4f}  "
              f"micro F1@th=0.5={f1_score(yarr, (logits > 0).astype(int)):.4f}")

    bil_kept = bil_total = 0
    for s in per_sample:
        pred = bil_pred_kept_total.get(s["key"], set())
        bil_kept += len(pred); bil_total += s["n_lines"]
    bil_keep_ratio = bil_kept / max(bil_total, 1)
    print(f"[Bilinear K=4 ckpt] keep_ratio={bil_keep_ratio:.3f}")

    all_logits_flat = np.concatenate([s["lr_logit"] for s in per_sample])
    all_lid_counts = np.array([s["n_lines"] for s in per_sample])
    target_keep = int(round(bil_keep_ratio * all_lid_counts.sum()))
    if target_keep > 0:
        th_match = float(np.partition(all_logits_flat, -target_keep)[-target_keep])
    else:
        th_match = float(all_logits_flat.max() + 1)
    th_natural = 0.0
    print(f"[LR thresholds] natural=0.00,  matched-keep-ratio={th_match:.3f} (keep≈{bil_keep_ratio:.3f})")

    if out_predictions is not None:
        for variant_name, th in [("natural", th_natural), ("matched", th_match)]:
            out_path = out_predictions.parent / out_predictions.name.replace(
                ".jsonl", f"_{variant_name}.jsonl"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            kept = total = 0
            with open(out_path, "w") as fout:
                for s in per_sample:
                    bil_row = bil_by_key[s["key"]]
                    pred_kept = sorted(int(lid) for lid, lg in zip(s["lids"], s["lr_logit"]) if lg > th)
                    pred_line_scores = {int(lid): float(p) for lid, p in zip(s["lids"], s["lr_prob"])}
                    out_row = {
                        "instance_id": bil_row["instance_id"],
                        "step_idx": bil_row["step_idx"],
                        "tool_call": bil_row["tool_call"],
                        "tool_response": bil_row["tool_response"],
                        "kept_frags": bil_row["kept_frags"],
                        "pred_kept_lines": pred_kept,
                        "pred_line_scores": pred_line_scores,
                        "next_turn": bil_row["next_turn"],
                        "history": bil_row["history"],
                        "_holdout_repo": bil_row.get("_holdout_repo"),
                        "_lr_threshold": th,
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    kept += len(pred_kept); total += s["n_lines"]
            print(f"[saved] {out_path}  ({len(per_sample)} rows, th={th:.3f}, keep_ratio={kept/max(total,1):.3f})")


if __name__ == "__main__":
    typer.run(main)
