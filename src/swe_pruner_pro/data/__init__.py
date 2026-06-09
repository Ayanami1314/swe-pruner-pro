"""Data preparation utilities for SWE-Pruner Pro.

Modules:
  parse_trajectories  Parse 5 HF trajectory datasets into unified per-step JSONL
  submodular_sample   Facility-location greedy sub-sampling to ~50k steps
  label_with_claude   Per-line keep/prune labelling via Claude Sonnet
  extract_features    SGLang hidden-state extraction + memmap packing

All four are typer CLIs invokable as ``python -m swe_pruner_pro.data.<name>``.
"""
