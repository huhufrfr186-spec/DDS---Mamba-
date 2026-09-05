"""Sequence-bootstrap confidence intervals across one or more training seeds.

For multiple ``--input`` files, each file is one seed's per-sequence official
metric export.  The estimator first averages each sequence over seeds, then
bootstraps sequences.  This is deliberately different from treating the
three seed-level aggregate scores as a bootstrap population.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dds_mamba.assets import load_asset_lock


def _read(path: Path, metric: str, sequence_column: str) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not rows[0] or sequence_column not in rows[0] or metric not in rows[0]:
        raise ValueError(f"{path} must be a non-empty CSV containing {sequence_column!r} and {metric!r}")
    result: dict[str, float] = {}
    for row in rows:
        sequence = row[sequence_column]
        if not sequence or sequence in result:
            raise ValueError(f"{path} has an empty or duplicate sequence identifier")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"{path} has a non-finite {metric} for {sequence}")
        result[sequence] = value
    return result


def _interval(values: np.ndarray, rng: np.random.Generator, resamples: int) -> tuple[float, float, float]:
    indices = rng.integers(0, len(values), size=(resamples, len(values)), endpoint=False)
    bootstrap = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _aligned_seed_values(paths: list[Path], metric: str, sequence_column: str) -> tuple[list[str], np.ndarray]:
    """Return deterministic ``[seed, sequence]`` values after exact ID validation."""
    if not paths:
        raise ValueError("at least one --input CSV is required")
    reports = [_read(path, metric, sequence_column) for path in paths]
    names = sorted(reports[0])
    if not names:
        raise ValueError("per-sequence CSV must contain at least one sequence")
    expected = set(names)
    for path, report in zip(paths[1:], reports[1:]):
        if set(report) != expected:
            raise ValueError(f"{path} does not contain exactly the same sequence identifiers as the first --input")
    return names, np.asarray([[report[name] for name in names] for report in reports], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, action="append", required=True,
        help="per-sequence evaluator CSV for one seed; repeat once per seed",
    )
    parser.add_argument("--metric", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument(
        "--other", type=Path, action="append",
        help="matching ablation CSV for the corresponding --input seed; repeat once per seed",
    )
    parser.add_argument("--manifest", type=Path, default=Path("manifests/dds_mamba_v1.yaml"))
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--resamples", type=int)
    args = parser.parse_args()
    manifest = load_asset_lock(args.manifest)
    resamples = int(args.resamples or manifest["training"]["bootstrap_resamples"])
    if resamples < 1:
        raise ValueError("resamples must be positive")
    names, primary_by_seed = _aligned_seed_values(args.input, args.metric, args.sequence_column)
    # The reported point estimate and interval use the same per-sequence
    # three-seed average.  Thus each sequence, not each rounded seed mean, is
    # the resampling unit.
    values = primary_by_seed.mean(axis=0)
    mean, lower, upper = _interval(values, np.random.default_rng(args.seed), resamples)
    report: dict[str, object] = {
        "metric": args.metric,
        "sequence_count": len(names),
        "seed_count": int(primary_by_seed.shape[0]),
        "seed_inputs": [str(path.resolve()) for path in args.input],
        "seed_means": [float(row.mean()) for row in primary_by_seed],
        "aggregation": "per-sequence mean across seeds, then sequence bootstrap",
        "resamples": resamples,
        "seed": args.seed,
        "mean": mean,
        "ci95": [lower, upper],
    }
    if args.other is not None:
        if len(args.other) != len(args.input):
            raise ValueError("paired bootstrap requires exactly one --other CSV for each --input seed")
        other_names, other_by_seed = _aligned_seed_values(args.other, args.metric, args.sequence_column)
        if other_names != names:
            raise ValueError("paired bootstrap requires exactly the same sequence identifiers for full and ablation CSVs")
        difference = (primary_by_seed - other_by_seed).mean(axis=0)
        diff_mean, diff_lower, diff_upper = _interval(difference, np.random.default_rng(args.seed), resamples)
        report["paired_difference_full_minus_other"] = {
            "mean": diff_mean,
            "ci95": [diff_lower, diff_upper],
            "other_seed_inputs": [str(path.resolve()) for path in args.other],
            "aggregation": "matched full-minus-ablation difference per seed and sequence, then mean across seeds and sequence bootstrap",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
