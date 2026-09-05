"""Regression coverage for multi-seed sequence-bootstrap aggregation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_ci.py"
    spec = importlib.util.spec_from_file_location("bootstrap_ci", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_seed_values_are_averaged_per_sequence_before_bootstrap() -> None:
    tool = _module()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        paths = []
        for index, rows in enumerate(((1.0, 3.0), (3.0, 5.0), (5.0, 7.0)), 1):
            path = root / f"seed_{index}.csv"
            path.write_text(f"sequence,metric\na,{rows[0]}\nb,{rows[1]}\n")
            paths.append(path)
        names, values = tool._aligned_seed_values(paths, "metric", "sequence")
        assert names == ["a", "b"]
        assert values.shape == (3, 2)
        assert values.mean(axis=0).tolist() == [3.0, 5.0]
