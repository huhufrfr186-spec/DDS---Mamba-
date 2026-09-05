"""Deterministic, test-free LaSOT training/validation partition."""
from __future__ import annotations

from hashlib import sha256


SPLIT_ID = "dds-mamba-v1-lasot-trainval-20260712"


def lasot_train_validation(names: list[str]) -> tuple[list[str], list[str]]:
    """Hold out exactly 20% of the official 1,120 training names by SHA-256 rank."""
    unique = sorted(set(names))
    if len(unique) != 1120:
        raise ValueError(f"expected the official 1,120 LaSOT training names, got {len(unique)}")
    ranked = sorted(unique, key=lambda name: sha256(f"{SPLIT_ID}:{name}".encode()).hexdigest())
    val = set(ranked[:224])
    return [name for name in unique if name not in val], [name for name in unique if name in val]
