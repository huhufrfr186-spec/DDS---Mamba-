"""Immutable, executable DDS-Mamba-v1 ablation definitions."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .assets import file_sha256, load_locked_yaml


@dataclass(frozen=True)
class AblationSpec:
    name: str
    network: Mapping[str, Any]
    online: Mapping[str, Any]
    description: str
    manifest_sha256: str


def load_ablation_spec(
    base_manifest: str | Path,
    variant: str = "full",
    ablation_manifest: str | Path = "manifests/dds_mamba_v1_ablations.yaml",
) -> AblationSpec:
    """Resolve one named ablation and bind it to the exact base manifest."""
    path = Path(ablation_manifest)
    document = load_locked_yaml(path)
    expected_base = str(document.get("base_manifest_sha256", ""))
    actual_base = file_sha256(base_manifest)
    if expected_base != actual_base:
        raise RuntimeError(
            "ablation manifest is not bound to this base manifest: "
            f"expected {expected_base}, got {actual_base}"
        )
    variants = document.get("variants", {})
    if variant not in variants:
        raise ValueError(f"unknown DDS-Mamba-v1 variant {variant!r}; expected one of {sorted(variants)}")
    value = variants[variant]
    if not isinstance(value, dict):
        raise ValueError(f"invalid variant definition for {variant}")
    network = dict(value.get("network", {}))
    online = dict(value.get("online", {}))
    return AblationSpec(
        variant,
        network,
        online,
        str(value.get("description", "")),
        file_sha256(path),
    )


def load_selected_thresholds(
    path: str | Path,
    *,
    base_manifest: str | Path,
    ablation_manifest_sha256: str,
    checkpoint: str | Path,
    variant: str,
) -> dict[str, object]:
    """Load a validation-only threshold selection bound to its exact run inputs."""
    path = Path(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("threshold selection must be a schema_version=1 JSON record")
    expected = {
        "manifest_sha256": file_sha256(base_manifest),
        "ablation_manifest_sha256": ablation_manifest_sha256,
        "checkpoint_sha256": file_sha256(checkpoint),
        "variant": variant,
    }
    mismatch = {key: (value.get(key), target) for key, target in expected.items() if value.get(key) != target}
    if mismatch:
        raise RuntimeError(f"threshold selection is incompatible with this prediction run: {mismatch}")
    if value.get("protocol") is None or "no public-test access" not in str(value["protocol"]):
        raise ValueError("threshold selection lacks the required validation-only protocol declaration")
    if value.get("debug_max_sequences") is not None:
        raise ValueError("a debug-capped threshold selection cannot be used for public predictions")
    overrides = value.get("selected_online_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("threshold selection has no selected_online_overrides mapping")
    return dict(overrides)
