"""Immutable external-asset manifest and verified checkpoint acquisition."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen
import shutil

import yaml


@dataclass(frozen=True)
class Asset:
    name: str; architecture: str; implementation: str; url: str; sha256: str; bytes: int; feature_tap: str; frozen: bool


def load_locked_yaml(path: str | Path, *, required_schema: int = 1) -> dict:
    """Load an immutable YAML document protected by a sibling SHA-256 file.

    The asset and ablation manifests both use this mechanism.  A manifest is
    deliberately not a convenience configuration file: changing a byte makes
    a different experimental method and requires regenerating its lock.
    """
    path = Path(path)
    lock = path.with_suffix(path.suffix + ".sha256")
    if not lock.exists():
        raise FileNotFoundError(f"immutable manifest requires sibling lock file: {lock}")
    expected = lock.read_text().strip().split()[0]
    actual = file_sha256(path)
    if expected != actual:
        raise RuntimeError(f"manifest lock violation: expected {expected}, got {actual}")
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict) or not document.get("immutable") or document.get("schema_version") != required_schema:
        raise ValueError(f"immutable YAML must be schema_version={required_schema} and immutable=true: {path}")
    return document


def load_asset_lock(path: str | Path) -> dict:
    """Load the immutable DDS-Mamba base asset/algorithm manifest."""
    return load_locked_yaml(path)


def asset(document: dict, name: str) -> Asset:
    value = document["asset_lock"][name]
    return Asset(name, value["architecture"], value["implementation"], value["checkpoint_url"], value["checkpoint_sha256"], int(value["checkpoint_bytes"]), value["feature_tap"], bool(value["frozen"]))


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""): h.update(block)
    return h.hexdigest()


def obtain_verified(item: Asset, directory: str | Path) -> Path:
    """Download once, then reject wrong bytes instead of silently changing method version."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True); destination = directory / Path(item.url).name
    if not destination.exists():
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urlopen(item.url) as src, partial.open("wb") as dst: shutil.copyfileobj(src, dst)
        partial.replace(destination)
    if destination.stat().st_size != item.bytes or file_sha256(destination) != item.sha256:
        raise RuntimeError(f"Asset lock violation for {item.name}: {destination}")
    return destination
