"""Materialize the immutable DDS-Mamba LaSOT validation partition."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from dds_mamba.splits import SPLIT_ID, lasot_train_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lasot-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("configs/splits"))
    args = parser.parse_args()
    names = [line.strip() for line in (args.lasot_root / "training_set.txt").read_text().splitlines() if line.strip()]
    train, validation = lasot_train_validation(names)
    args.out.mkdir(parents=True, exist_ok=True)
    train_path = args.out / "lasot_train_v1.json"
    validation_path = args.out / "lasot_val_v1.json"
    combined_path = args.out / "lasot_trainval_v1.json"
    train_path.write_text(json.dumps(train, indent=2) + "\n")
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    combined_path.write_text(json.dumps({"split_id": SPLIT_ID, "train": train, "validation": validation}, indent=2) + "\n")
    digest = lambda path: sha256(path.read_bytes()).hexdigest()
    record = {
        "schema_version": 1,
        "split_id": SPLIT_ID,
        "source_training_set_sha256": sha256("\n".join(names).encode("utf-8")).hexdigest(),
        "source_sequence_count": len(names),
        "train_sequence_count": len(train),
        "validation_sequence_count": len(validation),
        "files": {
            train_path.name: digest(train_path),
            validation_path.name: digest(validation_path),
            combined_path.name: digest(combined_path),
        },
    }
    (args.out / "lasot_split_record_v1.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
