# DDS-Mamba-v1 LaSOT training/validation lists

The canonical training partition is derived only from the official LaSOT
`training_set.txt`, never from the public Protocol-II testing set.  Before any
training run, materialize and version the following four files together:

```bash
PYTHONPATH=src python scripts/make_lasot_trainval_split.py \
  --lasot-root /data/LaSOT --out configs/splits
```

The resulting `lasot_train_v1.json` (896 sequence names),
`lasot_val_v1.json` (224 sequence names), `lasot_trainval_v1.json`, and
`lasot_split_record_v1.json` are the required split-list artefacts.  The record
contains the exact source-list and output SHA-256 values.  A paper release must
ship these materialized JSON files beside this README; this repository does not
vendor the LaSOT dataset or its official sequence list.
