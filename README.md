# DDS-Mamba v1

DDS-Mamba v1 is a manifest-locked, trainable long-term RGB visual tracker. The
canonical network is MAE ViT-B/16 for frozen template--search features plus
DINOv2 ViT-S/14 for frozen identity features, a two-layer position selective
SSM, a four-layer appearance selective SSM, a Kalman state, QACU, and an RFMB.
The exact algorithm is defined by the immutable manifests, not by command-line
defaults.

This repository contains complete source for training, validation, prediction,
official-layout adapters, immutable ablations, three-seed launch, sequence
bootstrap confidence intervals, and CUDA runtime measurement. It contains no
trained DDS-Mamba checkpoint and no public-benchmark score.

## Locked assets and manifests

- Base algorithm and assets: manifests/dds_mamba_v1.yaml
- Full and six ablations: manifests/dds_mamba_v1_ablations.yaml
- Both YAML files require their adjacent SHA-256 lock. A changed byte is a new
  method version.
- Template/search encoder: official MAE ViT-B/16, block-12 normalized patch
  tokens, 128/256 crops, frozen.
- Identity encoder: official DINOv2 ViT-S/14, normalized final CLS token, 224
  crop, frozen.

The model downloads absent assets then verifies byte count and SHA-256. To
verify cached assets:

    PYTHONPATH=src python scripts/verify_manifest_assets.py --assets assets

## Installation

Use the pinned CUDA image for a reproducible training environment:

    docker build -t dds-mamba:v1 .
    docker run --gpus all -it -v /data:/data dds-mamba:v1 bash

Or create a Python 3.10+ environment and install the pinned project
dependencies:

    python -m venv .venv
    . .venv/bin/activate
    pip install -e '.[dev]'

Run static and controller-level checks:

    PYTHONPATH=src python -m compileall -q src train_v1.py run_benchmark.py scripts
    PYTHONPATH=src python scripts/run_synthetic_experiments.py

## Train one seed

Obtain the official LaSOT training archive, including training_set.txt. First
materialize the validation split:

    PYTHONPATH=src python scripts/make_lasot_trainval_split.py \
      --lasot-root /data/LaSOT --out configs/splits

Copy or edit configs/train_v1.yaml so that lasot_root and output_dir are real
paths, then train:

    PYTHONPATH=src python train_v1.py --config configs/train_v1.yaml --variant full

The trainer uses native-resolution 16-frame clips and batch size one. It writes
last.pt, best.pt, history.json, run_manifest.json, and a materialized
checkpoint variant record. The selection metric is 21-threshold success AUC on
the deterministic LaSOT training validation split.

## Three seeds and ablations

The locked seeds are 20260711, 20260712, and 20260713.

    PYTHONPATH=src python scripts/run_three_seeds.py \
      --config configs/train_v1.yaml --variant full --output-root outputs

Run all immutable variants:

    PYTHONPATH=src python scripts/run_ablations.py \
      --config configs/train_v1.yaml --output-root outputs

Each variant is retrained. The checkpoint records its base-manifest hash,
ablation-manifest hash, and variant name; prediction refuses an incompatible
manifest.

## Public-benchmark prediction writers

After training, write predictions with the exact variant stored in best.pt:

    PYTHONPATH=src python run_benchmark.py \
      --assets assets --checkpoint outputs/full_seed_20260711/best.pt \
      --benchmark lasot --root /data/LaSOT --out predictions

The other locked protocols are anti_uav300_rgb with --split test-dev, webuav3m
with --split Test, and vtuav_rgb_longterm with a DUT-VTUAV-V root containing
test_LT. The writer accesses labels only for the first-frame initialization and
writes a provenance JSON beside evaluator-ready per-sequence rows.

Run the dataset owner's official evaluator on the saved files. Do not report a
public score before archiving the evaluator revision, command, raw prediction
directory, output log, checkpoint hash, and manifest hashes.

## Confidence intervals and runtime

Given one official per-sequence evaluator CSV for each trained seed, compute
the locked 10,000-resample interval.  The script first averages every
sequence across seeds and then resamples sequences; it does not bootstrap
three rounded seed-level scores:

    PYTHONPATH=src python scripts/bootstrap_ci.py \
      --input seed_20260711_per_sequence.csv \
      --input seed_20260712_per_sequence.csv \
      --input seed_20260713_per_sequence.csv \
      --metric success_auc --out success_auc_ci.json

For a paired full-minus-ablation interval, pass one matching ``--other`` CSV
for every ``--input`` seed in the same order. The script checks exact sequence
ID agreement before computing the per-sequence paired difference.
Measure device-resident controller-plus-model latency after 100 warm-up frames:

    PYTHONPATH=src python scripts/measure_runtime.py --help

The timing protocol excludes disk I/O, decode, and host-to-device transfer;
the JSON labels it accordingly.

