# Controlled experiments required for DDS-Mamba-v1

All rows below must be trained from scratch with the three manifest seeds,
selected on the fixed LaSOT training validation split, and evaluated with the
same saved-checkpoint prediction protocol.  No result may be reported until
the checkpoint hashes, prediction directories, evaluator commands, and logs
are archived.

## Mamba replacement controls

Use `manifests/dds_mamba_v1_controls.yaml` with `train_v1.py` or
`scripts/run_three_seeds.py`.

- `dual_gru_control`: causal GRU replacement in both branches.
- `dual_mlp_control`: causal prefix-MLP replacement in both branches.
- `dual_transformer_control`: causal Transformer replacement in both branches.
- `position_mlp_control` and `appearance_mlp_control`: branch-local controls.

These controls retain the frozen encoders, crop construction, token ordering,
heads, state controller, threshold policy, QACU, RFMB, Kalman filter, and
candidate cache.  They test whether the selective-SSM branch itself is useful,
not whether the complete online controller is useful.

## Deconfounded controller controls

- `no_kalman_only` and `no_candidate_cache_only` isolate the two recovery
  mechanisms.  `no_kalman_cache_joint` is reported only as their joint removal.
- `neutral_identity_evidence` keeps the identity-aware threshold path and DINO
  forward pass but neutralizes all controller/write identity evidence.
- `tied_active_state_only` ties states after active commits without modifying
  the appearance state during recovery.

## Threshold selection and sensitivity

Use the locked validation-only grid:

```bash
PYTHONPATH=src python scripts/select_validation_thresholds.py \
  --config configs/train_v1.yaml \
  --checkpoint outputs/full_seed_20260711/best.pt \
  --ablation-manifest manifests/dds_mamba_v1_controls.yaml \
  --out outputs/full_seed_20260711/threshold_selection.json
```

The script performs a deterministic coordinate sweep on the 224-sequence
LaSOT training validation split, records every candidate score, and refuses a
grid that permits public-test access.  Pass the resulting JSON to
`run_benchmark.py --threshold-selection ...`; the runner verifies its manifest,
variant, and checkpoint hashes before prediction.
