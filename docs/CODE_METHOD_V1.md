# DDS-Mamba-v1 code-to-method contract

The immutable base manifest and its SHA-256 lock define the reference algorithm.
The immutable ablation manifest is separately locked and bound to the base
manifest hash. No command-line flag may silently alter these definitions.

| Paper requirement | Canonical implementation |
| --- | --- |
| Frozen MAE ViT-B/16 and DINOv2 ViT-S/14 assets, preprocessing, hashes, and feature taps | manifests/dds_mamba_v1.yaml; src/dds_mamba/assets.py; src/dds_mamba/model_v1.py |
| One persistent position state, appearance state, Kalman filter, RFMB, QACU statistics, counters, and cache | src/dds_mamba/online.py:DDSOnlineState |
| Pre-frame RFMB context, stable score/index Top-K, context-scale gate, and read-only recurrence | DDSOnlineState.memory_arrays/read_open; model_v1.py:MemoryReader |
| Bounded box decoder, projected spatial gate, confidence quality, and shared candidate construction | model_v1.py; runtime.py:candidate_from_output |
| Cholesky/Joseph Kalman update, deterministic PSD repair, and reset covariance | src/dds_mamba/kalman.py |
| Exact active/weak/lost/recovery transitions and QACU write discipline | src/dds_mamba/online.py:DDSOnlineState.step |
| Exact enabled loss terms, masks, constants, and teacher probability | src/dds_mamba/losses.py; train_v1.py:_clip_loss |
| Detached state boundaries and identical online crops during train/inference | train_v1.py; runtime.py:DDSTracker |
| Three locked seeds, six executable ablations, bootstrap interval, and CUDA timing protocol | scripts/run_three_seeds.py; scripts/run_ablations.py; scripts/bootstrap_ci.py; scripts/measure_runtime.py |
| Four benchmark adapters and prediction provenance without post-initialization label access | src/dds_mamba/datasets.py; run_benchmark.py |

At a frame boundary the controller receives NumPy candidates detached from the
neural graph. This is intentional: crop index, gates, memory index, Kalman
update, cache, and committed state make exactly the same non-differentiable
decisions in training and inference. Gradients flow only through the selected
online-crop loss and, when sampled, the auxiliary teacher-crop loss.

The reference objective enables box, centre, decorrelation, temporal identity,
identity consistency, and projection-norm terms. Retrieval and query-norm
weights are explicitly zero. Teacher crops have unit weight and never update
the online state, cache, filter, QACU statistics, or RFMB.
