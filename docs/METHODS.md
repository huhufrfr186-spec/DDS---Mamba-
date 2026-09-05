# DDS-Mamba-v1 release contract

The canonical method is the locked base manifest, the locked ablation manifest,
and the source paths identified in CODE_METHOD_V1.md. The full neural
implementation is included: model_v1.py constructs frozen encoders and
trainable heads, runtime.py constructs candidates, and train_v1.py unrolls the
same DDSOnlineState used by run_benchmark.py.

Reference invariants:

1. RFMB reads use only \(M_{t-1}\), sort score ties by insertion index, and are
   input context rather than recurrent-state assignment.
2. QACU is the reference normal-mode appearance update; recovery freezes
   appearance, QACU statistics, and RFMB.
3. Candidate construction, validity checks, Kalman quality, and discrete branch
   decisions are shared by training and inference.
4. Kalman uses Cholesky gain solve, Joseph covariance update, deterministic PSD
   repair, and manifest-locked reset covariance.
5. Every ablation is an immutable named variant and is recorded in checkpoints
   and prediction provenance.

The source-level tests cover geometry, RFMB duplicate ordering, Kalman PSD
invariants, controller transitions, and manifest-bound ablations. Neural
forward/backward execution requires the pinned PyTorch/CUDA environment and
verified frozen assets.
