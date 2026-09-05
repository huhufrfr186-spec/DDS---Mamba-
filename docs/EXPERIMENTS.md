# Experimental protocol

The executable public protocols are LaSOT Protocol II (280 official test
sequences), Anti-UAV300 RGB Track 1 test-dev, WebUAV-3M V1.0 RGB Test (780
videos), and DUT-VTUAV-V visible-only test_LT. Prediction writers and strict
dataset adapters are present, but no public benchmark has been executed in
this workspace. Every public-result cell is therefore “not reproduced”.

The source runs three fixed training seeds for every immutable variant and
selects checkpoints by validation 21-threshold success AUC. For every metric,
the confidence-interval tool requires one per-sequence evaluator CSV for each
seed, averages each sequence over seeds, and then performs 10,000 sequence
resamples. Its paired-difference mode first forms matched full-minus-ablation
differences for every seed and sequence. The runtime tool performs 100
device-resident warm-up updates and reports median,
p90, memory, parameters, and profiler-reported FLOPs; it excludes I/O and
host-to-device transfer.

The only checked-in numeric experiment is DDS-Synthetic-LongTerm-v1. It was
rerun with the current V1 DDSOnlineState and its output lives in
results/synthetic_longterm_v1. It is a controller regression test and is not a
natural-image benchmark or a claim about any public dataset.
