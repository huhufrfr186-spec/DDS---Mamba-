# DDS-Mamba-v1 asset lock

This release fixes one encoder pair rather than treating encoders as a
tunable implementation detail. The template--search backbone is MAE
ViT-Base/16: its patch size makes a 256-pixel search image yield the required
16 by 16 feature grid, while its 128-pixel template yields an 8 by 8 grid.
The identity encoder is DINOv2 ViT-S/14, used only for frozen 224-pixel crop
embeddings and all RFMB/identity comparisons.

The complete URLs, byte sizes, SHA256 hashes, feature taps, resizing, and
normalization are locked in `manifests/dds_mamba_v1.yaml`. The downloader must
reject a weight file whose bytes or SHA256 differ from that file. Changing any
asset-lock field creates a new method version and requires a new experiment.

The selection uses the official MAE checkpoint release and official DINOv2
backbone release. The hash values are the published LFS SHA256 object IDs for
those exact `.pth` files; download-time verification is mandatory.
