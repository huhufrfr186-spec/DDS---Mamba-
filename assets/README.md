# Frozen assets

Do not commit checkpoint binaries to the release package.  The immutable
`manifests/dds_mamba_v1.yaml` records the official download URL, exact byte
count, and SHA-256 digest for each frozen encoder.

The training and benchmark entry points call `obtain_verified`, which downloads
a missing asset and rejects an existing file whose bytes or digest do not match
the manifest.  If an earlier or partial download is present, remove that file
before starting a run; it must never be renamed or used as an alternative
checkpoint.

After the first verified download, run:

```bash
PYTHONPATH=src python scripts/verify_manifest_assets.py \
  --manifest manifests/dds_mamba_v1.yaml --assets assets
```
