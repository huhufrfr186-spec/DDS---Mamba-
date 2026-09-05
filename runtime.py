"""Compatibility wrapper; the canonical runtime lives in ``src/dds_mamba``."""
from dds_mamba.runtime import DDSTracker, quality, to_crop, to_image

__all__ = ["DDSTracker", "quality", "to_crop", "to_image"]
