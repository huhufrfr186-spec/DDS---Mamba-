import numpy as np
from dds_mamba.geometry import crop_to_image, image_to_crop, iou


def test_crop_round_trip_and_iou():
    box = (100., 80., 20., 10.)
    crop = image_to_crop(box, 100., 80., 200.)
    assert np.allclose(crop_to_image(crop, 100., 80., 200.), box)
    assert iou(box, box) == 1.0
