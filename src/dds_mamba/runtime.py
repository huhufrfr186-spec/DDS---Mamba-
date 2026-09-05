"""Manifest-locked neural inference coupled to the single online controller."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .data import crop_border, image_tensor, normalize
from .geometry import Box, iou, roi_ratio
from .model_v1 import DDSV1
from .online import Candidate, DDSOnlineState, Mode, OnlineConfig


def to_image(box: np.ndarray, crop: Box, side: float) -> Box:
    """Map crop-normalized centre-size coordinates to a full-image box."""
    return (
        crop[0] + (float(box[0]) - 0.5) * side,
        crop[1] + (float(box[1]) - 0.5) * side,
        float(box[2]) * side,
        float(box[3]) * side,
    )


def to_crop(box: Box, crop: Box, side: float) -> np.ndarray:
    """Map a full-image centre-size box to crop-normalized coordinates."""
    return np.asarray(
        [0.5 + (box[0] - crop[0]) / side, 0.5 + (box[1] - crop[1]) / side, box[2] / side, box[3] / side],
        dtype=np.float32,
    )


def quality(
    logits: torch.Tensor,
    box_crop: torch.Tensor,
    *,
    alignment_sigma: float = 2.0,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """The manifest-locked confidence-derived map quality and peak score."""
    score = logits.sigmoid()[0]
    height, width = score.shape
    probability = (score + eps) / (score.sum() + height * width * eps)
    y, x = torch.meshgrid(
        torch.arange(height, device=score.device, dtype=score.dtype) + 0.5,
        torch.arange(width, device=score.device, dtype=score.dtype) + 0.5,
        indexing="ij",
    )
    center = torch.stack([(probability * x).sum(), (probability * y).sum()])
    det_center = torch.stack([box_crop[0, 0] * width, box_crop[0, 1] * height])
    peak = float(score.max())
    entropy = float((1.0 + (probability * probability.log()).sum() / np.log(height * width)).clamp(0.0, 1.0))
    align = float(torch.exp(-((center - det_center).square().sum()) / (2.0 * alignment_sigma**2)))
    return float(np.clip(peak * entropy * align, 0.0, 1.0)), peak


@torch.no_grad()
def candidate_from_output(
    model: DDSV1,
    state: DDSOnlineState,
    image: torch.Tensor,
    predicted: Box,
    e_init: torch.Tensor,
    crop: Box,
    side: float,
    crop_index: int,
    output: dict[str, torch.Tensor],
) -> Candidate:
    """Build the one detached candidate type shared by training and inference."""
    raw_crop_box = output["box_crop"][0].detach().cpu().numpy().astype(np.float64)
    detection = to_image(raw_crop_box, crop, side)
    roundtrip_error = float(np.max(np.abs(to_crop(detection, crop, side) - raw_crop_box)))
    embedding = model.identity(image, image.new_tensor(detection))[0].detach().cpu().numpy()
    network = model.manifest["algorithm"]["network"]
    q_map, q_peak = quality(
        output["confidence_logits"].detach(),
        output["box_crop"].detach(),
        alignment_sigma=float(network["alignment_sigma_tokens"]),
        eps=float(network["quality_eps"]),
    )
    full_box = image.new_tensor(
        [[detection[0] / state.width, detection[1] / state.height, detection[2] / state.width, detection[3] / state.height]]
    )
    reinit = model.reinitialize_position(full_box)[0].detach().cpu().numpy()
    return Candidate(
        detection,
        q_map,
        q_peak,
        iou(detection, predicted),
        float(np.dot(embedding, e_init[0].detach().cpu().numpy())),
        embedding,
        output["position_proposal"][0].detach().cpu().numpy(),
        output["appearance_proposal"][0].detach().cpu().numpy(),
        reinit,
        roi_ratio(detection, state.width, state.height),
        crop_index,
        roundtrip_error,
    )


@dataclass
class DDSTracker:
    model: DDSV1
    device: torch.device
    online_overrides: Mapping[str, object] | None = None
    template_tokens: torch.Tensor | None = None
    state: DDSOnlineState | None = None
    e_init: torch.Tensor | None = None

    @torch.no_grad()
    def initialize_image(self, image: torch.Tensor, initial_xywh: np.ndarray) -> None:
        """Initialize from one device-resident ``[1,3,H,W]`` RGB tensor."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("DDS tracker initialization requires one [1,3,H,W] image")
        height, width = image.shape[-2:]
        x, y, box_width, box_height = map(float, initial_xywh)
        box = (x + box_width / 2.0, y + box_height / 2.0, box_width, box_height)
        template = normalize(crop_border(image, image.new_tensor(box), 128))
        self.template_tokens = self.model.template_features(template)
        self.e_init = self.model.identity(image, image.new_tensor(box))
        self.state = DDSOnlineState(
            box,
            self.e_init[0].cpu().numpy(),
            self.model.d_model,
            width,
            height,
            OnlineConfig.from_manifest(self.model.manifest, self.online_overrides),
        )

    @torch.no_grad()
    def initialize(self, image_path: str | Path, initial_xywh: np.ndarray) -> None:
        self.initialize_image(image_tensor(image_path).unsqueeze(0).to(self.device), initial_xywh)

    @torch.no_grad()
    def update_image(self, image: torch.Tensor) -> Box:
        """Track one preloaded device-resident frame without disk I/O."""
        assert self.state is not None and self.template_tokens is not None and self.e_init is not None
        predicted = self.state.predict()
        keys, utilities = self.state.memory_arrays()  # exactly M_{t-1}
        key_tensor = None if keys is None else torch.from_numpy(keys)[None].to(self.device)
        utility_tensor = None if utilities is None else torch.from_numpy(utilities)[None].to(self.device)
        position = torch.from_numpy(self.state.position)[None].to(self.device)
        appearance = torch.from_numpy(self.state.appearance)[None].to(self.device)
        enabled = torch.tensor([self.state.read_open()], dtype=torch.bool, device=self.device)
        candidates: list[Candidate] = []
        for crop, side, crop_index in self.state.crop_specs(predicted):
            search = normalize(crop_border(image, image.new_tensor(crop), 256))
            prior = torch.from_numpy(to_crop(self.state.box, crop, side))[None].to(self.device)
            output = self.model.forward_frame(
                self.template_tokens, search, prior, position, appearance, self.e_init, key_tensor, utility_tensor, enabled
            )
            candidates.append(candidate_from_output(self.model, self.state, image, predicted, self.e_init, crop, side, crop_index, output))
        return self.state.step(candidates, predicted)

    @torch.no_grad()
    def update(self, image_path: str | Path) -> Box:
        return self.update_image(image_tensor(image_path).unsqueeze(0).to(self.device))

    @property
    def is_lost(self) -> bool:
        return self.state is not None and self.state.mode is Mode.LOST
