"""Literal implementation of the enabled Section 3.5 loss terms."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LossWeights:
    box: float = 1.0
    ctr: float = 1.0
    decorr: float = 0.1
    temp: float = 0.1
    consist: float = 0.1
    norm: float = 0.01
    retr: float = 0.0  # Explicitly disabled in the reference release.
    qnorm: float = 0.0  # Explicitly disabled in the reference release.
    l1: float = 5.0
    giou: float = 2.0
    temp_margin: float = 0.15
    id_margin: float = 0.5
    absent_weight: float = 1.0
    occluded_weight: float = 1.0
    ctr_sigma: float = 1.5
    focal_gamma: float = 2.0
    focal_beta: float = 4.0
    probability_eps: float = 1e-6
    cosine_eps: float = 1e-6
    target_clip_eps: float = 1e-3


def _mean_masked(value: Tensor, mask: Tensor) -> Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def safe_cosine(left: Tensor, right: Tensor, eps: float = 1e-6) -> Tensor:
    return (left * right).sum(-1) / left.norm(dim=-1).clamp_min(eps) / right.norm(dim=-1).clamp_min(eps)


def xyxy(box: Tensor) -> Tensor:
    return torch.cat([box[..., :2] - box[..., 2:] / 2.0, box[..., :2] + box[..., 2:] / 2.0], dim=-1)


def giou(predicted: Tensor, target: Tensor) -> Tensor:
    pred, truth = xyxy(predicted), xyxy(target)
    ix0, iy0 = torch.maximum(pred[..., 0], truth[..., 0]), torch.maximum(pred[..., 1], truth[..., 1])
    ix1, iy1 = torch.minimum(pred[..., 2], truth[..., 2]), torch.minimum(pred[..., 3], truth[..., 3])
    intersection = (ix1 - ix0).clamp_min(0) * (iy1 - iy0).clamp_min(0)
    pred_area = (pred[..., 2] - pred[..., 0]).clamp_min(0) * (pred[..., 3] - pred[..., 1]).clamp_min(0)
    truth_area = (truth[..., 2] - truth[..., 0]).clamp_min(0) * (truth[..., 3] - truth[..., 1]).clamp_min(0)
    union = pred_area + truth_area - intersection
    overlap = intersection / union.clamp_min(1e-6)
    cx0, cy0 = torch.minimum(pred[..., 0], truth[..., 0]), torch.minimum(pred[..., 1], truth[..., 1])
    cx1, cy1 = torch.maximum(pred[..., 2], truth[..., 2]), torch.maximum(pred[..., 3], truth[..., 3])
    cover = (cx1 - cx0).clamp_min(1e-6) * (cy1 - cy0).clamp_min(1e-6)
    return overlap - (cover - union) / cover


def centre_targets(box: Tensor, height: int, width: int, sigma: float = 1.5) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=box.device, dtype=box.dtype) + 0.5,
        torch.arange(width, device=box.device, dtype=box.dtype) + 0.5,
        indexing="ij",
    )
    center_x, center_y = box[..., 0, None, None] * width, box[..., 1, None, None] * height
    return torch.exp(-((x - center_x).square() + (y - center_y).square()) / (2.0 * sigma**2))


def objective(
    predicted_box: Tensor,
    confidence_logits: Tensor,
    position_proposal: Tensor,
    appearance_proposal: Tensor,
    identity_projection: Tensor,
    ground_truth_box: Tensor,
    ground_truth_identity: Tensor,
    visible_and_contained: Tensor,
    absent: Tensor,
    occluded: Tensor,
    previous_committed_identity_projection: Tensor | None,
    weights: LossWeights = LossWeights(),
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return exactly the six enabled terms in the displayed objective.

    The target map is zero on absent/occluded/uncontained frames.  Box,
    separation, temporal, identity, and norm terms are only enabled by ``v_t``.
    """
    visible = visible_and_contained.float()
    absent = absent.bool()
    occluded = occluded.bool()
    probability = confidence_logits.sigmoid().clamp(weights.probability_eps, 1.0 - weights.probability_eps)
    positive = centre_targets(ground_truth_box, *confidence_logits.shape[-2:], sigma=weights.ctr_sigma) * visible[..., None, None]
    sample_weight = torch.where(absent, probability.new_tensor(weights.absent_weight), probability.new_ones(absent.shape))
    sample_weight = torch.where(occluded, probability.new_tensor(weights.occluded_weight), sample_weight)
    focal = -(
        positive * (1.0 - probability).pow(weights.focal_gamma) * probability.log()
        + (1.0 - positive).pow(weights.focal_beta) * probability.pow(weights.focal_gamma) * (1.0 - probability).log()
    ).mean(dim=(-1, -2))
    ctr = (sample_weight * focal).mean()
    box = _mean_masked(
        weights.l1 * (predicted_box - ground_truth_box).abs().sum(-1) + weights.giou * (1.0 - giou(predicted_box, ground_truth_box)),
        visible,
    )
    decorr = _mean_masked(safe_cosine(position_proposal, appearance_proposal, weights.cosine_eps).square(), visible)
    consist = _mean_masked(1.0 - safe_cosine(identity_projection, ground_truth_identity, weights.cosine_eps), visible)
    norm = _mean_masked(torch.relu(identity_projection.new_tensor(weights.id_margin) - identity_projection.norm(dim=-1)).square(), visible)
    if previous_committed_identity_projection is None:
        temporal = identity_projection.new_zeros(())
    else:
        temporal = _mean_masked(
            torch.relu(identity_projection.new_tensor(weights.temp_margin) - safe_cosine(identity_projection, previous_committed_identity_projection, weights.cosine_eps)),
            visible,
        )
    total = (
        weights.box * box
        + weights.ctr * ctr
        + weights.decorr * decorr
        + weights.temp * temporal
        + weights.consist * consist
        + weights.norm * norm
    )
    return total, {"box": box, "ctr": ctr, "decorr": decorr, "temp": temporal, "consist": consist, "norm": norm}


def teacher_probability(epoch: int, total_epochs: int, fraction: float = 0.5) -> float:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("teacher forcing fraction must be in (0, 1]")
    cutoff = max(1, int(fraction * total_epochs))
    return 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * epoch / cutoff)).item()) if epoch < cutoff else 0.0
