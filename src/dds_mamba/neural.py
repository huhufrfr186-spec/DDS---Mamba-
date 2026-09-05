"""Trainable DDS-Mamba neural front end.

This module implements the two recurrent streams and a compact selective SSM
block in pure PyTorch.  It intentionally has no dependency on mamba-ssm, so a
release can be trained from source with one pinned PyTorch version.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def _sin2d(h: int, w: int, d: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if d % 4:
        raise ValueError("d_model must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    omega = torch.exp(torch.arange(d // 4, device=device, dtype=dtype) * (-math.log(10_000) / max(d // 4 - 1, 1)))
    return torch.cat([(x[..., None] * omega).sin(), (x[..., None] * omega).cos(),
                      (y[..., None] * omega).sin(), (y[..., None] * omega).cos()], dim=-1).reshape(1, h * w, d)


class SelectiveSSM(nn.Module):
    """Causal selective state-space layer with input-dependent time steps."""
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dt_rank: int = 16, convolution_kernel: int = 4) -> None:
        super().__init__(); self.d_model, self.d_inner, self.d_state = d_model, d_model * expand, d_state
        if convolution_kernel < 1:
            raise ValueError("selective SSM convolution kernel must be positive")
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, convolution_kernel, groups=self.d_inner, padding=convolution_kernel - 1)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner)); self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: Tensor) -> Tensor:
        b, length, _ = x.shape; u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2))
        params = self.x_proj(u); dt0, B, C = params.split((self.dt_proj.in_features, self.d_state, self.d_state), dim=-1)
        dt = F.softplus(self.dt_proj(dt0)) + 1e-4; A = -torch.exp(self.A_log).unsqueeze(0)
        state = x.new_zeros(b, self.d_inner, self.d_state); ys = []
        for t in range(length):
            dti, ui = dt[:, t], u[:, t]
            state = torch.exp(dti[..., None] * A) * state + (dti[..., None] * B[:, t, None, :] * ui[..., None])
            ys.append((state * C[:, t, None, :]).sum(-1) + self.D * ui)
        y = torch.stack(ys, dim=1) * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int, dt_rank: int, convolution_kernel: int = 4) -> None:
        super().__init__(); self.norm = nn.LayerNorm(d_model); self.ssm = SelectiveSSM(d_model, d_state, expand, dt_rank, convolution_kernel)
    def forward(self, x: Tensor) -> Tensor: return x + self.ssm(self.norm(x))


class MambaStack(nn.Module):
    def __init__(self, layers: int, d_model: int, d_state: int, expand: int, dt_rank: int, convolution_kernel: int = 4) -> None:
        super().__init__(); self.layers = nn.ModuleList([MambaBlock(d_model, d_state, expand, dt_rank, convolution_kernel) for _ in range(layers)])
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        return x


class GRUStack(nn.Module):
    """Causal recurrent control branch with the same token interface as MambaStack."""

    def __init__(self, layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, num_layers=layers, batch_first=True, dropout=0.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.gru(x)[0]


class CausalMLPBlock(nn.Module):
    """A causal prefix-MLP control without a state-space recurrence.

    Each output token sees the mean of the transformed prefix only.  This gives
    the MLP control the same causal token ordering as the selective SSM while
    deliberately removing input-dependent state dynamics.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        value = self.in_proj(self.norm(x))
        prefix = value.cumsum(dim=1)
        count = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        return x + self.out_proj(F.gelu(prefix / count))


class CausalMLPStack(nn.Module):
    def __init__(self, layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CausalMLPBlock(d_model) for _ in range(layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CausalTransformerStack(nn.Module):
    """Causal Transformer control with the DDS token order left unchanged."""

    def __init__(self, layers: int, d_model: int, **_: int) -> None:
        super().__init__()
        if d_model % 8:
            raise ValueError("the Transformer control requires d_model divisible by eight")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.shape[1]
        # True entries are masked by MultiheadAttention, hence this prevents a
        # token from reading a later token while retaining the Mamba ordering.
        causal_mask = torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1)
        return self.encoder(x, mask=causal_mask)


def sequence_stack(
    operator: str,
    layers: int,
    d_model: int,
    d_state: int,
    expand: int,
    dt_rank: int,
    convolution_kernel: int,
) -> nn.Module:
    """Build one immutable branch operator for a controlled Mamba replacement."""
    factories = {
        "mamba": MambaStack,
        "gru": GRUStack,
        "mlp": CausalMLPStack,
        "transformer": CausalTransformerStack,
    }
    if operator not in factories:
        raise ValueError(f"unknown branch operator {operator!r}; expected one of {sorted(factories)}")
    return factories[operator](
        layers,
        d_model,
        d_state=d_state,
        expand=expand,
        dt_rank=dt_rank,
        convolution_kernel=convolution_kernel,
    )


class TemplateSearchEncoder(nn.Module):
    """Patch encoder with template-conditioned search features.

    `freeze_encoder=True` makes this a frozen feature extractor after loading a
    released checkpoint; training from scratch sets it to False.
    """
    def __init__(self, d_model: int, patch: int = 16, freeze_encoder: bool = False) -> None:
        super().__init__(); self.patch = nn.Conv2d(3, d_model, patch, stride=patch); self.condition = nn.Linear(d_model, 2 * d_model)
        if freeze_encoder:
            for p in self.parameters(): p.requires_grad_(False)
    def forward(self, template: Tensor, search: Tensor) -> tuple[Tensor, int, int]:
        t = self.patch(template).flatten(2).transpose(1, 2).mean(1); s = self.patch(search); b, d, h, w = s.shape
        scale, bias = self.condition(t).chunk(2, -1); s = s.flatten(2).transpose(1, 2)
        return s * (1 + torch.tanh(scale[:, None])) + bias[:, None], h, w


class IdentityEncoder(nn.Module):
    """Small identity encoder; load and freeze a released checkpoint for final runs."""
    def __init__(self, dim: int, freeze: bool = False) -> None:
        super().__init__(); self.body = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2), nn.GELU(), nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1)); self.head = nn.Linear(64, dim)
        if freeze:
            for p in self.parameters(): p.requires_grad_(False)
    def forward(self, image: Tensor) -> Tensor: return _norm(self.head(self.body(image).flatten(1)))


def bounded_gate(
    logits: Tensor,
    g_min: float = .25,
    g_max: float = 4.0,
    eta: float = 0.5,
    temperature: float = 1.0,
    iterations: int = 24,
) -> Tensor:
    """Project a gate to [g_min,g_max] with unit mean by differentiable bisection."""
    if g_min <= 0 or g_max < g_min or eta < 0 or temperature <= 0:
        raise ValueError("invalid bounded spatial-gate constants")
    n = logits.shape[-1]
    u = 1 + eta * (n * (logits / temperature).softmax(-1) - 1)
    lo = (u - g_max).amin(-1, keepdim=True); hi = (u - g_min).amax(-1, keepdim=True)
    for _ in range(iterations):
        mid = (lo + hi) / 2; value = (u - mid).clamp(g_min, g_max).sum(-1, keepdim=True)
        lo, hi = torch.where(value > n, mid, lo), torch.where(value > n, hi, mid)
    return (u - (lo + hi) / 2).clamp(g_min, g_max)


class FrameOutput(NamedTuple):
    box: Tensor; confidence: Tensor; position: Tensor; appearance_proposal: Tensor; embedding: Tensor


@dataclass
class NeuralConfig:
    d_model: int = 256; d_state: int = 16; expand: int = 2; dt_rank: int = 16; pos_layers: int = 2; app_layers: int = 4; patch: int = 16; freeze_encoder: bool = False; freeze_identity: bool = False


class DDSMambaNet(nn.Module):
    """End-to-end neural DDS-Mamba front end used by training and inference."""
    def __init__(self, cfg: NeuralConfig = NeuralConfig()) -> None:
        super().__init__(); self.cfg = cfg; d = cfg.d_model
        self.encoder = TemplateSearchEncoder(d, cfg.patch, cfg.freeze_encoder); self.identity_encoder = IdentityEncoder(d, cfg.freeze_identity)
        self.box_pe = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d)); self.pos_in = nn.Linear(2 * d, d)
        self.position = MambaStack(cfg.pos_layers, d, cfg.d_state, cfg.expand, cfg.dt_rank); self.appearance = MambaStack(cfg.app_layers, d, cfg.d_state, cfg.expand, cfg.dt_rank)
        self.box_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 4)); self.gate_head = nn.Linear(d, 256)
        self.app_token = nn.Linear(d, d); self.confidence = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)); self.embedding = nn.Linear(d, d)

    def forward_frame(self, template: Tensor, search: Tensor, prior_box: Tensor, pos_state: Tensor, app_state: Tensor, read_context: Tensor | None = None) -> FrameOutput:
        features, h, w = self.encoder(template, search); b, n, d = features.shape
        pos_tokens = torch.stack([pos_state, self.pos_in(torch.cat([features.mean(1), self.box_pe(prior_box)], -1))], 1)
        pos_next = self.position(pos_tokens)[:, 1]; box = self.box_head(pos_next).sigmoid()
        gates = bounded_gate(self.gate_head(pos_next)[..., :n]); gated = features * gates[..., None]
        pe = _sin2d(h, w, d, search.device, search.dtype); null = gated.new_zeros(b, d) if read_context is None else read_context
        app_tokens = torch.cat([self.app_token(app_state)[:, None], gated + pe, null[:, None]], 1); app_next = self.appearance(app_tokens)[:, -1]
        conf = self.confidence(torch.cat([gated, app_next[:, None].expand(-1, n, -1)], -1)).squeeze(-1).view(b, h, w).sigmoid()
        return FrameOutput(box, conf, pos_next, app_next, _norm(self.embedding(app_next)))

    def forward_clip(self, template: Tensor, searches: Tensor, boxes: Tensor | None = None, teacher_forcing: bool = True) -> dict[str, Tensor]:
        b, t = searches.shape[:2]; d = self.cfg.d_model; pos, app = searches.new_zeros(b, d), searches.new_zeros(b, d); prior = boxes[:, 0] if boxes is not None else searches.new_tensor([.5, .5, .2, .2]).expand(b, 4)
        out_boxes, maps, embeddings = [], [], []
        for i in range(t):
            o = self.forward_frame(template, searches[:, i], prior, pos, app); pos = o.position
            quality = o.confidence.amax(dim=(1, 2)).detach(); beta = (1 - quality) * .95 + quality * .15; app = beta[:, None] * app + (1 - beta[:, None]) * o.appearance_proposal
            prior = boxes[:, i] if (teacher_forcing and boxes is not None) else o.box.detach(); out_boxes.append(o.box); maps.append(o.confidence); embeddings.append(o.embedding)
        return {"boxes": torch.stack(out_boxes, 1), "confidence": torch.stack(maps, 1), "embeddings": torch.stack(embeddings, 1), "template_embedding": self.identity_encoder(template)}
