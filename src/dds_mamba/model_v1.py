"""Manifest-locked neural DDS-Mamba v1.

Only the small projection, dual-state Mamba stacks, heads, and memory-reader
parameters are trainable. Both visual encoders are verified frozen assets.
"""
from __future__ import annotations

from pathlib import Path
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .assets import asset, load_asset_lock, obtain_verified
from .neural import bounded_gate, sequence_stack, _sin2d, _norm
from .data import crop_border, normalize


class FrozenTimmViT(nn.Module):
    def __init__(self, model_name: str, checkpoint: Path, feature: str, dynamic: bool) -> None:
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=False, num_classes=0, dynamic_img_size=dynamic)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        if not isinstance(state, dict):
            raise RuntimeError(f"frozen checkpoint has no state dictionary: {checkpoint}")
        # MAE releases decoder/mask keys; DINOv2 releases a mask token.  They are
        # not part of the locked feature tap.  Every encoder key must match.
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith(("decoder", "mask_token", "head", "fc_norm"))}
        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        allowed_missing = ("head", "fc_norm")
        bad_missing = [x for x in missing if not x.startswith(allowed_missing)]
        if bad_missing or unexpected:
            raise RuntimeError(f"frozen checkpoint is incompatible: missing={bad_missing}, unexpected={unexpected}")
        self.feature=feature
        for parameter in self.parameters(): parameter.requires_grad_(False)
        self.eval()
    def train(self, mode: bool = True): super().train(False); return self
    def forward_tokens(self, x: Tensor) -> Tensor:
        tokens=self.vit.forward_features(x)
        # timm returns either B,N,D tokens or B,D pooled features depending on model.
        if tokens.ndim != 3: raise RuntimeError("locked ViT must expose token features")
        return tokens[:, 1:]  # Explicitly exclude the class token for MAE.
    def forward_cls(self, x: Tensor) -> Tensor:
        tokens=self.vit.forward_features(x)
        if tokens.ndim != 3: return tokens
        return tokens[:, 0]


class MemoryReader(nn.Module):
    def __init__(
        self,
        state_dim: int,
        identity_dim: int,
        *,
        topk: int,
        threshold: float,
        temperature: float,
        kappa: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(state_dim + identity_dim, identity_dim)
        self.value = nn.Linear(identity_dim, identity_dim)
        self.out = nn.Linear(identity_dim, state_dim)
        self.null = nn.Parameter(torch.zeros(state_dim))
        self.topk, self.threshold, self.temperature, self.kappa, self.eps = topk, threshold, temperature, kappa, eps

    def forward(
        self,
        app: Tensor,
        e_init: Tensor,
        keys: Tensor | None,
        utility: Tensor | None,
        enabled: Tensor,
    ) -> Tensor:
        """Read fixed pre-frame entries with score/index deterministic ordering."""
        if keys is None or utility is None or not bool(enabled.any()):
            return self.null[None].expand_as(app)
        query = _norm(self.query(torch.cat([app, e_init], -1)))
        score = torch.einsum("bd,bkd->bk", query, keys) * utility
        valid = score >= self.threshold
        masked = score.masked_fill(~valid, float("-inf"))
        k = min(self.topk, masked.shape[1])
        # ``memory_arrays`` orders keys by insertion index.  Stable descending
        # sorting therefore implements score descending, insertion index
        # ascending ties on every supported PyTorch release.
        indices = torch.argsort(masked, dim=-1, descending=True, stable=True)[..., :k]
        values = masked.gather(-1, indices)
        finite = torch.isfinite(values)
        safe_values = torch.where(finite, values, torch.zeros_like(values))
        weights = F.softmax(safe_values / self.temperature, -1) * finite.to(safe_values.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        selected = keys.gather(1, indices[..., None].expand(-1, -1, keys.shape[-1]))
        read = torch.einsum("bk,bkd->bd", weights, selected)
        a = self.value(read)
        projection = self.out(a)
        scale = (self.kappa * app.norm(dim=-1) / a.norm(dim=-1).clamp_min(self.eps)).clamp(max=1.0)
        accepted = enabled.bool() & valid.any(dim=-1)
        return torch.where(accepted[:, None], projection * scale[:, None], self.null[None])


class DDSV1(nn.Module):
    def __init__(
        self,
        manifest_path: str | Path,
        asset_dir: str | Path,
        d_model: int = 256,
        *,
        variant: str = "full",
        network_overrides: dict | None = None,
    ) -> None:
        super().__init__(); self.manifest=load_asset_lock(manifest_path); self.d_model=d_model; self.variant = variant
        network = self.manifest.get("algorithm", {}).get("network", {})
        expected = {"d_model": 256, "position_layers": 2, "appearance_layers": 4, "ssm_state": 16, "ssm_expand": 2, "ssm_convolution_kernel": 4, "dt_rank": 16, "min_normalized_box": 0.001}
        if any(network.get(key) != value for key, value in expected.items()) or d_model != expected["d_model"]:
            raise ValueError("DDS-Mamba-v1 architecture is locked by the immutable manifest")
        mae=asset(self.manifest,"template_search_encoder"); dino=asset(self.manifest,"identity_encoder")
        if not mae.frozen or not dino.frozen or mae.implementation != "timm==1.0.19 model_name=vit_base_patch16_224" or dino.implementation != "timm==1.0.19 model_name=vit_small_patch14_dinov2.lvd142m":
            raise ValueError("DDS-Mamba-v1 only accepts its locked frozen encoder pair")
        overrides = dict(network_overrides or {})
        unknown_overrides = set(overrides) - {"bound_gate", "position_operator", "appearance_operator"}
        if unknown_overrides:
            raise ValueError(f"unknown network ablation setting(s): {sorted(unknown_overrides)}")
        self.bound_gate = bool(overrides.get("bound_gate", True))
        self.position_operator = str(overrides.get("position_operator", "mamba"))
        self.appearance_operator = str(overrides.get("appearance_operator", "mamba"))
        self.gate_min = float(network["gate_min"])
        self.gate_max = float(network["gate_max"])
        self.gate_temperature = float(network["gate_temperature"])
        self.gate_eta = float(network["gate_eta"])
        self.min_normalized_box = float(network["min_normalized_box"])
        memory = self.manifest.get("algorithm", {}).get("memory_reader", {})
        required_memory = {"topk", "threshold", "temperature", "kappa", "eps"}
        if set(memory) != required_memory:
            raise ValueError("immutable manifest must define all memory_reader settings")
        self.template_vit=FrozenTimmViT("vit_base_patch16_224",obtain_verified(mae,asset_dir),mae.feature_tap,True)
        self.identity_vit=FrozenTimmViT("vit_small_patch14_dinov2.lvd142m",obtain_verified(dino,asset_dir),dino.feature_tap,False)
        self.search_proj=nn.Linear(768,d_model); self.template_affine=nn.Linear(768,2*d_model); self.box_pe=nn.Sequential(nn.Linear(4,d_model),nn.GELU(),nn.Linear(d_model,d_model)); self.pos_in=nn.Linear(2*d_model,d_model)
        self.position=sequence_stack(self.position_operator,int(network["position_layers"]),d_model,int(network["ssm_state"]),int(network["ssm_expand"]),int(network["dt_rank"]),int(network["ssm_convolution_kernel"]))
        self.appearance=sequence_stack(self.appearance_operator,int(network["appearance_layers"]),d_model,int(network["ssm_state"]),int(network["ssm_expand"]),int(network["dt_rank"]),int(network["ssm_convolution_kernel"]))
        self.box_head=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Linear(d_model,4)); self.gate_head=nn.Linear(d_model,256); self.app_token=nn.Linear(d_model,d_model); self.confidence=nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU(),nn.Linear(d_model,1)); self.identity_projector=nn.Linear(d_model,384); self.memory_reader=MemoryReader(d_model,384,topk=int(memory["topk"]),threshold=float(memory["threshold"]),temperature=float(memory["temperature"]),kappa=float(memory["kappa"]),eps=float(memory["eps"]))
        self.position_reinitializer=nn.Sequential(nn.Linear(4,d_model),nn.GELU(),nn.Linear(d_model,d_model))
    @torch.no_grad()
    def identity(self, image: Tensor, boxes_cxcywh: Tensor) -> Tensor:
        return _norm(self.identity_vit.forward_cls(normalize(crop_border(image,boxes_cxcywh,224))))
    @torch.no_grad()
    def template_features(self, template: Tensor) -> Tensor: return self.template_vit.forward_tokens(template)
    def reinitialize_position(self, full_image_cxcywh: Tensor) -> Tensor:
        """The trainable 4→256→256 recovery reset in the Methods contract."""
        return self.position_reinitializer(full_image_cxcywh)
    def forward_frame(self, template_tokens: Tensor, search: Tensor, prior_crop_box: Tensor, pos_state: Tensor, app_state: Tensor, e_init: Tensor, memory_keys: Tensor | None, memory_utility: Tensor | None, read_enabled: Tensor) -> dict[str,Tensor]:
        raw=self.template_vit.forward_tokens(search); b,n,_=raw.shape
        if n != 256:
            raise RuntimeError(f"locked 256x256 search must expose 16x16=256 patches, got {n}")
        features=self.search_proj(raw); scale,bias=self.template_affine(template_tokens.mean(1)).chunk(2,-1); features=features*(1+torch.tanh(scale[:,None]))+bias[:,None]
        pos=self.position(torch.stack([pos_state,self.pos_in(torch.cat([features.mean(1),self.box_pe(prior_crop_box)],-1))],1))[:,1]
        raw_box = self.box_head(pos)
        center = raw_box[..., :2].sigmoid()
        size = self.min_normalized_box + (1.0 - self.min_normalized_box) * raw_box[..., 2:].sigmoid()
        box = torch.cat([center, size], dim=-1)
        logits_gate = self.gate_head(pos)
        if self.bound_gate:
            gate = bounded_gate(logits_gate, self.gate_min, self.gate_max, self.gate_eta, self.gate_temperature)
        else:
            alpha = F.softmax(logits_gate / self.gate_temperature, dim=-1)
            gate = 1.0 + self.gate_eta * (n * alpha - 1.0)
        gated=features*gate[...,None]; h=w=16; context=self.memory_reader(app_state,e_init,memory_keys,memory_utility,read_enabled); appearance=self.appearance(torch.cat([self.app_token(app_state)[:,None],gated+_sin2d(h,w,self.d_model,search.device,search.dtype),context[:,None]],1))[:,-1]; logits=self.confidence(torch.cat([gated,appearance[:,None].expand(-1,n,-1)],-1)).squeeze(-1).view(b,h,w); identity_raw=self.identity_projector(appearance)
        return {"box_crop":box,"confidence_logits":logits,"position_proposal":pos,"appearance_proposal":appearance,"identity_projection":identity_raw,"appearance_embedding":_norm(identity_raw),"read_context":context}
