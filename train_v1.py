"""Controller-driven DDS-Mamba-v1 training.

All online crops, discrete gates, RFMB entries, KF updates, and committed
states are created by the same ``DDSOnlineState`` used by ``run_benchmark.py``.
Ground truth enters only through losses and the explicitly auxiliary teacher
crop; it never changes the online state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from dds_mamba.assets import file_sha256
from dds_mamba.data import TrainingClip, TrainingClips, crop_border, image_tensor, lasot_sequences, normalize
from dds_mamba.geometry import Box, iou, roi_ratio
from dds_mamba.losses import LossWeights, objective, teacher_probability
from dds_mamba.model_v1 import DDSV1
from dds_mamba.online import Candidate, DDSOnlineState, OnlineConfig
from dds_mamba.runtime import candidate_from_output, to_crop
from dds_mamba.splits import lasot_train_validation
from dds_mamba.variants import load_ablation_spec


def _load_names(path: Path) -> list[str]:
    value = json.loads(path.read_text()) if path.suffix == ".json" else [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("train_split must be a JSON list or one sequence name per line")
    return value


def _training_names(cfg: dict) -> tuple[list[str], list[str]]:
    root = Path(cfg["lasot_root"])
    if cfg["train_split"] == "official_training_set":
        names = [line.strip() for line in (root / "training_set.txt").read_text().splitlines() if line.strip()]
        return lasot_train_validation(names)
    train = _load_names(Path(cfg["train_split"]))
    validation = _load_names(Path(cfg["validation_split"]))
    return train, validation


def _cxcywh(xywh: np.ndarray) -> Box:
    x, y, width, height = map(float, xywh)
    return x + width / 2.0, y + height / 2.0, width, height


def _contained(box: Box, crop: Box, tolerance: float = 2.0) -> bool:
    left, top = crop[0] - crop[2] / 2.0, crop[1] - crop[3] / 2.0
    right, bottom = crop[0] + crop[2] / 2.0, crop[1] + crop[3] / 2.0
    return (
        box[0] - box[2] / 2.0 >= left - tolerance
        and box[1] - box[3] / 2.0 >= top - tolerance
        and box[0] + box[2] / 2.0 <= right + tolerance
        and box[1] + box[3] / 2.0 <= bottom + tolerance
    )


def _coin(seed: int, purpose: str, epoch: int, sequence: str, frame_index: int) -> float:
    """A platform-independent Bernoulli draw for auxiliary training choices."""
    payload = f"{seed}|{purpose}|{epoch}|{sequence}|{frame_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _negative_crop(target: Box, side: float, width: int, height: int) -> Box | None:
    """Deterministic same-frame negative with IoU < .05, if geometry permits."""
    candidates = [(-side / 2.0, -side / 2.0), (width + side / 2.0, -side / 2.0), (-side / 2.0, height + side / 2.0), (width + side / 2.0, height + side / 2.0)]
    for center_x, center_y in candidates:
        crop = (float(center_x), float(center_y), side, side)
        if iou(target, crop) < 0.05:
            return crop
    return None


def _candidate(
    model: DDSV1,
    state: DDSOnlineState,
    image: torch.Tensor,
    predicted: Box,
    template_tokens: torch.Tensor,
    e_init: torch.Tensor,
    crop: Box,
    side: float,
    crop_index: int,
    keys: torch.Tensor | None,
    utilities: torch.Tensor | None,
    read_enabled: torch.Tensor,
) -> tuple[Candidate, dict[str, torch.Tensor]]:
    """One neural crop evaluation; returned candidate is deliberately detached."""
    search = normalize(crop_border(image, image.new_tensor(crop), 256))
    prior = torch.from_numpy(to_crop(state.box, crop, side))[None].to(image.device)
    position = torch.from_numpy(state.position)[None].to(image.device)
    appearance = torch.from_numpy(state.appearance)[None].to(image.device)
    out = model.forward_frame(template_tokens, search, prior, position, appearance, e_init, keys, utilities, read_enabled)
    return candidate_from_output(model, state, image, predicted, e_init, crop, side, crop_index, out), out


def _supervised_memory(keys: list[np.ndarray], weights: list[float], device: torch.device) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    if not keys:
        return None, None, torch.zeros(1, dtype=torch.bool, device=device)
    return (
        torch.from_numpy(np.stack(keys, axis=0).astype(np.float32))[None].to(device),
        torch.tensor(weights, dtype=torch.float32, device=device)[None],
        torch.ones(1, dtype=torch.bool, device=device),
    )


def _clip_loss(
    clip: TrainingClip,
    model: DDSV1,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    synthetic_absent_probability: float,
    teacher_weight: float,
    loss_weights: LossWeights,
    online_overrides: dict[str, object],
    run_seed: int,
) -> torch.Tensor:
    init_image = image_tensor(clip.template_frame).unsqueeze(0).to(device)
    height, width = init_image.shape[-2:]
    initial_box = _cxcywh(clip.template_box_xywh)
    template = normalize(crop_border(init_image, init_image.new_tensor(initial_box), 128))
    template_tokens = model.template_features(template)
    e_init = model.identity(init_image, init_image.new_tensor(initial_box))
    state = DDSOnlineState(
        initial_box,
        e_init[0].cpu().numpy(),
        model.d_model,
        width,
        height,
        OnlineConfig.from_manifest(model.manifest, online_overrides),
    )
    total = init_image.new_zeros(())
    previous_valid = False
    positive_keys: list[np.ndarray] = []
    positive_weights: list[float] = []
    use_supervised_bank = epoch < int(model.manifest["training"]["rfmb_teacher_epochs"] * total_epochs)
    probability = teacher_probability(epoch, total_epochs, float(model.manifest["training"]["teacher_forcing_epochs"]))
    for frame_index, (path, xywh, visible, absent, occluded) in enumerate(zip(clip.frames, clip.boxes_xywh, clip.visible, clip.absent, clip.occluded)):
        image = image_tensor(path).unsqueeze(0).to(device)
        target_box = _cxcywh(xywh)
        predicted = state.predict()
        memory_keys, memory_utilities = state.memory_arrays()
        keys = None if memory_keys is None else torch.from_numpy(memory_keys)[None].to(device)
        utilities = None if memory_utilities is None else torch.from_numpy(memory_utilities)[None].to(device)
        read_enabled = torch.tensor([state.read_open()], dtype=torch.bool, device=device)
        candidates: list[Candidate] = []
        outputs: dict[int, tuple[dict[str, torch.Tensor], Box, float]] = {}
        for crop, side, crop_index in state.crop_specs(predicted):
            candidate, out = _candidate(model, state, image, predicted, template_tokens, e_init, crop, side, crop_index, keys, utilities, read_enabled)
            candidates.append(candidate)
            outputs[crop_index] = (out, crop, side)
        selected = state.select_candidate(candidates)
        # In an active fallback, supervise the sole actual online crop; in lost
        # mode, use the Kalman crop if no candidate passed the eligibility gate.
        selected_index = selected.crop_index if selected is not None else min(outputs)
        out, crop, side = outputs[selected_index]
        target_crop = torch.from_numpy(to_crop(target_box, crop, side))[None].to(device).clamp(
            loss_weights.target_clip_eps, 1.0 - loss_weights.target_clip_eps
        )
        valid = torch.tensor([bool(visible) and not bool(absent) and not bool(occluded) and _contained(target_box, crop)], device=device)
        absent_tensor = torch.tensor([bool(absent)], device=device)
        occluded_tensor = torch.tensor([bool(occluded)], device=device)
        if bool(valid.item()):
            gt_identity = model.identity(image, image.new_tensor(target_box))
        else:
            gt_identity = torch.zeros_like(out["appearance_embedding"])
        previous_projection = None
        if previous_valid:
            previous_app = torch.from_numpy(state.appearance)[None].to(device)
            previous_projection = model.identity_projector(previous_app)
        frame_loss, _ = objective(
            out["box_crop"],
            out["confidence_logits"],
            out["position_proposal"],
            out["appearance_proposal"],
            out["identity_projection"],
            target_crop,
            gt_identity,
            valid,
            absent_tensor,
            occluded_tensor,
            previous_projection,
            loss_weights,
        )
        total = total + frame_loss

        # Deterministic synthetic absent supervision for box-only sources.  It
        # is an auxiliary crop and has no path to the online state.
        source_has_absence = bool(np.any(clip.absent))
        synth = (
            bool(visible)
            and not source_has_absence
            and _coin(run_seed, "synthetic-absent", epoch, clip.name, frame_index) < synthetic_absent_probability
        )
        if synth:
            negative = _negative_crop(target_box, side, width, height)
            if negative is not None:
                _, neg_out = _candidate(model, state, image, predicted, template_tokens, e_init, negative, side, 99, keys, utilities, read_enabled)
                zeros = torch.zeros(1, dtype=torch.bool, device=device)
                negative_loss, _ = objective(
                    neg_out["box_crop"], neg_out["confidence_logits"], neg_out["position_proposal"], neg_out["appearance_proposal"], neg_out["identity_projection"],
                    target_crop, torch.zeros_like(neg_out["appearance_embedding"]), zeros, torch.ones_like(zeros), zeros, None, loss_weights,
                )
                total = total + negative_loss

        # Teacher forcing is an auxiliary loss crop.  It sees either an online
        # positive bank (first 1/4 epochs) or M_{t-1}; it never reaches state.
        teach = bool(valid.item()) and _coin(run_seed, "teacher", epoch, clip.name, frame_index) < probability
        if teach:
            teacher_side = state.active_crop_side()
            teacher_crop = (target_box[0], target_box[1], teacher_side, teacher_side)
            if use_supervised_bank:
                teacher_keys, teacher_utilities, teacher_enabled = _supervised_memory(positive_keys[-5:], positive_weights[-5:], device)
            else:
                teacher_keys, teacher_utilities, teacher_enabled = keys, utilities, read_enabled
            _, teacher_out = _candidate(model, state, image, predicted, template_tokens, e_init, teacher_crop, teacher_side, 100, teacher_keys, teacher_utilities, teacher_enabled)
            teacher_target = torch.from_numpy(to_crop(target_box, teacher_crop, teacher_side))[None].to(device).clamp(
                loss_weights.target_clip_eps, 1.0 - loss_weights.target_clip_eps
            )
            teacher_identity = model.identity(image, image.new_tensor(target_box))
            teacher_loss, _ = objective(
                teacher_out["box_crop"], teacher_out["confidence_logits"], teacher_out["position_proposal"], teacher_out["appearance_proposal"], teacher_out["identity_projection"],
                teacher_target, teacher_identity, torch.ones(1, dtype=torch.bool, device=device), torch.zeros(1, dtype=torch.bool, device=device), torch.zeros(1, dtype=torch.bool, device=device), previous_projection, loss_weights,
            )
            total = total + teacher_weight * teacher_loss

        state.step(candidates, predicted)
        if state.last_active_commit and state.last_write and selected is not None:
            positive_keys.append(selected.embedding.copy())
            positive_weights.append(float(selected.q_map * selected.q_iou))
        previous_valid = bool(valid.item())
    return total / max(1, len(clip.frames))


@torch.no_grad()
def _validation_success_auc(
    model: DDSV1,
    sequences: list,
    device: torch.device,
    online_overrides: dict[str, object],
    threshold_count: int,
) -> float:
    """LaSOT-style 21-threshold success AUC used for checkpoint selection."""
    from dds_mamba.runtime import DDSTracker

    overlaps: list[float] = []
    model.eval()
    for sequence in sequences:
        tracker = DDSTracker(model, device, online_overrides)
        tracker.initialize(sequence.rgb_frames[0], sequence.boxes_xywh[0])
        for path, truth, visible in zip(sequence.rgb_frames[1:], sequence.boxes_xywh[1:], sequence.visible[1:]):
            prediction = tracker.update(path)
            if visible:
                overlaps.append(0.0 if tracker.is_lost else iou(prediction, _cxcywh(truth)))
    if not overlaps:
        raise ValueError("validation split contains no visible frames")
    overlap = np.asarray(overlaps, dtype=np.float64)
    if threshold_count < 2:
        raise ValueError("validation_success_threshold_count must be at least two")
    thresholds = np.linspace(0.0, 1.0, threshold_count, dtype=np.float64)
    return float(np.mean([(overlap >= threshold).mean() for threshold in thresholds]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_v1.yaml"))
    parser.add_argument("--variant", default="full")
    parser.add_argument("--ablation-manifest", type=Path, default=Path("manifests/dds_mamba_v1_ablations.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if int(cfg["batch_size"]) != 1:
        raise ValueError("DDS-Mamba-v1 uses native-resolution controller clips; batch_size must be 1")
    if int(cfg.get("workers", 0)) != 0:
        raise ValueError("DDS-Mamba-v1 native-image clips are loaded in the main process; workers must be 0")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names, validation_names = _training_names(cfg)
    dataset = TrainingClips(lasot_sequences(cfg["lasot_root"], names), cfg["clip_length"], seed, cfg["clips_per_sequence"])
    validation_sequences = lasot_sequences(cfg["lasot_root"], validation_names)
    variant = load_ablation_spec(cfg["manifest"], args.variant, args.ablation_manifest)
    model = DDSV1(cfg["manifest"], cfg["asset_dir"], variant=variant.name, network_overrides=dict(variant.network)).to(device)
    if int(cfg["clip_length"]) != int(model.manifest["training"]["clip_length"]):
        raise ValueError("config clip_length must match the immutable manifest")
    loss_weights = LossWeights(**model.manifest["training"]["losses"])
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    best_validation = float("-inf")
    for epoch in range(int(cfg["epochs"])):
        model.train()
        dataset.set_epoch(epoch)
        running = 0.0
        order = list(range(len(dataset)))
        random.Random(seed + epoch).shuffle(order)
        for index in order:
            loss = _clip_loss(
                dataset[index], model, device, epoch, int(cfg["epochs"]), float(model.manifest["training"]["synthetic_absent_probability"]),
                float(model.manifest["training"]["teacher_loss_weight"]), loss_weights,
                dict(variant.online), seed,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
            optimizer.step()
            running += float(loss.detach())
        validation = _validation_success_auc(
            model, validation_sequences, device, dict(variant.online), int(model.manifest["training"]["validation_success_threshold_count"])
        ) if (epoch + 1) % int(cfg["validate_every"]) == 0 else float("nan")
        record = {"epoch": epoch, "loss": running / len(order), "teacher_probability": teacher_probability(epoch, int(cfg["epochs"]), float(model.manifest["training"]["teacher_forcing_epochs"])), "validation_success_auc": validation}
        history.append(record)
        print(record, flush=True)
        checkpoint_meta = {
            "model": model.state_dict(), "manifest_sha256": file_sha256(cfg["manifest"]), "ablation_manifest_sha256": variant.manifest_sha256,
            "variant": variant.name, "variant_network": dict(variant.network), "variant_online": dict(variant.online), "config": cfg, "epoch": epoch,
        }
        torch.save(checkpoint_meta, output / "last.pt")
        if validation > best_validation:
            best_validation = validation
            checkpoint_meta["validation_success_auc"] = validation
            torch.save(checkpoint_meta, output / "best.pt")
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    environment = {
        "manifest_sha256": file_sha256(cfg["manifest"]),
        "ablation_manifest_sha256": variant.manifest_sha256,
        "variant": variant.name,
        "variant_description": variant.description,
        "config": cfg,
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": sys.version,
    }
    (output / "run_manifest.json").write_text(json.dumps(environment, indent=2) + "\n")


if __name__ == "__main__":
    main()
