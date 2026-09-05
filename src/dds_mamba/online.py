"""The one and only persistent DDS-Mamba online state.

This module deliberately contains no torch tensors.  The neural model proposes
values; this controller owns the committed values.  Thus the training unroll and
the benchmark runner make exactly the same discrete decisions at frame
boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

from .geometry import Box, decode, encode, finite_box, iou, roi_ratio
from .kalman import ConstantVelocityKalman, KalmanConfig
from .memory import RFMB, cosine


class Mode(str, Enum):
    ACTIVE = "active"
    LOST = "lost"


@dataclass(frozen=True)
class OnlineConfig:
    # Section 3.6.1.  These values are part of the algorithm, not CLI flags.
    crop_factor: float = 4.0
    weak_crop_growth: float = 1.25
    lost_crop_ratio: float = 2.0
    lost_crop_growth: float = 1.5
    lost_crop_ratio_max: float = 6.0
    crop_min: float = 32.0
    crop_max_factor: float = 2.0
    crop_duplicate_epsilon: float = 1e-4
    roi_valid: float = 0.50
    fall_iou: float = 0.15
    active_peak: float = 0.60
    commit_quality: float = 0.30
    measurement_map: float = 0.30
    measurement_identity: float = 0.50
    active_identity: float = 0.55
    lost_limit: int = 3
    reacquire_map: float = 0.35
    reacquire_peak: float = 0.80
    reacquire_identity: float = 0.60
    reacquire_count: int = 2
    association_iou: float = 0.30
    association_embedding: float = 0.60
    recovery_iou: float = 0.25
    fallback_association_iou: float = 0.50
    fallback_reacquire_map: float = 0.20
    fallback_measurement_map: float = 0.15
    fallback_commit_quality: float = 0.25
    fallback_peak: float = 0.80
    fallback_fall_iou: float = 0.15
    beta_max: float = 0.95
    ema: float = 0.95
    memory_capacity: int = 100
    memory_decay: float = 0.01
    memory_duplicate: float = 0.95
    memory_refresh_delta: float = 0.05
    read_quality: float = 0.50
    idle_age: float = 0.10
    write_roi: float = 0.70
    write_quality: float = 0.70
    write_peak: float = 0.80
    write_template_identity: float = 0.60
    write_alignment_iou: float = 0.50
    # Exact non-neural controller constants that used to be hidden in code.
    qacu_warmup_commits: int = 20
    qacu_beta_min_warmup: float = 0.15
    qacu_beta_min_intercept: float = 0.25
    qacu_beta_min_slope: float = 0.20
    qacu_beta_min_lower: float = 0.05
    qacu_beta_min_upper: float = 0.25
    min_box_size: float = 1.0
    candidate_center_min_factor: float = -1.0
    candidate_center_max_factor: float = 2.0
    roundtrip_tolerance: float = 1e-5
    # These switches are used only by immutable ablation variants.
    shared_state: bool = False
    enable_qacu: bool = True
    enable_rfmb: bool = True
    use_identity: bool = True
    use_kalman: bool = True
    use_candidate_cache: bool = True
    # Controls used only by the follow-up causal ablations.  Defaults preserve
    # the V1 controller bit-for-bit for every existing manifest variant.
    tie_recovery_state: bool = True
    neutral_identity_evidence: bool = False
    kalman: KalmanConfig = KalmanConfig()

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> "OnlineConfig":
        values = dict(manifest.get("algorithm", {}).get("online", {}))
        if overrides:
            values.update(overrides)
        allowed = {field for field in cls.__dataclass_fields__} - {"kalman"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown immutable online setting(s): {sorted(unknown)}")
        return cls(**values, kalman=KalmanConfig.from_manifest(manifest))


@dataclass(frozen=True)
class Candidate:
    """A detached proposal for one crop; ``crop_index`` is the tie breaker."""
    box: Box
    q_map: float
    q_peak: float
    q_iou: float
    identity: float
    embedding: np.ndarray
    position: np.ndarray
    appearance: np.ndarray
    reinit_position: np.ndarray
    roi: float
    crop_index: int
    # ``to_crop(to_image(box))`` error in normalized crop coordinates.  It is
    # measured at candidate construction and guarded before any controller use.
    roundtrip_error: float = 0.0


class DDSOnlineState:
    """Atomic branch transition from Sections 3.3--3.4.

    ``box`` is always the last reliable box, never a weak-frame prediction.
    ``step`` returns the frame output box while keeping that distinction inside
    the state, which is what makes the following active crop unambiguous.
    """

    def __init__(
        self,
        initial_box: Box,
        e_init: np.ndarray | None,
        state_dim: int,
        width: int,
        height: int,
        config: OnlineConfig | None = None,
    ) -> None:
        self.cfg = config or OnlineConfig()
        self.width, self.height = int(width), int(height)
        self.mode = Mode.ACTIVE
        self.box = initial_box  # b_last
        self.kf = ConstantVelocityKalman(encode(initial_box, width, height), self.cfg.kalman)
        self.e_init = None if e_init is None else self._unit(e_init)
        self.position = np.zeros(state_dim, dtype=np.float32)
        self.appearance = np.zeros(state_dim, dtype=np.float32)
        self.memory = RFMB(
            self.cfg.memory_capacity,
            self.cfg.memory_decay,
            self.cfg.memory_duplicate,
            self.cfg.memory_refresh_delta,
        )
        self.commit_count = 0
        self.q_last = 1.0
        self.mu = 0.5
        self.weak_count = 0  # c_t
        self.lost_count = 0  # d_t
        self.cache: Candidate | None = None
        self.cache_count = 0
        self.last_active_commit = False
        self.last_write = False

    @staticmethod
    def _unit(value: np.ndarray) -> np.ndarray | None:
        value = np.asarray(value, dtype=np.float32)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(value).all() or norm < 1e-6:
            return None
        return value / norm

    @property
    def id_available(self) -> bool:
        return self.cfg.use_identity and self.e_init is not None

    def predict(self) -> Box:
        """Advance the full-image KF once, before crop construction."""
        if not self.cfg.use_kalman:
            return self.box
        proposal = self.kf.predict()
        if not np.isfinite(proposal).all() or not np.isfinite(self.kf.P).all():
            self.kf.reset(encode(self.box, self.width, self.height))
            return self.box
        decoded = decode(proposal, self.width, self.height)
        if not finite_box(decoded, self.cfg.min_box_size):
            self.kf.reset(encode(self.box, self.width, self.height))
            return self.box
        return decoded

    def active_crop_side(self) -> float:
        """The active-mode crop side, also used by loss-only teacher crops."""
        max_side = self.cfg.crop_max_factor * max(self.width, self.height)
        side = self.cfg.crop_factor * max(float(self.box[2]), float(self.box[3]), 1.0)
        return float(np.clip(side * self.cfg.weak_crop_growth ** self.weak_count, self.cfg.crop_min, max_side))

    def crop_specs(self, predicted_box: Box) -> list[tuple[Box, float, int]]:
        """Return ``(centered square box, side, crop_index)`` in deterministic order."""
        max_side = self.cfg.crop_max_factor * max(self.width, self.height)
        size = lambda b: self.cfg.crop_factor * max(float(b[2]), float(b[3]), 1.0)
        if self.mode is Mode.ACTIVE:
            side = self.active_crop_side()
            return [((self.box[0], self.box[1], float(side), float(side)), float(side), 0)]
        ratio = min(self.cfg.lost_crop_ratio_max, self.cfg.lost_crop_ratio * self.cfg.lost_crop_growth ** self.lost_count)
        side = float(np.clip(ratio * max(size(predicted_box), size(self.box)), self.cfg.crop_min, max_side))
        first = ((predicted_box[0], predicted_box[1], side, side), side, 0)
        second = ((self.box[0], self.box[1], side, side), side, 1)
        delta = max(abs(first[0][i] - second[0][i]) for i in range(3))
        return [first] if delta <= self.cfg.crop_duplicate_epsilon else [first, second]

    def read_open(self) -> bool:
        return (
            self.cfg.enable_rfmb
            and self.cfg.memory_capacity > 0
            and self.mode is Mode.ACTIVE
            and self.commit_count > 0
            and self.q_last < self.cfg.read_quality
            and bool(self.memory.entries)
            and self.id_available
        )

    def memory_arrays(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.cfg.enable_rfmb or not self.memory.entries:
            return None, None
        # Ascending insertion index is the secondary top-k order used by the
        # neural reader's stable sort.
        entries = sorted(self.memory.entries, key=lambda entry: entry.index)
        keys = np.stack([entry.key for entry in entries]).astype(np.float32)
        utilities = np.asarray([entry.utility(self.cfg.memory_decay) for entry in entries], dtype=np.float32)
        return keys, utilities

    def _identity(self, candidate: Candidate) -> float:
        if not self.id_available:
            return 1.0
        if self.cfg.neutral_identity_evidence:
            return 1.0
        candidate_embedding = self._unit(candidate.embedding)
        if candidate_embedding is None:
            return float("-inf")
        memory_score = max(
            (
                max(0.0, cosine(candidate_embedding, entry.key)) * entry.utility(self.cfg.memory_decay)
                for entry in self.memory.entries
            ),
            default=float("-inf"),
        )
        return max(float(candidate.identity), memory_score)

    def _valid(self, candidate: Candidate) -> bool:
        center_x, center_y = candidate.box[:2]
        if (
            not finite_box(candidate.box, self.cfg.min_box_size)
            or not np.isfinite(candidate.roi)
            or candidate.roi < self.cfg.roi_valid
            or not np.isfinite(candidate.roundtrip_error)
            or candidate.roundtrip_error > self.cfg.roundtrip_tolerance
            or center_x < self.cfg.candidate_center_min_factor * self.width
            or center_x > self.cfg.candidate_center_max_factor * self.width
            or center_y < self.cfg.candidate_center_min_factor * self.height
            or center_y > self.cfg.candidate_center_max_factor * self.height
        ):
            return False
        return (not self.id_available) or self._unit(candidate.embedding) is not None

    def _measurement_ok(self, candidate: Candidate) -> bool:
        if not self._valid(candidate):
            return False
        if not self.id_available:
            return candidate.q_map >= self.cfg.fallback_measurement_map and candidate.q_iou >= self.cfg.fallback_fall_iou
        return (
            candidate.q_map >= self.cfg.measurement_map
            and candidate.q_iou >= self.cfg.fall_iou
            and self._identity(candidate) >= self.cfg.measurement_identity
        )

    def _active_commit_ok(self, candidate: Candidate) -> bool:
        if not self._measurement_ok(candidate):
            return False
        quality = candidate.q_map * candidate.q_iou
        if not self.id_available:
            return candidate.q_peak >= self.cfg.fallback_peak and quality >= self.cfg.fallback_commit_quality
        return (
            candidate.q_peak >= self.cfg.active_peak
            and quality >= self.cfg.commit_quality
            and self._identity(candidate) >= self.cfg.active_identity
        )

    def _eligible_lost(self, candidate: Candidate) -> bool:
        if not self._valid(candidate):
            return False
        if not self.id_available:
            return candidate.q_peak >= self.cfg.fallback_peak and candidate.q_map >= self.cfg.fallback_reacquire_map
        return (
            candidate.q_peak >= self.cfg.reacquire_peak
            and candidate.q_map >= self.cfg.reacquire_map
            and self._identity(candidate) >= self.cfg.reacquire_identity
        )

    def _rank(self, candidates: list[Candidate]) -> Candidate | None:
        if not candidates:
            return None
        if self.id_available:
            key = lambda c: (c.q_map * np.clip((self._identity(c) + 1.0) / 2.0, 0.0, 1.0), -c.crop_index)
        else:
            key = lambda c: (c.q_map, -c.crop_index)
        return max(candidates, key=key)

    def _associate(self, candidate: Candidate) -> bool:
        if self.cache is None or iou(candidate.box, self.cache.box) < (
            self.cfg.association_iou if self.id_available else self.cfg.fallback_association_iou
        ):
            return False
        if not self.id_available:
            return True
        return cosine(candidate.embedding, self.cache.embedding) >= self.cfg.association_embedding

    def _quality(self, candidate: Candidate) -> float:
        return float(np.clip(candidate.q_map * candidate.q_iou, 0.0, 1.0))

    def _update_kf(self, candidate: Candidate) -> tuple[bool, Box]:
        if not self.cfg.use_kalman:
            return True, candidate.box
        r_id = np.clip((self._identity(candidate) + 1.0) / 2.0, 0.0, 1.0) if self.id_available else 1.0
        success = self.kf.update(encode(candidate.box, self.width, self.height), self._quality(candidate) * r_id)
        return success, (decode(self.kf.x[:4], self.width, self.height) if success else self.box)

    def _qacu(self, candidate: Candidate) -> None:
        quality = self._quality(candidate)
        if self.cfg.enable_qacu:
            beta_min = self.cfg.qacu_beta_min_warmup if self.commit_count < self.cfg.qacu_warmup_commits else float(
                np.clip(
                    self.cfg.qacu_beta_min_intercept - self.cfg.qacu_beta_min_slope * self.mu,
                    self.cfg.qacu_beta_min_lower,
                    self.cfg.qacu_beta_min_upper,
                )
            )
            beta = (1.0 - quality) * self.cfg.beta_max + quality * beta_min
            next_appearance = beta * self.appearance + (1.0 - beta) * candidate.appearance
        else:
            next_appearance = candidate.appearance.copy()
        if self.cfg.shared_state:
            # State-tying ablation: the QACU output is the only recurrent
            # vector supplied to both neural branches on the next frame.
            self.position = next_appearance.copy()
            self.appearance = next_appearance.copy()
        else:
            self.appearance = next_appearance
            self.position = candidate.position.copy()
        self.mu = self.cfg.ema * self.mu + (1.0 - self.cfg.ema) * quality
        self.q_last = quality
        self.commit_count += 1

    def _write_if_allowed(self, candidate: Candidate, output: Box) -> bool:
        if not self.cfg.enable_rfmb or not self.id_available:
            return False
        identity_to_template = 1.0 if self.cfg.neutral_identity_evidence else cosine(candidate.embedding, self.e_init)
        ok = (
            roi_ratio(output, self.width, self.height) >= self.cfg.write_roi
            and iou(output, candidate.box) >= self.cfg.write_alignment_iou
            and self._quality(candidate) >= self.cfg.write_quality
            and candidate.q_peak >= self.cfg.write_peak
            and identity_to_template >= self.cfg.write_template_identity
        )
        if ok:
            self.memory.write(candidate.embedding, self._quality(candidate))
        return bool(ok)

    def select_candidate(self, candidates: list[Candidate]) -> Candidate | None:
        """The selected detached proposal, exposed for the training loss only."""
        if self.mode is Mode.ACTIVE:
            return self._rank([candidate for candidate in candidates if self._valid(candidate)])
        return self._rank([candidate for candidate in candidates if self._eligible_lost(candidate)])

    def step(self, candidates: list[Candidate], predicted_box: Box) -> Box:
        """Commit one branch and return this frame's output, without test labels."""
        self.last_active_commit = False
        self.last_write = False
        if self.mode is Mode.ACTIVE:
            selected = self.select_candidate(candidates)
            if selected is not None and self._measurement_ok(selected):
                kf_ok, posterior = self._update_kf(selected)
                if kf_ok and self._active_commit_ok(selected):
                    self._qacu(selected)
                    self.box = posterior
                    self.weak_count = 0
                    self.lost_count = 0
                    self.cache = None
                    self.cache_count = 0
                    self.memory.age_all(1.0)
                    self.last_active_commit = True
                    self.last_write = self._write_if_allowed(selected, posterior)
                    return posterior
                # Weak measurement: retain neural states and b_last, but expose the KF posterior.
                self.memory.age_all(self.cfg.idle_age)
                self.weak_count += 1
                if self.weak_count >= self.cfg.lost_limit:
                    self.mode, self.lost_count = Mode.LOST, 0
                self.cache, self.cache_count = None, 0
                return posterior if kf_ok else self.box
            # Active fallback.  The KF has already supplied ``predicted_box``.
            self.memory.age_all(self.cfg.idle_age)
            self.weak_count += 1
            if self.weak_count >= self.cfg.lost_limit:
                self.mode, self.lost_count = Mode.LOST, 0
            self.cache, self.cache_count = None, 0
            return predicted_box

        eligible = self.select_candidate(candidates)
        self.memory.age_all(self.cfg.idle_age)
        if eligible is None:
            self.cache, self.cache_count = None, 0
            self.lost_count += 1
            return predicted_box
        self.cache_count = self.cache_count + 1 if self.cfg.use_candidate_cache and self._associate(eligible) else 1
        self.cache = eligible
        required = self.cfg.reacquire_count if self.cfg.use_candidate_cache else 1
        if self.cache_count < required:
            self.lost_count += 1
            return predicted_box
        kf_ok, posterior = self._update_kf(eligible)
        # Standard recovery only when the KF agrees; otherwise detector reinitialization.
        output = posterior if kf_ok and eligible.q_iou >= self.cfg.recovery_iou else eligible.box
        if output is eligible.box and self.cfg.use_kalman:
            self.kf.reset(encode(eligible.box, self.width, self.height))
        self.box = output
        self.position = eligible.reinit_position.copy()
        if self.cfg.shared_state and self.cfg.tie_recovery_state:
            self.appearance = eligible.reinit_position.copy()
        self.mode = Mode.ACTIVE
        self.weak_count = self.lost_count = 0
        self.cache = None
        self.cache_count = 0
        # Appearance, QACU statistics, and RFMB are intentionally frozen on recovery.
        return output
