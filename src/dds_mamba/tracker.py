"""Atomic DDS-Mamba controller; neural inference is intentionally injected upstream."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import numpy as np

from .config import DDSConfig
from .geometry import Box, encode, decode, finite_box, iou, roi_ratio
from .kalman import ConstantVelocityKalman
from .memory import RFMB, cosine


class Mode(str, Enum):
    ACTIVE = "active"
    LOST = "lost"


@dataclass(frozen=True)
class Candidate:
    box: Box
    map_quality: float
    peak_quality: float
    iou_quality: float
    identity: float | None
    embedding: np.ndarray
    position_state: np.ndarray
    appearance_proposal: np.ndarray


@dataclass(frozen=True)
class Snapshot:
    mode: Mode
    box: Box
    commits: int
    lost_frames: int
    candidate_count: int
    memory_size: int
    appearance_state: np.ndarray


class DDSController:
    def __init__(self, initial_box: Box, appearance_state: np.ndarray, config: DDSConfig, initial_embedding: np.ndarray | None = None) -> None:
        if not finite_box(initial_box): raise ValueError("initial_box must be finite with positive size")
        self.cfg, self.mode, self.box = config, Mode.ACTIVE, initial_box
        self.position = np.zeros_like(appearance_state); self.appearance = appearance_state.astype(float).copy()
        self.kalman = ConstantVelocityKalman(encode(initial_box, config.image_width, config.image_height))
        self.memory = RFMB(config.memory_capacity, config.memory_decay, config.duplicate_threshold, config.refresh_delta)
        self.initial_embedding = None if initial_embedding is None else initial_embedding / max(np.linalg.norm(initial_embedding), config.eps)
        self.commits = self.lost_frames = self.candidate_count = 0
        self.cached_box: Box | None = None; self.cached_embedding: np.ndarray | None = None

    def _valid(self, c: Candidate) -> bool:
        return finite_box(c.box) and roi_ratio(c.box, self.cfg.image_width, self.cfg.image_height) >= .50

    def _identity(self, c: Candidate) -> float:
        """Use the stronger of initial-template and RFMB evidence.

        The RFMB is only an identity cue for gating.  It never writes to the
        recurrent appearance state, which is updated exclusively by QACU.
        """
        if self.initial_embedding is None:
            return 1.0
        direct = -2.0 if c.identity is None else float(c.identity)
        memory = max((max(0.0, cosine(c.embedding, e.key)) * e.utility(self.cfg.memory_decay)
                      for e in self.memory.entries), default=-2.0)
        return max(direct, memory)

    def _measurement_ok(self, c: Candidate) -> bool:
        return self._valid(c) and c.map_quality >= self.cfg.measurement_map and c.iou_quality >= self.cfg.fall_iou and self._identity(c) >= self.cfg.measurement_identity

    def _active_ok(self, c: Candidate) -> bool:
        q = c.map_quality * c.iou_quality
        return self._measurement_ok(c) and c.peak_quality >= self.cfg.active_peak and q >= self.cfg.commit_quality and self._identity(c) >= self.cfg.active_identity

    def _associate(self, c: Candidate) -> bool:
        if self.cached_box is None or self.cached_embedding is None: return False
        return iou(c.box, self.cached_box) >= self.cfg.association_iou and cosine(c.embedding, self.cached_embedding) >= self.cfg.association_embedding

    def _commit_active(self, c: Candidate) -> Box:
        quality = c.map_quality * c.iou_quality
        reliable = quality * max(0.0, (self._identity(c) + 1) / 2)
        if not self.kalman.update(encode(c.box, self.cfg.image_width, self.cfg.image_height), reliable, self.cfg.eps):
            raise RuntimeError("Kalman update failed")
        self.box = decode(self.kalman.x[:4], self.cfg.image_width, self.cfg.image_height)
        beta_min = .15 if self.commits < 20 else float(np.clip(.25 - .20 * getattr(self, "mean_quality", .5), .05, .25))
        beta = (1 - quality) * self.cfg.beta_max + quality * beta_min
        self.appearance = beta * self.appearance + (1 - beta) * c.appearance_proposal
        self.position = c.position_state.copy(); self.commits += 1
        self.mean_quality = self.cfg.ema * getattr(self, "mean_quality", .5) + (1 - self.cfg.ema) * quality
        self.memory.age_all(1.0)
        if roi_ratio(self.box, self.cfg.image_width, self.cfg.image_height) >= .70 and quality >= .70 and c.peak_quality >= .80 and self._identity(c) >= .60:
            self.memory.write(c.embedding, quality)
        self.lost_frames = self.candidate_count = 0; self.cached_box = self.cached_embedding = None
        return self.box

    def step(self, candidate: Candidate) -> Snapshot:
        """Apply exactly one controller transition after the neural front end evaluated the frame."""
        self.kalman.predict()  # prediction is never a memory or appearance-state assignment
        if self.mode is Mode.ACTIVE and self._active_ok(candidate):
            self._commit_active(candidate)
            return self.snapshot()
        self.memory.age_all(self.cfg.idle_age)
        if self.mode is Mode.ACTIVE:
            self.mode = Mode.LOST; self.lost_frames = 0
        else:
            self.lost_frames += 1
        eligible = self._valid(candidate) and candidate.map_quality >= self.cfg.reacquire_map and self._identity(candidate) >= self.cfg.reacquire_identity
        if eligible:
            self.candidate_count = self.candidate_count + 1 if self._associate(candidate) else 1
            self.cached_box, self.cached_embedding = candidate.box, candidate.embedding.copy()
            if self.candidate_count >= self.cfg.reacquire_count:
                self.box = candidate.box; self.position = candidate.position_state.copy(); self.mode = Mode.ACTIVE
                self.lost_frames = self.candidate_count = 0; self.cached_box = self.cached_embedding = None
        else:
            self.candidate_count = 0; self.cached_box = self.cached_embedding = None
        return self.snapshot()

    def snapshot(self) -> Snapshot:
        return Snapshot(self.mode, self.box, self.commits, self.lost_frames, self.candidate_count, len(self.memory.entries), self.appearance.copy())
