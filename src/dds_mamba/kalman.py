"""Numerically guarded constant-velocity Kalman filter used by DDS-Mamba-v1."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class KalmanConfig:
    """All filter parameters are immutable entries of the base manifest.

    The values are diagonal in the normalized ``Encode`` coordinate system.
    They are not fitted from a test set and they are intentionally stored in
    the release manifest rather than inferred from an unavailable dataset.
    """

    dt: float = 1.0
    acceleration_std: tuple[float, float, float, float] = (0.01, 0.01, 0.005, 0.005)
    measurement_std: tuple[float, float, float, float] = (0.04, 0.04, 0.02, 0.02)
    initial_covariance: tuple[float, ...] = (0.04, 0.04, 0.02, 0.02, 0.10, 0.10, 0.05, 0.05)
    reinitialize_multiplier: float = 4.0
    reliability_floor: float = 1e-3
    covariance_jitter: float = 1e-9
    covariance_eigen_floor: float = 1e-9

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "KalmanConfig":
        values = dict(manifest.get("algorithm", {}).get("kalman", {}))
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown immutable Kalman setting(s): {sorted(unknown)}")
        for name in ("acceleration_std", "measurement_std"):
            if name in values:
                values[name] = tuple(float(x) for x in values[name])
        if "initial_covariance" in values:
            values["initial_covariance"] = tuple(float(x) for x in values["initial_covariance"])
        result = cls(**values)
        if len(result.acceleration_std) != 4 or len(result.measurement_std) != 4 or len(result.initial_covariance) != 8:
            raise ValueError("Kalman diagonal vectors must contain 4, 4, and 8 entries")
        if min(*result.acceleration_std, *result.measurement_std, *result.initial_covariance) <= 0:
            raise ValueError("Kalman standard deviations and initial covariance diagonal must be positive")
        if result.dt <= 0 or result.reliability_floor <= 0 or result.covariance_jitter <= 0 or result.covariance_eigen_floor <= 0:
            raise ValueError("Kalman scalar constants must be positive")
        return result

    @property
    def initial_P(self) -> np.ndarray:
        return np.diag(np.asarray(self.initial_covariance, dtype=np.float64))

    @property
    def reinitialize_P(self) -> np.ndarray:
        return self.initial_P * self.reinitialize_multiplier


class ConstantVelocityKalman:
    """Eight-dimensional Joseph-form filter with deterministic PSD repair."""

    def __init__(
        self,
        measurement: np.ndarray,
        config: KalmanConfig | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        self.cfg = config or KalmanConfig()
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (4,) or not np.isfinite(measurement).all():
            raise ValueError("Kalman initialization requires a finite four-vector")
        self.x = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        self.P = self._repair(self.cfg.initial_P if covariance is None else covariance)
        dt = self.cfg.dt
        self.F = np.block([[np.eye(4), dt * np.eye(4)], [np.zeros((4, 4)), np.eye(4)]])
        self.H = np.block([np.eye(4), np.zeros((4, 4))])
        sigma_a = np.diag(np.square(np.asarray(self.cfg.acceleration_std, dtype=np.float64)))
        self.Q = np.block(
            [
                [(dt**4 / 4.0) * sigma_a, (dt**3 / 2.0) * sigma_a],
                [(dt**3 / 2.0) * sigma_a, (dt**2) * sigma_a],
            ]
        )

    def _repair(self, covariance: np.ndarray) -> np.ndarray:
        """Return a finite symmetric positive-definite covariance deterministically."""
        value = np.asarray(covariance, dtype=np.float64)
        if value.shape != (8, 8) or not np.isfinite(value).all():
            return self.cfg.reinitialize_P.copy()
        value = (value + value.T) / 2.0
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(value)
        except np.linalg.LinAlgError:
            return self.cfg.reinitialize_P.copy()
        eigenvalues = np.maximum(eigenvalues, self.cfg.covariance_eigen_floor)
        repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
        repaired = (repaired + repaired.T) / 2.0
        # The Cholesky factor is the downstream numerical contract.  Add a
        # deterministic diagonal jitter if round-off still prevents it.
        for multiplier in (1.0, 10.0, 100.0, 1_000.0):
            candidate = repaired + np.eye(8) * self.cfg.covariance_jitter * multiplier
            try:
                np.linalg.cholesky(candidate)
                return candidate
            except np.linalg.LinAlgError:
                continue
        return self.cfg.reinitialize_P.copy()

    def reset(self, measurement: np.ndarray) -> None:
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (4,) or not np.isfinite(measurement).all():
            raise ValueError("Kalman reset requires a finite four-vector")
        self.x = np.r_[measurement, np.zeros(4, dtype=np.float64)]
        self.P = self.cfg.reinitialize_P.copy()

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self._repair(self.F @ self.P @ self.F.T + self.Q)
        return self.x[:4].copy()

    def update(self, measurement: np.ndarray, reliability: float) -> bool:
        measurement = np.asarray(measurement, dtype=np.float64)
        if measurement.shape != (4,) or not np.isfinite(measurement).all():
            return False
        reliability = max(float(reliability), self.cfg.reliability_floor)
        R = np.diag(np.square(np.asarray(self.cfg.measurement_std, dtype=np.float64))) / reliability
        S = (self.H @ self.P @ self.H.T + R)
        S = (S + S.T) / 2.0
        try:
            chol = np.linalg.cholesky(S + np.eye(4) * self.cfg.covariance_jitter)
            # solve(S, H P)^T without forming an inverse.
            right = self.H @ self.P
            gain = np.linalg.solve(chol.T, np.linalg.solve(chol, right)).T
        except np.linalg.LinAlgError:
            return False
        residual = measurement - self.H @ self.x
        posterior_x = self.x + gain @ residual
        identity = np.eye(8)
        posterior_P = (identity - gain @ self.H) @ self.P @ (identity - gain @ self.H).T + gain @ R @ gain.T
        posterior_P = self._repair(posterior_P)
        if not np.isfinite(posterior_x).all() or not np.isfinite(posterior_P).all():
            return False
        self.x, self.P = posterior_x, posterior_P
        return True
