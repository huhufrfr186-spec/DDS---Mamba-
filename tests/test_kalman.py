import numpy as np

from dds_mamba.kalman import ConstantVelocityKalman, KalmanConfig


def test_kalman_cholesky_update_keeps_covariance_symmetric_positive_definite():
    cfg = KalmanConfig()
    kf = ConstantVelocityKalman(np.array([0.5, 0.5, -2.0, -2.0]), cfg)
    for _ in range(10):
        kf.predict()
        assert kf.update(np.array([0.5, 0.5, -2.0, -2.0]), reliability=0.8)
        assert np.allclose(kf.P, kf.P.T)
        assert np.linalg.eigvalsh(kf.P).min() > 0


def test_kalman_reset_uses_reinitialization_covariance():
    cfg = KalmanConfig(reinitialize_multiplier=3.0)
    kf = ConstantVelocityKalman(np.array([0.5, 0.5, -2.0, -2.0]), cfg)
    kf.reset(np.array([0.4, 0.6, -1.0, -1.0]))
    assert np.allclose(kf.x[:4], [0.4, 0.6, -1.0, -1.0])
    assert np.allclose(kf.P, cfg.reinitialize_P)
