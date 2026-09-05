import numpy as np

from dds_mamba.online import Candidate, DDSOnlineState, Mode


def candidate(map_q=.9, peak=.9, iou_q=.9, ident=.9, crop_index=0):
    return Candidate(
        (100.0, 100.0, 40.0, 40.0),
        map_q,
        peak,
        iou_q,
        ident,
        np.array([1.0, 0.0]),
        np.ones(2),
        np.ones(2),
        np.array([2.0, 2.0]),
        1.0,
        crop_index,
    )


def test_active_commit_qacu_and_lost_recovery_freeze_contract():
    tracker = DDSOnlineState((100.0, 100.0, 40.0, 40.0), np.array([1.0, 0.0]), 2, 400, 300)
    tracker.step([candidate()], tracker.predict())
    committed = tracker.appearance.copy()
    assert tracker.mode is Mode.ACTIVE and tracker.commit_count == 1 and tracker.last_active_commit
    for _ in range(3):
        tracker.step([candidate(.01, .01, .01, -1.0)], tracker.predict())
    assert tracker.mode is Mode.LOST and np.allclose(tracker.appearance, committed)
    tracker.step([candidate()], tracker.predict())
    tracker.step([candidate()], tracker.predict())
    assert tracker.mode is Mode.ACTIVE and tracker.commit_count == 1
    assert np.allclose(tracker.appearance, committed)  # Recovery resets position only.


def test_lost_tie_break_prefers_kalman_crop_index_zero():
    tracker = DDSOnlineState((100.0, 100.0, 40.0, 40.0), np.array([1.0, 0.0]), 2, 400, 300)
    for _ in range(3):
        tracker.step([candidate(.01, .01, .01, -1.0)], tracker.predict())
    first = candidate(crop_index=0)
    second = candidate(crop_index=1)
    assert tracker.select_candidate([second, first]).crop_index == 0
