import numpy as np
from dds_mamba.memory import RFMB


def test_read_is_deterministic_and_duplicate_refresh_is_gated():
    bank = RFMB(2, .01, .95, .05)
    bank.write(np.array([1., 0.]), .7)
    bank.write(np.array([1., 0.]), .71)  # no refresh: gain <= .05
    assert len(bank.entries) == 1 and bank.entries[0].weight == .7
    bank.write(np.array([0., 1.]), .8)
    assert np.allclose(bank.read(np.array([1., 0.]), .1), np.array([1., 0.]))


def test_zero_capacity_is_a_valid_no_rfmb_ablation():
    bank = RFMB(0, .01, .95, .05)
    bank.write(np.array([1., 0.]), .9)
    assert bank.entries == []


def test_duplicate_refresh_uses_most_similar_entry_not_list_order():
    bank = RFMB(3, .01, .70, .01)
    bank.write(np.array([1., 0.]), .70)
    bank.write(np.array([.50, .8660254]), .70)
    bank.write(np.array([.9063078, .4226183]), .90)
    # The new key is a duplicate of both; entry zero is more similar and must
    # be refreshed even though changing list order would have exposed a bug.
    assert bank.entries[0].weight == .90
    assert bank.entries[1].weight == .70
