from pathlib import Path

import torch

from dds_mamba.assets import load_locked_yaml
from dds_mamba.neural import sequence_stack
from dds_mamba.variants import load_ablation_spec


ROOT = Path(__file__).resolve().parents[1]


def test_every_operator_control_preserves_the_token_shape_and_is_causal_interface_compatible():
    x = torch.randn(2, 7, 256)
    for operator in ("mamba", "gru", "mlp", "transformer"):
        stack = sequence_stack(operator, 2, 256, 16, 2, 16, 4)
        assert stack(x).shape == x.shape


def test_operator_controls_do_not_allow_an_earlier_token_to_read_a_later_token():
    torch.manual_seed(7)
    x = torch.randn(1, 7, 256)
    changed_future = x.clone()
    changed_future[:, 4:] += torch.randn_like(changed_future[:, 4:])
    for operator in ("mamba", "gru", "mlp", "transformer"):
        stack = sequence_stack(operator, 2, 256, 16, 2, 16, 4).eval()
        assert torch.allclose(stack(x)[:, :4], stack(changed_future)[:, :4], atol=1e-5, rtol=1e-5)


def test_causal_control_manifest_is_locked_and_has_the_required_single_component_controls():
    controls = ROOT / "manifests" / "dds_mamba_v1_controls.yaml"
    document = load_locked_yaml(controls)
    required = {
        "full",
        "dual_gru_control",
        "dual_mlp_control",
        "dual_transformer_control",
        "position_mlp_control",
        "appearance_mlp_control",
        "no_kalman_only",
        "no_candidate_cache_only",
        "neutral_identity_evidence",
        "tied_active_state_only",
    }
    assert required <= set(document["variants"])
    for name in required:
        spec = load_ablation_spec(ROOT / "manifests" / "dds_mamba_v1.yaml", name, controls)
        assert spec.name == name
    assert document["variants"]["no_kalman_only"]["online"] == {"use_kalman": False, "use_candidate_cache": True}
    assert document["variants"]["no_candidate_cache_only"]["online"] == {"use_kalman": True, "use_candidate_cache": False}


def test_threshold_grid_is_locked_and_explicitly_forbids_public_test_access():
    grid = load_locked_yaml(ROOT / "configs" / "threshold_grid_v1.yaml")
    assert grid["selection"]["public_test_access"] == "forbidden"
    assert grid["selection"]["objective"] == "success_auc_21_thresholds"
    assert "recovery_iou" in grid["parameters"]
