from pathlib import Path

from dds_mamba.variants import load_ablation_spec


ROOT = Path(__file__).resolve().parents[1]


def test_every_required_ablation_is_locked_and_loadable():
    names = {"full", "tied_state", "unprojected_gate", "no_qacu", "no_rfmb", "no_identity_gate", "no_kalman_cache"}
    loaded = {
        load_ablation_spec(ROOT / "manifests/dds_mamba_v1.yaml", name, ROOT / "manifests/dds_mamba_v1_ablations.yaml").name
        for name in names
    }
    assert loaded == names
