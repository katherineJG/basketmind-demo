import pytest
import torch

from basketmind.utils.config import load_config, project_root
from basketmind.utils.device import describe_device, get_device
from basketmind.utils.seed import seed_everything

CONFIGS = ["data", "features", "graph", "model", "training", "evaluation", "inference"]


@pytest.mark.parametrize("name", CONFIGS)
def test_configs_load(name):
    assert isinstance(load_config(name), dict)


def test_blueprint_defaults():
    t, m = load_config("training"), load_config("model")
    assert (t["learning_rate"], t["batch_size"], t["early_stopping_patience"]) == (0.0003, 128, 7)
    te = m["temporal_encoder"]
    assert (te["embedding_dim"], te["num_layers"]) == (128, 2)


def test_seed_is_reproducible():
    seed_everything(1)
    a = torch.rand(3)
    seed_everything(1)
    assert torch.equal(a, torch.rand(3))


def test_device_detection():
    assert get_device("cpu").type == "cpu"
    assert get_device().type in {"cpu", "cuda"}
    assert "gpu_available" in describe_device()


def test_project_root_has_pyproject():
    assert (project_root() / "pyproject.toml").exists()
