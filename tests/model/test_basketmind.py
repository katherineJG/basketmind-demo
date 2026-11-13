import pytest
import torch
from tests.model_helpers import N_COUNTRIES, N_CUSTOMERS, N_ITEMS, make_batch, make_graph, model_cfg

from basketmind.models import BasketMindModel
from basketmind.training.checkpoint import load_checkpoint, save_checkpoint
from basketmind.utils.device import get_device
from basketmind.utils.seed import seed_everything

DEV = get_device()  # the real target (AMD GPU when available), so ROCm op support is exercised


def build(**kw) -> BasketMindModel:
    seed_everything(0)
    return BasketMindModel(model_cfg(**kw), N_ITEMS, N_CUSTOMERS, N_COUNTRIES).to(DEV)


def run(model, quantity=True):
    items = torch.randint(0, N_ITEMS, (3, 4), device=DEV) if quantity else None
    return model(make_batch(device=DEV), make_graph(device=DEV), items)


def test_forward_shapes_and_finite_on_default_device():
    out = run(build())
    assert out["basket_logits"].shape == (3, N_ITEMS) and out["fused"].shape == (3, 128)
    assert out["quantity"].shape == (3, 4) and "reversal" not in out  # reversal head disabled by default
    assert all(torch.isfinite(v).all() for v in out.values())


def test_every_trainable_parameter_receives_a_finite_gradient():
    model = build(reversal=True)
    out = run(model)
    assert out["reversal"].shape == (3, 4)
    (out["basket_logits"].mean() + out["quantity"].mean() + out["reversal"].mean()).backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, missing


@pytest.mark.parametrize(
    "on",
    [
        ("temporal",),
        ("graph",),
        ("customer",),
        ("temporal", "graph"),
        ("temporal", "customer"),
        ("graph", "customer"),
    ],
)
def test_branch_ablations_run_and_train(on):
    model = build(branches={k: k in on for k in ("temporal", "graph", "customer")})
    out = run(model)
    assert out["basket_logits"].shape == (3, N_ITEMS) and torch.isfinite(out["basket_logits"]).all()
    out["basket_logits"].mean().backward()
    disabled = {"temporal": "temporal.", "graph": "graph_encoder.", "customer": "customer."}
    for name, prefix in disabled.items():
        has = any(n.startswith(prefix) for n, _ in model.named_parameters())
        assert has == (name in on), (name, on)  # off branches own no parameters


def test_graph_branch_requires_a_snapshot():
    with pytest.raises(ValueError, match="GraphSnapshot"):
        build()(make_batch(device=DEV), None)
    build(branches={"graph": False})(make_batch(device=DEV), None)  # fine when the graph branch is off


def test_invalid_configs_rejected():
    with pytest.raises(ValueError, match="at least one"):
        build(branches={"temporal": False, "graph": False, "customer": False})
    cfg = model_cfg()
    cfg["fusion"]["output_dim"] = 64
    with pytest.raises(ValueError, match="dimensions"):
        BasketMindModel(cfg, N_ITEMS, N_CUSTOMERS, N_COUNTRIES)


def test_empty_graph_snapshot_runs_forward_and_backward():
    model = build()
    empty = make_graph(n_edges=0, device=DEV)  # early snapshots have no edges at all
    out = model(make_batch(device=DEV), empty)
    assert empty.edge_index.shape[1] == 0 and torch.isfinite(out["basket_logits"]).all()
    out["basket_logits"].mean().backward()  # the graph encoder keeps only its self terms; no crash


def test_graph_snapshot_changes_item_scores():
    model = build().eval()
    batch = make_batch(device=DEV)
    with torch.no_grad():
        a = model(batch, make_graph(device=DEV, seed=1))["basket_logits"]
        b = model(batch, make_graph(device=DEV, seed=2))["basket_logits"]
    assert not torch.allclose(a, b, atol=1e-5)


def test_maximum_history_length_runs():
    model = build()
    batch = make_batch(counts=(50, 7), device=DEV)  # 50 = max_sequence_length
    out = model(batch, make_graph(device=DEV))
    assert out["basket_logits"].shape == (2, N_ITEMS) and torch.isfinite(out["basket_logits"]).all()


def test_same_seed_same_model():
    a, b = build(), build()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))  # exact
    batch, graph = make_batch(device=DEV), make_graph(device=DEV)
    with torch.no_grad():
        a.eval(), b.eval()
        # Not bit-exact on a GPU: scatter/embedding_bag sum in a varying order (~1e-6 differences measured).
        assert torch.allclose(a(batch, graph)["basket_logits"], b(batch, graph)["basket_logits"], atol=1e-4)


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    model = build(reversal=True).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, graph = make_batch(device=DEV), make_graph(device=DEV)
    items = torch.randint(0, N_ITEMS, (3, 4), device=DEV)
    with torch.no_grad():
        before = model(batch, graph, items)
    path = save_checkpoint(tmp_path / "ckpt" / "m.pt", model, {"epoch": 3, "val_recall@10": 0.1}, opt)
    loaded, payload = load_checkpoint(path, DEV)
    loaded.eval()
    with torch.no_grad():
        after = loaded(batch, graph, items)
    for k in before:
        assert torch.allclose(before[k], after[k], atol=1e-6), k
    assert (
        payload["metadata"] == {"epoch": 3, "val_recall@10": 0.1} and payload["optimizer_state"] is not None
    )
    assert loaded.init_args["n_items"] == N_ITEMS and loaded.reversal_head is not None


def test_checkpoint_never_silently_overwrites(tmp_path):
    model = build()
    path = save_checkpoint(tmp_path / "m.pt", model)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model)
    save_checkpoint(path, model, overwrite=True)
