import pandas as pd
import pytest
import torch
from tests.model_helpers import make_batch
from torch import nn

from basketmind.graph import GraphSnapshot
from basketmind.models.batch import Batch, pad_flat
from basketmind.models.customer import CustomerContextEncoder, FusionNetwork
from basketmind.models.graph_encoder import ProductGraphEncoder, WeightedSAGEConv
from basketmind.models.temporal import TemporalEncoder
from basketmind.utils.config import load_config

CFG = load_config("model")
D = CFG["item_embedding_dim"]
FEAT = CFG["basket_feature_dim"]


def temporal():
    torch.manual_seed(0)
    return TemporalEncoder(CFG["temporal_encoder"], FEAT, D).eval()  # eval: dropout off, so exact comparisons


def test_pad_flat_layout_and_mask():
    flat = torch.arange(6.0).repeat_interleave(2).reshape(6, 2)  # rows 0..5
    padded, mask = pad_flat(flat, torch.tensor([2, 1, 3]))
    assert padded.shape == (3, 3, 2) and mask.sum().item() == 6
    assert padded[0, 1, 0] == 1 and padded[1, 0, 0] == 2 and padded[2, 2, 0] == 5
    assert padded[0, 2].abs().sum() == 0 and padded[1, 1:].abs().sum() == 0  # padding rows are zero


def test_temporal_is_causal():
    enc, counts = temporal(), torch.tensor([5])
    vecs, feats = torch.randn(5, D), torch.randn(5, FEAT)
    with torch.no_grad():
        base, _ = enc.encode_sequence(vecs, feats, counts)
        vecs2, feats2 = vecs.clone(), feats.clone()
        vecs2[3:] += 1.0  # change the 4th and 5th baskets
        feats2[3:] -= 1.0
        changed, _ = enc.encode_sequence(vecs2, feats2, counts)
    assert torch.allclose(base[:, :3], changed[:, :3], atol=1e-6)  # earlier positions cannot see the future
    assert not torch.allclose(base[:, 3:], changed[:, 3:], atol=1e-3)


def test_temporal_padding_does_not_leak():
    enc = temporal()
    vecs, feats = torch.randn(7, D), torch.randn(7, FEAT)
    with torch.no_grad():
        together = enc(vecs, feats, torch.tensor([5, 2]))  # second example is padded to length 5
        alone = enc(vecs[5:], feats[5:], torch.tensor([2]))
        longer_first = enc(vecs[:5], feats[:5], torch.tensor([5]))
    assert torch.allclose(together[1], alone[0], atol=1e-5)
    assert torch.allclose(together[0], longer_first[0], atol=1e-5)


def test_temporal_readout_is_last_real_basket_and_finite():
    enc, counts = temporal(), torch.tensor([3, 1])
    vecs, feats = torch.randn(4, D), torch.randn(4, FEAT)
    with torch.no_grad():
        states, _ = enc.encode_sequence(vecs, feats, counts)
        out = enc(vecs, feats, counts)
    assert out.shape == (2, D) and torch.isfinite(out).all()
    assert torch.equal(out[0], states[0, 2]) and torch.equal(out[1], states[1, 0])


def test_temporal_rejects_history_longer_than_max():
    enc, n = temporal(), CFG["temporal_encoder"]["max_sequence_length"] + 1
    with pytest.raises(ValueError, match="exceeds"):
        enc(torch.randn(n, D), torch.randn(n, FEAT), torch.tensor([n]))


def test_weighted_sage_matches_hand_computation():
    conv = WeightedSAGEConv(2, 2)
    with torch.no_grad():
        conv.lin_neigh.weight.copy_(torch.eye(2))
        conv.lin_self.weight.zero_()
        conv.lin_self.bias.zero_()
    x = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[1, 2], [0, 0]])  # nodes 1 and 2 point at node 0
    out = conv(x, edge_index, torch.tensor([1.0, 3.0]))
    assert torch.allclose(out[0], (1 * x[1] + 3 * x[2]) / 4)  # weighted mean of the neighbours
    assert out[1:].abs().sum() == 0  # no incoming edges: nothing aggregated, no NaN


def test_graph_encoder_uses_edge_weights_and_handles_isolated_nodes():
    torch.manual_seed(0)
    enc = ProductGraphEncoder(CFG["graph_encoder"]).eval()
    x = torch.randn(6, D)
    ei = torch.tensor([[1, 2, 3], [0, 0, 4]])
    g1 = GraphSnapshot(ei, torch.tensor([1.0, 1.0, 1.0]), pd.Timestamp("2011-01-01"), 6, 1)
    g2 = GraphSnapshot(ei, torch.tensor([1.0, 5.0, 1.0]), pd.Timestamp("2011-01-01"), 6, 1)
    with torch.no_grad():
        a, b = enc(x, g1), enc(x, g2)
    assert a.shape == (6, D) and torch.isfinite(a).all()  # node 5 is isolated
    assert not torch.allclose(a[0], b[0], atol=1e-4)  # reweighting node 0's neighbours changes it
    with pytest.raises(ValueError, match="nodes"):
        enc(torch.randn(5, D), g1)


def test_customer_encoder_unseen_id_and_numeric_features_matter():
    torch.manual_seed(0)
    enc = CustomerContextEncoder(CFG["customer_encoder"], n_customers=10, n_countries=4).eval()
    num = torch.randn(2, CFG["customer_encoder"]["numeric_dim"])
    ids, country = torch.tensor([0, 0]), torch.tensor([0, 1])  # id 0 = unseen customer
    with torch.no_grad():
        out = enc(ids, country, num)
        out2 = enc(ids, country, num + 1.0)
    assert out.shape == (2, 64) and enc.out_dim == 64 and torch.isfinite(out).all()
    assert not torch.allclose(out, out2, atol=1e-4)  # the numeric features alone move the output


def test_fusion_shape_and_input_dim():
    fusion = FusionNetwork(128 + 128 + 64, 128, 0.2)
    out = fusion([torch.randn(3, 128), torch.randn(3, 128), torch.randn(3, 64)])
    assert out.shape == (3, 128)
    assert isinstance(fusion.net[3], nn.Dropout) and isinstance(fusion.net[2], nn.LayerNorm)


def test_batch_validate_catches_inconsistent_batches():
    b = make_batch()
    b.validate(FEAT, CFG["customer_encoder"]["numeric_dim"])
    bad = make_batch()
    bad.basket_counts = torch.tensor([3, 0, 5])  # an example without history
    with pytest.raises(AssertionError):
        bad.validate(FEAT, CFG["customer_encoder"]["numeric_dim"])
    wrong = Batch(**{**b.__dict__, "basket_feats": b.basket_feats[:-1]})
    with pytest.raises(AssertionError):
        wrong.validate(FEAT, CFG["customer_encoder"]["numeric_dim"])
