from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from basketmind.utils.device import get_device

DEV = get_device()
CUSTOMER = "1"


def test_cached_graph_vectors_give_the_same_outputs_as_encoding_every_time(service_parts):
    service, _, _ = service_parts
    batch, _ = service._prepare(CUSTOMER, service.as_of)
    items = np.array([0, 1])
    with torch.no_grad():
        full = service.model(batch, service._graph_gpu, torch.as_tensor(items, device=DEV)[None, :])
    cached = service._forward(batch, items=items)
    for key in ("basket_logits", "quantity", "fused"):
        assert torch.allclose(full[key], cached[key], atol=1e-6), key


def test_requests_do_not_run_the_graph_encoder_but_a_supplied_graph_still_does(service_parts):
    service, _, _ = service_parts
    calls = []
    service.model.graph_encoder.register_forward_hook(lambda *_: calls.append(1))
    service.recommend(CUSTOMER, k=2)
    service.recommend(CUSTOMER, k=2, exclude_history=True)
    service.explain(CUSTOMER, service.task.items[0])
    assert calls == []
    service.score_customer(CUSTOMER, graph=service.graph)  # a different point in time must be encoded
    assert len(calls) == 1


def test_parallel_requests_match_sequential_ones(service_parts):
    service, _, _ = service_parts
    one = service.recommend(CUSTOMER, k=2)
    with ThreadPoolExecutor(6) as pool:
        many = list(pool.map(lambda _: service.recommend(CUSTOMER, k=2), range(24)))
    for recs in many:
        assert [r.product_id for r in recs] == [r.product_id for r in one]
        assert [r.score for r in recs] == pytest.approx([r.score for r in one], abs=1e-5)


def test_encode_graph_refuses_training_mode(service_parts):
    service, _, _ = service_parts
    service.model.train()
    with pytest.raises(RuntimeError, match="eval"):
        service.model.encode_graph(service._graph_gpu)
    service.model.eval()
