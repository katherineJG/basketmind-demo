import subprocess
import sys
import textwrap

import pytest

from basketmind.inference import NotFound, PrecomputedService
from basketmind.inference.demo_export import write_bundle

K = 2  # the synthetic catalogue has two products


@pytest.fixture
def served(service_parts, tmp_path):
    live, _, _ = service_parts
    live.analytics = {"cancellations": {"purchase_lines": 3}}
    write_bundle(live, tmp_path, k=K)
    return live, PrecomputedService.load(tmp_path), tmp_path


def test_the_precomputed_service_answers_exactly_like_the_live_one(served):
    live, demo, _ = served
    assert live.customers(10) == demo.customers(10) and live.n_customers("") == demo.n_customers("")
    assert live.overview() == demo.overview()
    assert live.analytic("cancellations") == demo.analytic("cancellations")
    assert live.metadata == demo.metadata and demo.as_of == live.as_of
    for customer in live.by_customer:
        assert live.customer(customer) == demo.customer(customer)
        assert live.history(customer) == demo.history(customer)
        for exclude in (False, True):
            for k in (1, K):
                assert live.recommend(customer, k, exclude) == demo.recommend(customer, k, exclude)
        for product in live.items:
            try:
                expected = live.explain(customer, product)
            except NotFound:
                continue
            listed = {r.product_id for e in (False, True) for r in demo.recommend(customer, K, e)}
            if product in listed:  # the demo explains products in the customer's precomputed lists
                assert demo.explain(customer, product) == expected
    for product in live.items:
        assert live.product(product) == demo.product(product)
        assert live.related(product, 5) == demo.related(product, 5)


def test_unknown_things_are_not_found_in_the_same_way(served):
    live, demo, _ = served
    for call in (
        lambda s: s.customer("nobody"),
        lambda s: s.recommend("nobody"),
        lambda s: s.product("NOPE"),
        lambda s: s.explain("1", "NOPE"),
        lambda s: s.analytic("missing"),
    ):
        with pytest.raises(NotFound) as live_error:
            call(live)
        with pytest.raises(NotFound) as demo_error:
            call(demo)
        assert live_error.value.args == demo_error.value.args or "precomputed" in str(demo_error.value)


def test_the_bundle_refuses_to_contain_a_local_path(service_parts, tmp_path):
    live, _, _ = service_parts
    live.metadata = {"checkpoint": "C:\\Users\\someone\\best.pt"}
    with pytest.raises(ValueError, match="local path"):
        write_bundle(live, tmp_path, k=K)
    assert not (tmp_path / "bundle.json").exists()


def test_the_demo_api_runs_in_a_process_where_torch_cannot_be_imported(served):
    _, _, bundle = served
    code = textwrap.dedent(
        f"""
        import sys
        sys.modules["torch"] = None  # any import of torch now raises ImportError
        from fastapi.testclient import TestClient
        from basketmind.inference import PrecomputedService
        from basketmind_api.main import create_app

        service = PrecomputedService.load({str(bundle)!r})
        with TestClient(create_app(service=service)) as client:
            assert client.get("/health").json()["device"] == "precomputed"
            customer = client.get("/customers").json()["items"][0]["customer_id"]
            recs = client.get(f"/recommendations/{{customer}}?k=2").json()
            assert recs["items"] or client.get("/analytics/overview").status_code == 200
            assert client.get("/models/metadata").status_code == 200
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip().endswith("ok")
