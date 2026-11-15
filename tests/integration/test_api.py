import pytest
from fastapi.testclient import TestClient

from basketmind_api.main import create_app


@pytest.fixture
def client(service_parts):
    service, _, _ = service_parts
    with TestClient(create_app(service=service)) as c:
        c.service = service
        yield c


def test_every_blueprint_route_exists_and_is_documented(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {
        "/health",
        "/customers",
        "/customers/{customer_id}",
        "/customers/{customer_id}/history",
        "/recommendations/{customer_id}",
        "/recommendations/{customer_id}/explain",
        "/products/{product_id}",
        "/products/{product_id}/related",
        "/analytics/overview",
        "/models/metadata",
    }


def test_health_reports_the_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["as_of"] == "2011-08-01"


def test_recommendations_come_from_the_model_not_from_constants(client):
    r = client.get("/recommendations/1", params={"k": 2})
    assert r.status_code == 200
    body = r.json()
    direct = client.service.recommend("1", k=2)
    assert [i["product_id"] for i in body["items"]] == [d.product_id for d in direct]
    assert [i["score"] for i in body["items"]] == pytest.approx([d.score for d in direct])
    assert body["k"] == 2 and body["exclude_history"] is False and "not probabilities" in body["note"]
    assert [i["rank"] for i in body["items"]] == [1, 2]


def test_exclude_history_and_query_validation(client):
    only_new = client.get("/recommendations/2", params={"k": 5, "exclude_history": True}).json()
    assert [i["product_id"] for i in only_new["items"]] == ["85123A"]
    assert client.get("/recommendations/2", params={"k": 0}).status_code == 422
    assert client.get("/recommendations/2", params={"k": 101}).status_code == 422
    assert client.get("/customers", params={"limit": 0}).status_code == 422
    assert client.get("/customers", params={"offset": -1}).status_code == 422


def test_unknown_ids_give_404_with_a_message(client):
    for url in (
        "/customers/nobody",
        "/customers/nobody/history",
        "/recommendations/nobody",
        "/products/NOPE",
    ):
        r = client.get(url)
        assert r.status_code == 404 and "not found" in r.json()["detail"], url
    assert client.get("/products/NOPE/related").status_code == 404
    assert client.get("/recommendations/1/explain", params={"product_id": "NOPE"}).status_code == 404


def test_customer_routes(client):
    page = client.get("/customers", params={"limit": 1, "offset": 1}).json()
    assert page["total"] == 2 and [c["customer_id"] for c in page["items"]] == ["2"]
    assert client.get("/customers", params={"prefix": "1"}).json()["total"] == 1
    profile = client.get("/customers/1").json()
    assert profile["n_baskets"] == 5 and profile["seen_in_training"] is True
    hist = client.get("/customers/1/history", params={"limit": 2}).json()
    assert [h["invoice"] for h in hist] == ["104", "103"] and hist[0]["items"][0]["product_id"] == "22000"


def test_explain_requires_a_product_and_returns_evidence(client):
    assert client.get("/recommendations/2/explain").status_code == 422  # product_id is required
    ev = client.get("/recommendations/2/explain", params={"product_id": "85123A"}).json()
    assert ev["bought_before"] is False and ev["linked_to_history"][0]["product_id"] == "22000"
    assert "not an account of the network" in ev["note"]


def test_product_overview_and_metadata_routes(client):
    assert client.get("/products/22000").json()["baskets_containing"] == 5
    assert client.get("/products/22000/related").json()[0]["product_id"] == "85123A"
    ov = client.get("/analytics/overview").json()
    assert ov["baskets"] == 6 and ov["customers"] == 2 and ov["top_products"][0]["product_id"] == "22000"
    assert client.get("/models/metadata").json()["metadata"] == {"variant": "test"}


def test_api_starts_degraded_when_the_model_cannot_be_loaded():
    def broken():
        raise FileNotFoundError("artifacts/checkpoints/best.pt")

    with TestClient(create_app(factory=broken)) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        assert "best.pt" in health["detail"]
        r = c.get("/customers")
        assert r.status_code == 503 and "best.pt" in r.json()["detail"]
