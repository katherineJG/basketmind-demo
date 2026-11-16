"""All HTTP routes. Every answer comes from the inference service; nothing here is hard-coded."""

from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from basketmind.inference import DataService, NotFound
from basketmind_api import schemas as s
from basketmind_api.services.dependencies import get_service

router = APIRouter()
Service = Annotated[DataService, Depends(get_service)]
Identifier = Annotated[str, Query(min_length=1, max_length=20)]


def _or_404(call, *args, **kwargs):
    try:
        return call(*args, **kwargs)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e


@router.get("/health", response_model=s.Health)
def health(request: Request) -> s.Health:
    service = getattr(request.app.state, "service", None)
    if service is None:
        reason = getattr(request.app.state, "load_error", "the model has not been loaded")
        return s.Health(status="degraded", model_loaded=False, detail=reason)
    return s.Health(
        status="ok",
        model_loaded=True,
        device=str(service.device),
        as_of=str(service.as_of.date()),
        detail="model loaded",
    )


@router.get("/customers", response_model=s.CustomerPage)
def list_customers(
    service: Service,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    prefix: Annotated[str, Query(max_length=20)] = "",
) -> s.CustomerPage:
    return s.CustomerPage(
        total=service.n_customers(prefix),
        limit=limit,
        offset=offset,
        items=service.customers(limit, offset, prefix),
    )


@router.get("/customers/{customer_id}", response_model=s.CustomerProfile)
def customer(customer_id: str, service: Service) -> s.CustomerProfile:
    return s.CustomerProfile(**_or_404(service.customer, customer_id))


@router.get("/customers/{customer_id}/history", response_model=list[s.BasketRecord])
def history(
    customer_id: str, service: Service, limit: Annotated[int, Query(ge=1, le=200)] = 20
) -> list[s.BasketRecord]:
    return [s.BasketRecord(**row) for row in _or_404(service.history, customer_id, limit)]


@router.get("/recommendations/{customer_id}", response_model=s.RecommendationList)
def recommendations(
    customer_id: str,
    service: Service,
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    exclude_history: bool = False,
) -> s.RecommendationList:
    recs = _or_404(service.recommend, customer_id, k, exclude_history)
    return s.RecommendationList(
        customer_id=customer_id,
        k=k,
        exclude_history=exclude_history,
        as_of=str(service.as_of.date()),
        items=[s.RecommendationItem(**asdict(r)) for r in recs],
    )


@router.get("/recommendations/{customer_id}/explain", response_model=s.Explanation)
def explain(
    customer_id: str,
    service: Service,
    product_id: Identifier,
    top: Annotated[int, Query(ge=1, le=20)] = 5,
) -> s.Explanation:
    return s.Explanation(**_or_404(service.explain, customer_id, product_id, top))


@router.get("/products/{product_id}", response_model=s.Product)
def product(product_id: str, service: Service) -> s.Product:
    return s.Product(**_or_404(service.product, product_id))


@router.get("/products/{product_id}/related", response_model=list[s.LinkedProduct])
def related(
    product_id: str, service: Service, k: Annotated[int, Query(ge=1, le=50)] = 10
) -> list[s.LinkedProduct]:
    return [s.LinkedProduct(**row) for row in _or_404(service.related, product_id, k)]


@router.get("/analytics/overview", response_model=s.Overview)
def overview(service: Service) -> s.Overview:
    return s.Overview(**service.overview())


@router.get("/analytics/cancellations", response_model=s.CancellationAnalytics)
def cancellations(service: Service) -> s.CancellationAnalytics:
    return s.CancellationAnalytics(**_or_404(service.analytic, "cancellations"))


@router.get("/models/performance", response_model=s.PerformanceAnalytics)
def model_performance(service: Service) -> s.PerformanceAnalytics:
    return s.PerformanceAnalytics(**_or_404(service.analytic, "performance"))


@router.get("/models/metadata", response_model=s.ModelMetadata)
def model_metadata(service: Service) -> s.ModelMetadata:
    return s.ModelMetadata(metadata=service.metadata)
