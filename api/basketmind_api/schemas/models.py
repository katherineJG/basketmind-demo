"""Response models for every route (Pydantic validates what the service returns)."""

from typing import Any

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: str  # "ok" when the model is loaded, "degraded" otherwise
    model_loaded: bool
    detail: str = ""
    device: str | None = None
    as_of: str | None = None  # the moment predictions refer to (just after the last basket in the data)


class CustomerSummary(BaseModel):
    customer_id: str
    n_baskets: int
    last_basket: str
    country: str


class CustomerPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CustomerSummary]


class CustomerProfile(BaseModel):
    customer_id: str
    n_baskets: int
    first_basket: str
    last_basket: str
    country: str
    total_spend: float
    avg_basket_size: float
    seen_in_training: bool = Field(description="False for customers with no basket in the training period")


class BasketLine(BaseModel):
    product_id: str
    description: str
    quantity: int


class BasketRecord(BaseModel):
    invoice: str
    date: str
    n_items: int
    total_value: float
    items: list[BasketLine]


class RecommendationItem(BaseModel):
    rank: int
    product_id: str
    description: str
    score: float = Field(
        description="Raw model logit. Ranks products for this customer; not a calibrated probability."
    )
    in_history: bool
    times_bought: int
    last_bought: str | None
    predicted_quantity: float


class RecommendationList(BaseModel):
    customer_id: str
    k: int
    exclude_history: bool
    as_of: str
    items: list[RecommendationItem]
    note: str = (
        "Model predictions from a research prototype. Scores rank products; they are not probabilities."
    )


class LinkedProduct(BaseModel):
    product_id: str
    description: str
    association: float = Field(description="PPMI co-purchase strength between the two products")


class Explanation(BaseModel):
    customer_id: str
    product_id: str
    description: str
    model_rank: int
    model_score: float
    bought_before: bool
    times_bought: int
    last_bought: str | None
    linked_to_history: list[LinkedProduct]
    popularity_rank: int
    note: str = (
        "Evidence from the data (history and co-purchase links), not an account of the network's internals."
    )


class Product(BaseModel):
    product_id: str
    description: str
    baskets_containing: int
    last_sold: str
    in_catalogue: bool
    n_related: int


class MonthlyActivity(BaseModel):
    month: str
    baskets: int
    value: float


class TopProduct(BaseModel):
    product_id: str
    description: str
    baskets: int


class CountryCount(BaseModel):
    country: str
    baskets: int


class Overview(BaseModel):
    """Descriptive statistics of the historical data. These are NOT model predictions."""

    customers: int
    baskets: int
    products: int
    first_basket: str
    last_basket: str
    total_value: float
    monthly: list[MonthlyActivity]
    top_products: list[TopProduct]
    top_countries: list[CountryCount]


class CancellationMonth(BaseModel):
    month: str
    purchase_lines: int
    cancellation_lines: int
    cancellation_rate: float


class CancelledProduct(BaseModel):
    product_id: str
    description: str
    cancellation_lines: int
    share_of_lines: float


class LabelHorizon(BaseModel):
    purchase_lines_with_complete_window: int
    base_rate: float
    attribution_precision: float


class CancellationAnalytics(BaseModel):
    """Historical cancellation analysis. No reversal model is served; this shows whether one is feasible."""

    purchase_lines: int
    cancellation_lines: int
    cancellation_share_of_lines: float
    linked: dict[str, float]
    horizons: dict[str, LabelHorizon]
    monthly: list[CancellationMonth]
    top_cancelled: list[CancelledProduct]


class Interval(BaseModel):
    mean: float
    lo: float
    hi: float


class ModelPerformance(BaseModel):
    id: str
    name: str
    group: str = Field(description="'deep' for the deep learning models, 'other' for reference methods")
    selected: bool = Field(
        description="True for the BasketMind variant chosen on validation (the one served)"
    )
    test: dict[str, Interval]
    test_new_items: dict[str, Interval]
    validation: dict[str, float] | None
    seed_std_recall10: float
    coverage: float
    head_share: float


class PairedComparison(BaseModel):
    a: str
    b: str
    metric: str
    new_items: bool
    mean_diff: float
    lo: float
    hi: float
    p: float


class SliceCell(BaseModel):
    slice: str
    model: str
    mean: float
    half: float
    n: int


class PerformanceAnalytics(BaseModel):
    """Test-set results with customer-level 95% intervals. Built offline from finished experiments."""

    models: list[ModelPerformance]
    paired: list[PairedComparison]
    slices: dict[str, list[SliceCell]]
    meta: dict[str, Any]


class ModelMetadata(BaseModel):
    metadata: dict[str, Any]
    disclaimer: str = (
        "Educational research prototype, not a medical, financial or production decision system."
    )
