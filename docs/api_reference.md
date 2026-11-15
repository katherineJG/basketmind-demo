# API reference

**Not implemented yet.** The REST API is milestone M5 in the blueprint and I have not started it. This page records the planned
surface so the dashboard and tests can be written against it. `api/basketmind_api/` currently holds empty packages only.

Planned FastAPI routes (all responses validated with Pydantic):

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service and model status |
| GET | `/customers` | List customers |
| GET | `/customers/{customer_id}` | Customer profile |
| GET | `/customers/{customer_id}/history` | Past baskets |
| GET | `/recommendations/{customer_id}` | Ranked next-basket recommendations |
| GET | `/recommendations/{customer_id}/explain` | Evidence for the recommendations (planned from the product graph) |
| GET | `/products/{product_id}` | Product details |
| GET | `/products/{product_id}/related` | Strongest co-purchase neighbours |
| GET | `/analytics/overview` | Descriptive statistics |
| GET | `/models/metadata` | Loaded checkpoint, variant and metrics |

The service must load a real trained checkpoint (`training/checkpoint.py`) and reuse the training-time feature code, so that
training and inference preprocessing cannot drift apart.
