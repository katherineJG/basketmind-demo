# API reference

A FastAPI service that answers questions about customers and products and returns next-basket recommendations from the
exported BasketMind model. It is an educational research prototype, not a production service. Interactive documentation
(OpenAPI) is served at `/docs` once it is running.

## Running it

```powershell
python scripts\export_model.py                # copies the best-validation checkpoint to artifacts\checkpoints\best.pt
python -m uvicorn basketmind_api.main:app     # loads the model and data once at startup (about 30 seconds)
```

Startup needs `data/processed/` (from `scripts/run_m1.py`) and the exported checkpoint. If either is missing the API still starts:
`/health` reports `"status": "degraded"` with the reason, and every other route answers `503`. Predictions refer to one moment,
`as_of`, which is just after the last basket in the data (9 December 2011).

## Routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Service and model status, device and `as_of` |
| GET | `/customers?limit&offset&prefix` | Page of customers, most baskets first (`limit` 1-200) |
| GET | `/customers/{customer_id}` | Profile: baskets, spend, country, whether seen in training |
| GET | `/customers/{customer_id}/history?limit` | Past baskets, newest first, with product names |
| GET | `/recommendations/{customer_id}?k&exclude_history` | Ranked recommendations (`k` 1-100) |
| GET | `/recommendations/{customer_id}/explain?product_id&top` | Evidence for one product (see below) |
| GET | `/products/{product_id}` | Description, how many baskets contain it, last sold |
| GET | `/products/{product_id}/related?k` | Strongest co-purchase neighbours in the product graph |
| GET | `/analytics/overview` | Descriptive statistics of the historical data |
| GET | `/analytics/cancellations` | Historical cancellation rates, most-cancelled products and the 30/90-day label feasibility figures |
| GET | `/models/performance` | Test results with customer-level 95% intervals, paired comparisons and history-length slices |
| GET | `/models/metadata` | The served variant, seed, parameters, validation and test metrics |

The two analytics routes serve files written offline by `python scripts\export_analytics.py` (it reads the saved per-example test
results and the processed data; nothing is recomputed per request). If the files are missing the routes answer `404` with that
command in the message. `/analytics/cancellations` is descriptive history; no reversal model is served.

`/models/performance` also lists the personal-frequency baseline (and its paired comparisons with BasketMind, item-item CF and the
GRU) when `experiments/analysis/per_example_repeat.parquet` exists, which `scripts/run_repeat_baselines.py` writes. It is not a
served model; see [repeat purchases](repeat_purchases.md).

## Things worth knowing

- **Scores are not probabilities.** Each recommendation carries a raw model logit that ranks products for that customer. The
  weighted training loss makes them uncalibrated, so the API never presents them as chances.
- **`exclude_history=true`** removes products the customer has already bought, which tests discovery. Products with no remaining
  candidate are omitted, so the list can be shorter than `k`.
- **`predicted_quantity`** is the quantity head's estimate of units if the customer buys the product. It has not been evaluated
  (see the model card), so treat it as indicative.
- **Explanations are evidence from the data, not from inside the network:** whether and how often the customer bought the product,
  the PPMI links between it and products in the customer's history, its rank in the model's scores for this customer, and its overall
  popularity rank. The response says so in its `note`.
- **Descriptive vs predictive.** `/analytics/overview`, `/customers`, `/products` and `/history` describe the past.
  Only `/recommendations` (and its `explain` rank and score) comes from the model.
- **Cold-start customers** (no basket in the training period) are served with the "unseen customer" embedding; the profile shows
  `seen_in_training: false`. In the evaluation these customers were scored less accurately than item-item CF would score them.
- **Errors:** `404` for an unknown customer or product (or no basket before `as_of`), `422` for invalid parameters, `503` if the model
  is not loaded.

## Design

The routes are thin: every answer comes from `basketmind.inference.BasketMindService`, which builds model inputs with the same
code the training pipeline uses (`features/build.py`, `training/data.build_batch`, `graph/build.py`). A test checks that serving a
historical example gives exactly the logits the training pipeline computes for it.
