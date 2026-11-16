# Architecture

## Data flow

```
raw xlsx -> clean (label every line) -> baskets -> time split -> customer sequences -> next-basket examples
                                           |                                             |
                                           +--> monthly PPMI graph snapshots             +--> features (basket, customer)
                                                                  \                      /
                                                                   +-> batches -> BasketMind -> scores
```

Every stage reads and writes files under `data/processed/`, so each can be rerun on its own.

## Modules (`src/basketmind/`)

| Package | Key pieces | Purpose |
|---|---|---|
| `data` | `ingest.py`, `clean.py`, `baskets.py`, `pipeline.py` | Load both sheets with canonical column names; label every line (`purchase`, `cancellation`, `adjustment`, `non_product`, `invalid_price`, `duplicate_overlap`); aggregate invoices into baskets; split by date; write the quality report |
| `sequences` | `build_customer_sequences`, `build_next_basket_examples`, `verify_temporal_integrity` | One example per basket with history: up to 50 strictly earlier baskets as input, the basket as target. The verifier fails on any future or cross-customer history |
| `features` | `build_feature_store` | 8 basket features and 6 customer features; normalisers fitted on training data only |
| `graph` | `calculate_ppmi`, `build_product_graph`, `create_graph_snapshots` | Symmetric PPMI co-purchase graph from baskets strictly before a cutoff; monthly snapshots |
| `models` | `Batch`, `TemporalEncoder`, `ProductGraphEncoder`, `CustomerContextEncoder`, `FusionNetwork`, `BasketMindModel` | The hybrid network (below) |
| `baselines` | `task.py`, `models.py`, `run.py`, `tune.py` | Popularity, item-item CF, matrix factorisation, GRU; shared catalogue and evaluation; Optuna search |
| `training` | `data.py`, `losses.py`, `engine.py`, `tune.py`, `checkpoint.py`, `final_eval.py` | Batching by graph snapshot, multi-task loss, seeded training loop, tuning, checkpoints, one-time test evaluation |
| `evaluation` | `ranking.py`, `final_report.py` | Recall@K, Precision@K, NDCG@K, MRR over the whole catalogue; the combined comparison report |
| `utils` | `config`, `device`, `seed`, `logging` | YAML loading, ROCm/CPU device selection and `release_gpu()`, seeding |

## The model

`BasketMindModel(cfg, n_items, n_customers, n_countries)`; `forward(batch, graph, quantity_items)` returns a dict.

1. **Shared item embedding** (128-d). It feeds both the temporal and the graph branch.
2. **Temporal branch.** Each history basket becomes the mean of its item embeddings concatenated with 8 basket features, then
   a linear projection, a learned position embedding and a 2-layer, 4-head, causal Transformer encoder with padding masked.
   The output is the state at the last real basket.
3. **Graph branch.** A 2-layer weighted GraphSAGE (`WeightedSAGEConv`, a PyG `MessagePassing` subclass: PyG's stock
   `SAGEConv` ignores edge weights) updates every item vector from its PPMI-weighted neighbours. The branch's output for a
   customer is the mean, over their history baskets, of each basket's mean updated item vector.
4. **Customer branch.** Customer id embedding (id 0 = unseen), country embedding and 6 normalised summary features through
   a 128 then 64 MLP.
5. **Fusion.** Concatenate the enabled branches, then Linear, GELU, LayerNorm, Dropout, Linear to 128-d.
6. **Heads.** Basket logits are the fused vector dotted with the item vectors (graph output when the graph branch is on)
   plus a per-item bias. A pair head predicts log1p quantity for chosen items. A reversal head exists in the code but is
   disabled, and its loss is refused until cancellation labels are validated.

`branches: {temporal, graph, customer}` in `configs/model.yaml` switch components on or off for ablations.
`untie_output: true` scores items with a separate output table instead (off by default: it scored lower on validation).

## Serving

`inference.BasketMindService` turns a customer's history into ranked recommendations and answers the profile, history, product,
related-product, explanation and overview questions. It rebuilds the feature store with the training code, builds one product graph
snapshot as of the end of the data, and reuses `training.data.build_batch` and `features.build.customer_feature_rows`, so there is
a single implementation of preprocessing. `api/basketmind_api` wraps it in FastAPI routes with Pydantic response models
([API reference](api_reference.md)). `scripts/export_model.py` copies the best-validation checkpoint to `artifacts/` with a
metadata file; the API reads both. The graph branch's output (one vector per product) depends only on the weights and the
graph snapshot, so the service computes it once at start-up and passes it to `BasketMindModel.forward(graph_vectors=...)`
instead of re-running the graph encoder over every edge for each request. Outputs are identical to the uncached path (checked
on the real model: same top-10 lists, score difference 0), a recommendation drops from about 2.6 s to 4 ms on CPU, and
concurrent requests no longer multiply memory use. A graph passed in explicitly (another point in time) is still encoded on demand. `scripts/export_analytics.py` writes the cancellation and performance summaries the API serves.

The service is split in two so that a small host can serve the API. `inference/data_service.py` (`DataService`) answers everything
that needs no model (customers, history, products, related products, overview, evidence for a recommendation) and imports no
torch. `BasketMindService` adds the model. `PrecomputedService` extends `DataService` and reads the model's answers from a bundle
(`artifacts/demo/`, written by `scripts/export_demo.py`): the top 100 products per customer under both policies with scores,
quantities and catalogue-wide ranks. It can only explain products in those lists. It is tested against the live service
route by route, and in a process where importing torch raises an error.

The Streamlit dashboard (`dashboard/`) sits on top of the API and imports nothing from `basketmind`: all data arrives over HTTP
through `dashboard/utils/api_client.py`, which turns every failure into a message rather than a stack trace. Pages are separated into
historical descriptions (customers, products, cancellations, overview) and model output (predictions, performance), each with its
own banner. Colours come from the project design system; charts use one hue plus grey and always offer a table view.

`deployment/` runs the two as separate containers. The API image installs CPU torch (there is no ROCm in a container) and the
project package, and sets `BASKETMIND_ROOT=/app`; the dashboard image installs only Streamlit, Plotly, pandas and httpx. Data and
artifacts are mounted read-only rather than baked in, so the images stay small and contain nothing derived from the licensed data.
The compose file starts the dashboard only after the API's `/health` check passes. The API image installs the whole `pyproject.toml`
dependency list (including MLflow and Optuna, which serving does not need); splitting serving dependencies into an extra would
slim it, but changes the packaging, so I have not done it.

## Checkpoints

`save_checkpoint` stores weights, the constructor arguments, optional optimiser state and plain-value metadata; it refuses to
overwrite a file unless asked. `load_checkpoint` rebuilds the model from the stored arguments with weights-only loading.
