# BasketMind

BasketMind predicts which products a customer will buy next, using deep learning on the UCI Online Retail II data set.
I built it as a research project and as my final project for the Deep Learning module of my machine learning engineering
course. The model combines a causal Transformer over a customer's purchase history, a GraphSAGE encoder over a product
co-purchase graph, and a customer-context network. The branches are fused into one score per product, and a second
head predicts quantities.

> **Status:** the data pipeline, four baselines, the hybrid model, tuning, ablations, one-time test evaluation, the
> statistical analysis, the course report, the REST API, the Streamlit dashboard and the Docker deployment are done. This is an educational research prototype, not a production system.

## What I found

I compared BasketMind with four simpler methods on a strict time-based split, tuning every learned model on validation
only and evaluating the test split once per run. **BasketMind did not beat the strongest baselines.**

| Test, repeat purchases allowed | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| BasketMind (graph + customer) + repeat score (added after the report) | **0.228** | n/a | n/a |
| Personal frequency (added after the report, see below) | 0.224 | **0.293** | **0.470** |
| Personal frequency + item-item CF (added after the report) | 0.225 | **0.293** | 0.462 |
| Item-item collaborative filtering | 0.145 | 0.188 | 0.326 |
| Tuned GRU | 0.131 ± 0.001 | 0.175 | 0.325 |
| **BasketMind (graph + customer), selected on validation** | 0.125 ± 0.004 | 0.164 | 0.314 |
| BasketMind (full hybrid, with the Transformer branch) | 0.108 ± 0.003 | 0.143 | 0.285 |
| Matrix factorisation | 0.108 ± 0.001 | 0.144 | 0.282 |
| Popularity | 0.047 | 0.077 | 0.198 |

Mean ± standard deviation over three seeds (popularity and item-item CF are deterministic). Full tables, including a
new-items-only setting, are in [`docs/final_comparison.md`](docs/final_comparison.md). The main lessons:

- The graph branch carries real signal. The Transformer branch did not help on this data, and I did not diagnose why.
- Default training settings gave a model no better than ranking by popularity. Tuning mattered more than architecture.
- Repeat purchases carry most of the accuracy: every personalised model loses 57-61% of its score when only new items count.
- **A baseline I left out beats everything.** Ranking a customer's own past items by how often (and how recently) they bought
  them scores 0.224 Recall@10 on test, 55% above item-item CF and 79% above BasketMind. It is the standard baseline for this
  task and I should have included it from the start. The course report, which I submitted before I ran it, still calls
  item-item CF the strongest baseline; that is no longer true. See [`docs/repeat_purchases.md`](docs/repeat_purchases.md) and the
  [addendum](reports/BasketMind_Report_Addendum.pdf), which lists the statements in the report it corrects.
- **Adding that same repeat score to BasketMind lifts it from 0.125 to 0.228**, level with the baseline (+0.003 over it). The
  improvement comes from the count term, not the network, and it costs new-item discovery: 98% of the top 10 becomes the
  customer's own history. The API still serves the unchanged model.

The full write-up is the course report: [`reports/BasketMind_Final_Project_Report.pdf`](reports/BasketMind_Final_Project_Report.pdf).

## How it is built

- **No leakage by construction.** Splits are by date, never random. History uses strictly earlier timestamps, graph
  snapshots are built monthly from earlier baskets only, and normalisers are fitted on training data only. A check over
  every example confirms none contains future information.
- **Test split touched once.** Selection, tuning and early stopping use validation only. `scripts/evaluate_test.py`
  refuses to score a run on test twice.
- **Everything is reproducible and tested.** Seeded runs, YAML configs, MLflow tracking, Optuna searches, and a
  pytest suite covering the data pipeline, metrics, model components, training and evaluation tooling.
- **AMD GPU first.** Developed on an AMD Radeon RX 7900 GRE with PyTorch on ROCm for Windows, with CPU fallback.

## Setup

Requires Windows 11, Python 3.12 and, for GPU training, an AMD GPU supported by ROCm for Windows.

```powershell
.\scripts\install_rocm_torch.ps1                  # ROCm PyTorch into %USERPROFILE%\bm-venv (use -Venv to change)
& $env:USERPROFILE\bm-venv\Scripts\python.exe -m pip install -e ".[dev]"
& $env:USERPROFILE\bm-venv\Scripts\python.exe scripts\check_gpu.py
& $env:USERPROFILE\bm-venv\Scripts\python.exe -m pytest
```

Install the ROCm wheels first and never let pip replace them with a CPU or CUDA build. `torch` is deliberately not
listed in `pyproject.toml` for that reason. Two ROCm-on-Windows quirks shaped the code: `nn.GRU`'s backward pass fails
(the GRU baseline uses an explicit `GRUCell` loop), and a process can hang at exit if GPU tensors are still alive
(`release_gpu()` in `utils/device.py` prevents it).

## Data

Download the UCI Online Retail II data set (CC BY 4.0) from
https://archive.ics.uci.edu/dataset/502/online+retail+ii and place the workbook at `data/raw/online_retail_II.xlsx`.
Raw and processed data are gitignored and never modified. Please keep the data set's attribution if you reuse results.

## Reproducing the results

Run these in order with the venv's Python (shown as `python`):

```powershell
python scripts\run_m1.py                                   # clean, aggregate, split, build examples + quality report
python scripts\run_baselines.py                            # popularity, item-item CF, matrix factorisation, GRU
python scripts\tune.py --model gru --trials 30             # Optuna on validation (also: matrix_factorisation)
python scripts\tune_neural.py --trials 24                  # BasketMind settings, validation only
python scripts\train.py --name abl_graph_customer --branches graph,customer --tuned --no-test --seed 42
python scripts\evaluate_test.py --runs abl_graph_customer:42,abl_graph_customer:43,abl_graph_customer:44
python scripts\report_final.py --variants abl_full,abl_graph_customer     # docs/final_comparison.md
python reports\build_report.py                             # rebuilds the PDF (needs Google Chrome)
python scripts\collect_per_example.py                      # per-example test results (refits the baselines, ~15 min)
python scripts\analyse_results.py                          # docs/evaluation_rigour.md: intervals, paired tests, slices
python scripts\analyse_cancellations.py                    # docs/cancellation_labels.md
```

## Running the API

The API needs `data\processed` and a trained BasketMind run, so do the first steps of "Reproducing the results" first (through
`evaluate_test.py`). Without a checkpoint it still starts, but `/health` reports `degraded` and the other routes answer 503.

```powershell
python scripts\export_model.py                # copies the best-validation BasketMind checkpoint to artifacts\
python -m uvicorn basketmind_api.main:app     # then open http://127.0.0.1:8000/docs
```

It serves next-basket recommendations, customer and product views and evidence-based explanations from the real model
(about 70 ms per recommendation after start-up). See the [API reference](docs/api_reference.md).

## Running the dashboard

```powershell
python scripts\export_analytics.py            # once: writes the cancellation and model-performance JSON the API serves
python -m uvicorn basketmind_api.main:app     # terminal 1
python -m streamlit run dashboard\app.py      # terminal 2, then open http://localhost:8501
```

Six pages: executive overview, customer intelligence, next basket predictions, product intelligence, cancellation
intelligence and model performance. The dashboard only calls the API (set `BASKETMIND_API_URL`, or use the sidebar box,
if it is not on `127.0.0.1:8000`). Pages that describe history and pages that show model predictions carry different
banners, every chart has a table view, and the performance page shows the test results with their confidence intervals,
including where BasketMind loses to the baselines (the personal-frequency baseline included, once
`scripts\run_repeat_baselines.py` has been run and `export_analytics.py` re-run).

## Running both with Docker

```powershell
python scripts\run_m1.py                      # once: data\processed from the raw dataset (see Setup)
python scripts\export_model.py                # once: artifacts\checkpoints and model metadata
python scripts\export_analytics.py            # once: artifacts\analytics
docker compose -f deployment\docker-compose.yml up --build
```

The API is on http://localhost:8000 (`/health`, `/docs`) and the dashboard on http://localhost:8501. The dashboard waits for the
API's health check, which takes about 30 seconds while the model and data load. The images contain code only: `data/processed` and
`artifacts` are mounted read-only from your machine, because they are derived from the raw dataset and the trained checkpoint, which
are not in Git. Containers have no ROCm GPU, so the API runs on CPU torch there; the output is the same model but scores can differ
from the GPU run in the last decimal places. I built and ran this on Docker Desktop for Windows.

## Repository map

| Path | Contents |
|---|---|
| `src/basketmind/` | `data`, `features`, `sequences`, `graph`, `models`, `baselines`, `training`, `evaluation`, `utils` |
| `configs/` | YAML configuration for data, features, graph, model, training, baselines and evaluation |
| `api/` | The FastAPI service (routes, schemas, dependencies) |
| `dashboard/` | The Streamlit dashboard: `app.py`, six pages, API client and theme |
| `deployment/` | Dockerfiles for the API and the dashboard, and the compose file that runs both |
| `scripts/` | Command-line entry points for each stage |
| `tests/` | Unit, integration and model tests (`pytest`) |
| `docs/` | Architecture, methodology, experiment log, model card, data dictionary and generated reports |
| `reports/` | The course report (PDF), an addendum that corrects it (`BasketMind_Report_Addendum.pdf`), and the scripts that build them from the saved results |
| `BASKETMIND master.md` | The engineering blueprint I followed |

## Documentation

[Architecture](docs/architecture.md) · [Research methodology](docs/research_methodology.md) ·
[Experiments](docs/experiments.md) · [Model card](docs/model_card.md) · [Data dictionary](docs/data_dictionary.md) ·
[Data quality report](docs/data_quality_report.md) · [Evaluation rigour](docs/evaluation_rigour.md) ·
[Cancellation labels](docs/cancellation_labels.md) · [API reference](docs/api_reference.md)

## Limitations and what is next

One retailer over two years; only 70% of test target items exist in the training catalogue; three seeds per model. The
[evaluation rigour report](docs/evaluation_rigour.md) adds customer-level confidence intervals, paired comparisons and slices. Its
main finding: item-item CF's overall lead comes from customers with short histories, and the neural models only overtake it with
11 or more earlier baskets. [Cancellation labels](docs/cancellation_labels.md) look usable as a weak 30-day label, but no reversal
head is built yet. [Repeat purchases](docs/repeat_purchases.md) are 55% of target items, and almost all of every model's accuracy
comes from them. Giving BasketMind a repeat score closes the gap but does not beat the baseline by more than 0.003. What is left
is new-item discovery, where item-item CF still leads.

## Licence

No licence has been chosen yet, so all rights are reserved by default. The Online Retail II data set is CC BY 4.0.
