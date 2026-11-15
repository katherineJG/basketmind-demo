# BasketMind

BasketMind predicts which products a customer will buy next, using deep learning on the UCI Online Retail II data set.
I built it as a research project and as my final project for the Deep Learning module of my machine learning engineering
course. The model combines a causal Transformer over a customer's purchase history, a GraphSAGE encoder over a product
co-purchase graph, and a customer-context network. The branches are fused into one score per product, and a second
head predicts quantities.

> **Status:** the data pipeline, four baselines, the hybrid model, tuning, ablations, one-time test evaluation and the
> course report are done. The API, dashboard and Docker deployment from the blueprint are **not built yet**.
> This is an educational research prototype, not a production system.

## What I found

I compared BasketMind with four simpler methods on a strict time-based split, tuning every learned model on validation
only and evaluating the test split once per run. **BasketMind did not beat the strongest baselines.**

| Test, repeat purchases allowed | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| Item-item collaborative filtering | **0.145** | **0.188** | **0.326** |
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

The full write-up is the course report: [`reports/BasketMind_Final_Project_Report.pdf`](reports/BasketMind_Final_Project_Report.pdf).

## How it is built

- **No leakage by construction.** Splits are by date, never random. History uses strictly earlier timestamps, graph
  snapshots are built monthly from earlier baskets only, and normalisers are fitted on training data only. A check over
  every example confirms none contains future information.
- **Test split touched once.** Selection, tuning and early stopping use validation only. `scripts/evaluate_test.py`
  refuses to score a run on test twice.
- **Everything is reproducible and tested.** Seeded runs, YAML configs, MLflow tracking, Optuna searches, and a
  pytest suite (110 tests) covering the data pipeline, metrics, model components, training and evaluation tooling.
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
```

## Repository map

| Path | Contents |
|---|---|
| `src/basketmind/` | `data`, `features`, `sequences`, `graph`, `models`, `baselines`, `training`, `evaluation`, `utils` |
| `configs/` | YAML configuration for data, features, graph, model, training, baselines and evaluation |
| `scripts/` | Command-line entry points for each stage |
| `tests/` | Unit, integration and model tests (`pytest`) |
| `docs/` | Architecture, methodology, experiment log, model card, data dictionary and generated reports |
| `reports/` | The course report (PDF) and the script that builds it from the saved results |
| `BASKETMIND master.md` | The engineering blueprint I followed |

## Documentation

[Architecture](docs/architecture.md) · [Research methodology](docs/research_methodology.md) ·
[Experiments](docs/experiments.md) · [Model card](docs/model_card.md) · [Data dictionary](docs/data_dictionary.md) ·
[Data quality report](docs/data_quality_report.md) · [API reference (planned)](docs/api_reference.md)

## Limitations and what is next

One retailer over two years; only 70% of test target items exist in the training catalogue; cold-start customers are
not yet broken out; three seeds and no confidence intervals yet. Next: statistical comparisons and cold-start slices,
a repeat-purchase path for BasketMind, then the API, dashboard and Docker deployment from the blueprint.

## Licence

No licence has been chosen yet, so all rights are reserved by default. The Online Retail II data set is CC BY 4.0.
