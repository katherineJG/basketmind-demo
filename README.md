# BasketMind

Hybrid deep learning system for next-basket prediction on the UCI Online Retail II dataset:
a temporal Transformer, a GraphSAGE product encoder and a customer-context encoder, fused and
trained with multi-task heads. The engineering blueprint is in `BASKETMIND master.md`.

> Status: **M0 (foundation) and M1 (data engineering) done.** No models yet, so no model results exist.
> Data facts and rules: `docs/data_quality_report.md` and `docs/data_dictionary.md`.

## Requirements

- Windows 11, AMD Radeon RX 7900 GRE (ROCm PyTorch; CPU fallback exists)
- Python 3.12, AMD ROCm 7.2.1 PyTorch wheels (torch 2.9.1)

## Setup

```powershell
.\scripts\install_rocm_torch.ps1                       # creates %USERPROFILE%\bm-venv, installs ROCm torch
%USERPROFILE%\bm-venv\Scripts\python.exe -m pip install -e ".[dev]"
%USERPROFILE%\bm-venv\Scripts\python.exe scripts\check_gpu.py
%USERPROFILE%\bm-venv\Scripts\python.exe -m pytest
%USERPROFILE%\bm-venv\Scripts\python.exe -m ruff check .
```

Install the ROCm wheels first and never let pip replace them with a CPU or CUDA build.
`torch` is deliberately not listed in `pyproject.toml` for that reason.

## Dataset

UCI Online Retail II (CC BY 4.0): https://archive.ics.uci.edu/dataset/502/online+retail+ii
Place the file at `data/raw/online_retail_II.xlsx`. Raw data is never modified or committed.

```powershell
%USERPROFILE%\bm-venv\Scripts\python.exe scripts\run_m1.py   # --refresh re-reads the xlsx (~45 s)
```

This writes `data/processed/*` and `docs/data_quality_report.md`, and fails if any next-basket example
contains future information.

## Layout

See section 10 of the blueprint. Configs live in `configs/`, code in `src/basketmind/`.
