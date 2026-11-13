# Windows: run the commands directly in PowerShell if `make` is unavailable.
PY ?= %USERPROFILE%/bm-venv/Scripts/python.exe

install:
	powershell -File scripts/install_rocm_torch.ps1
	$(PY) -m pip install -e ".[dev]"

check-gpu:
	$(PY) scripts/check_gpu.py

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest
