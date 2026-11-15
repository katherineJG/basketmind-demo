# Windows: run the commands directly in PowerShell if `make` is unavailable.
# PY defaults to the venv created by scripts/install_rocm_torch.ps1; override with `make PY=path/to/python.exe`.
PY ?= $(USERPROFILE)/bm-venv/Scripts/python.exe

install:
	powershell -File scripts/install_rocm_torch.ps1
	$(PY) -m pip install -e ".[dev]"

check-gpu:
	$(PY) scripts/check_gpu.py

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest
