# Installs AMD's official ROCm 7.2.1 PyTorch wheels for Windows into a Python 3.12 venv.
# Default venv: %USERPROFILE%\bm-venv (a short path avoids the Windows 260-character limit); override with -Venv.
# Run once, then: <venv>\Scripts\python.exe -m pip install -e ".[dev]"
param([string]$Venv = (Join-Path $env:USERPROFILE "bm-venv"))
$py = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $py)) { py -3.12 -m venv $Venv }
$b = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1"
& $py -m pip install --upgrade pip
& $py -m pip install --no-cache-dir "$b/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl" "$b/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl" "$b/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl" "$b/rocm-7.2.1.tar.gz"
& $py -m pip install --no-cache-dir "$b/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$b/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl" "$b/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
