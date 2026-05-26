# Pipeline completo UCI HAR (ejecutar desde la raíz del proyecto)
$ErrorActionPreference = "Stop"
$env:WANDB_MODE = "offline"
$py = Join-Path (Join-Path (Split-Path $PSScriptRoot -Parent) ".venv") "Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py (Join-Path $PSScriptRoot "01_download_data.py")
& $py (Join-Path $PSScriptRoot "02_preprocess.py")
& $py (Join-Path $PSScriptRoot "03_cross_validation.py") --epochs 30 --no-wandb
& $py (Join-Path $PSScriptRoot "04_final_training.py") --epochs 40 --no-wandb
& $py (Join-Path $PSScriptRoot "05_evaluate.py")
& $py (Join-Path $PSScriptRoot "06_visualize_errors.py")
Write-Host "Pipeline completado."
