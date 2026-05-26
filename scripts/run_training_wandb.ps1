# Entrenamiento con Weights & Biases (CV + modelo final).
# Requiere .env con WANDB_API_KEY o variable de entorno ya definida.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line -match "^([^=]+)=(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}
if (-not $env:WANDB_API_KEY) {
    Write-Error "Defina WANDB_API_KEY en .env (copie desde .env.example) o en el entorno."
}
if (-not $env:WANDB_MODE) {
    $env:WANDB_MODE = "online"
}

$py = Join-Path (Join-Path $root ".venv") "Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

Set-Location $root
Write-Host "W&B: proyecto har-uci-classification | modo $env:WANDB_MODE"

& $py (Join-Path $PSScriptRoot "03_cross_validation.py") --epochs 30
& $py (Join-Path $PSScriptRoot "04_final_training.py") --epochs 40
& $py (Join-Path $PSScriptRoot "05_evaluate.py")
& $py (Join-Path $PSScriptRoot "06_visualize_errors.py")
$sync = Join-Path $PSScriptRoot "sync_evidencias.py"
if (Test-Path $sync) {
    & $py $sync
}
Write-Host "Entrenamiento con W&B completado. Revise https://wandb.ai o la carpeta wandb/"
