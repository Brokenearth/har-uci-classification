# Clasificación de Actividades Humanas (UCI HAR)

Sistema reproducible en Python para clasificar seis actividades humanas a partir de nueve señales inerciales del [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). Compara **LSTM** vs **Conv1D+LSTM** con validación cruzada por sujeto, registro en Weights & Biases e interfaz web para inferencia.

## Estructura

```
ProyectoF13CI/
├── data/                     # raw/ + processed/ (.npz, stats.json)
├── src/har_classifier/       # Paquete con toda la lógica
│   ├── loader.py             # Lectura UCI (archivos .txt)
│   ├── torch_dataset.py      # HARDataset (PyTorch)
│   ├── models/, training/, evaluation/, utils/
├── scripts/                  # Pipeline 01–06 (orquestación)
├── app/                      # flask_app.py, streamlit_app.py
├── site/                     # Demo estática para Netlify
├── netlify.toml              # publish = site
├── results/                  # Salidas del pipeline
└── evidencias/               # Copia para entrega
```

Los `scripts/0X_*.py` llaman módulos en `src/har_classifier/` (p. ej. `training/final_train.py`). No es código duplicado: scripts = entrada; paquete = librería. La carpeta `outputs/` es legado; usar solo `results/`.

## Requisitos

- Python 3.10+
- (Opcional) GPU NVIDIA con PyTorch CUDA

```powershell
py -3 -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt
```

Si PowerShell bloquea `activate`, usa `.venv\Scripts\python.exe` directamente.

## Pipeline completo

```powershell
.venv\Scripts\python.exe scripts/01_download_data.py
.venv\Scripts\python.exe scripts/02_preprocess.py
.venv\Scripts\python.exe scripts/03_cross_validation.py --epochs 30 --no-wandb
.venv\Scripts\python.exe scripts/04_final_training.py --epochs 40 --no-wandb
.venv\Scripts\python.exe scripts/05_evaluate.py
.venv\Scripts\python.exe scripts/06_visualize_errors.py
.venv\Scripts\python.exe scripts/test_models.py
```

### Weights & Biases (opcional)

1. Copie `.env.example` → `.env` y pegue su API key en `WANDB_API_KEY` (no suba `.env` a Git).
2. Ejecute entrenamiento con registro en W&B:

```powershell
.\scripts\run_training_wandb.ps1
```

Runs en el proyecto [har-uci-classification](https://wandb.ai) (carpeta local `wandb/` si usa `WANDB_MODE=offline`). Para desactivar W&B sin cambiar scripts: añada `--no-wandb` a los pasos 03 y 04.

## Interfaz web

**Flask (recomendado en Windows)** — señales 9 canales, etiquetas y probabilidades:

```powershell
.venv\Scripts\python.exe app/flask_app.py
```

O `scripts\run_flask.bat` → http://127.0.0.1:5000

**Streamlit** — http://localhost:8501

```powershell
.venv\Scripts\python.exe scripts/run_streamlit.py
```

## Demo en Netlify (estático)

Netlify no ejecuta Flask ni PyTorch. La carpeta `site/` es una demo con **predicciones precomputadas** del conjunto de test (misma UI: probabilidades + 9 canales + figuras de evidencias).

Regenerar tras cambiar el modelo o los datos:

```powershell
.venv\Scripts\python.exe scripts/build_netlify_site.py
```

**Desplegar** (repositorio `Brokenearth/har-uci-classification`):

1. Sube a GitHub el repo con `site/data/samples.json.gz`, `netlify.toml` y el resto de `site/`.
2. En [Netlify](https://app.netlify.com/) → **Add new site** → **Import an existing project** → elige el repo de GitHub.
3. Netlify detecta `netlify.toml` (`publish = "site"`). Pulsa **Deploy site**.
4. La URL será algo como `https://<nombre>.netlify.app`.

Alternativa sin Git: arrastra la carpeta `site/` en **Netlify Drop** (https://app.netlify.com/drop).

## Carpeta de evidencias (`evidencias/`)

Copia lista para entregar o revisar sin reentrenar: `metrics_final.json`, `comparacion_modelos.csv`, `confusion_matrix.png`, `loss_curves.png`, `final_model.pt`. Ver `evidencias/LEEME.md` para ejecutar la app con ese modelo.

Actualizar tras un nuevo pipeline:

```powershell
.venv\Scripts\python.exe scripts\sync_evidencias.py
```

## Evidencias del plan (`results/`)

| Artefacto | Descripción |
|-----------|-------------|
| `data/processed/har_processed.npz` | Train / val / test normalizados |
| `results/metrics/comparacion_modelos.csv` | CV 2-fold LSTM vs Conv1D+LSTM |
| `results/models/best_model.pt` | Mejor fold CV (`ensure_best_model.py` o tras 03) |
| `results/models/final_model.pt` | Modelo final (entrenado solo con train) |
| `results/metrics/metrics_final.json` | Métricas **val** (literal c) + test (literal d) |
| `results/metrics/training_history.json` | Historial entrenamiento final |
| `results/figures/loss_curves.png` | Curvas pérdida / accuracy |
| `results/figures/confusion_matrix.png` | Matriz de confusión (**validación**, literal c) |
| `results/figures/aciertos_*.png` | 5 aciertos en **test** (literal d) |
| `results/figures/errores_*.png` | 5 errores en **test** (literal d) |
| `results/reports/error_analysis.md` | Análisis de confusiones |

**Métrica principal (literal c):** exactitud en **validación** (~97.8 %). **Test** (~92.6 %) se usa para ejemplos cualitativos (literal d) e interfaz web (literal f).

## Modelos

- **LSTMClassifier**: `(B, T, C)`, LSTM 2×128, dropout 0.3.
- **Conv1DLSTMClassifier**: Conv1d→BN→ReLU→MaxPool (32→64), LSTM, clasificación.

Validación: `StratifiedGroupKFold` por sujeto sin fuga entre train y val.
