# Carpeta de evidencias — UCI HAR

Artefactos principales del proyecto para revisión o demostración **sin volver a entrenar**.

| Archivo | Descripción |
|---------|-------------|
| `metrics_final.json` | Métricas en **val** (literal c) y **test** (referencia / literal d) |
| `comparacion_modelos.csv` | Validación cruzada 2-fold: LSTM vs Conv1D+LSTM |
| `confusion_matrix.png` | Matriz de confusión en conjunto de **validación** (literal c) |
| `loss_curves.png` | Curvas de entrenamiento final (solo train; val para early stopping) |
| `final_model.pt` | Pesos del modelo Conv1D+LSTM entrenado solo con train |

## Ejecutar la app Flask sin entrenar

1. Instalar dependencias (desde la raíz del proyecto):

   ```powershell
   py -3 -m venv .venv
   .venv\Scripts\pip.exe install -r requirements.txt
   ```

2. Copiar el modelo a la ruta que usa la aplicación:

   ```powershell
   New-Item -ItemType Directory -Force -Path results\models
   Copy-Item evidencias\final_model.pt results\models\final_model.pt
   ```

   También debe existir `data/processed/har_processed.npz` (generado con `scripts/02_preprocess.py`).

3. Arrancar la interfaz:

   ```powershell
   .venv\Scripts\python.exe app\flask_app.py
   ```

   Abrir http://127.0.0.1:5000

## Regenerar esta carpeta

Tras un nuevo entrenamiento o evaluación:

```powershell
.venv\Scripts\python.exe scripts\sync_evidencias.py
```
