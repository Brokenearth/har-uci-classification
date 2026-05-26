# HAR con smartphone — UCI HAR

¿Qué hace una persona con el móvil en la cintura: camina, sube escaleras o solo está sentada?  
Este proyecto responde eso con **redes recurrentes en PyTorch**, usando el [dataset UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones): **6 actividades**, **9 señales inerciales** por ventana de 2,56 s.

Comparamos **LSTM** frente a **Conv1D+LSTM**, elegimos la mejor arquitectura con validación cruzada **por sujeto** (sin mezclar personas entre train y val) y dejamos una **app web** para explorar predicciones en el conjunto de test.

| | |
|---|---|
| **Validación (métrica principal)** | ~**97,8 %** exactitud |
| **Test (referencia / demo)** | ~**92,6 %** |
| **Ganador en CV** | Conv1D+LSTM (92,45 % vs 90,61 % LSTM) |

Repositorio: [Brokenearth/har-uci-classification](https://github.com/Brokenearth/har-uci-classification)

---

## En 30 segundos

```powershell
py -3 -m venv .venv
.venv\Scripts\pip.exe install -r requirements.txt

.venv\Scripts\python.exe scripts/01_download_data.py
.venv\Scripts\python.exe scripts/02_preprocess.py
.venv\Scripts\python.exe scripts/03_cross_validation.py --epochs 30 --no-wandb
.venv\Scripts\python.exe scripts/04_final_training.py --epochs 40 --no-wandb
.venv\Scripts\python.exe scripts/05_evaluate.py
.venv\Scripts\python.exe scripts/06_visualize_errors.py
```

Luego abre la interfaz:

```powershell
.venv\Scripts\python.exe app/flask_app.py
```

→ [http://127.0.0.1:5000](http://127.0.0.1:5000)

> Si PowerShell no deja usar `activate`, invoca siempre `.venv\Scripts\python.exe`.

---

## Qué hay dentro del repo

```
ProyectoF13CI/
├── data/                 # Datos crudos y har_processed.npz
├── src/har_classifier/   # Lógica: modelos, entrenamiento, métricas
├── scripts/              # Pipeline 01 → 06 (punto de entrada)
├── app/                  # Flask y Streamlit
├── site/ + netlify.toml  # Demo estática (sin servidor Python)
├── results/              # Salidas al correr el pipeline
└── evidencias/           # Copia lista para entregar / demostrar
```

Los scripts `01`–`06` **orquestan**; el paquete `har_classifier` **implementa**. No es código duplicado.  
La carpeta `outputs/` es legado — ignórala y usa `results/`.

---

## Modelos

**LSTM** — memoria temporal directa sobre la ventana `(128 × 9)`.

**Conv1D+LSTM** — convoluciones locales (32 → 64) + LSTM 2×128; es el modelo final del proyecto.

Ambos entrenan con `StratifiedGroupKFold` agrupando por **sujeto**, para que nadie aparezca a la vez en train y val de un mismo fold.

---

## Weights & Biases (opcional)

Si quieres ver pérdida y accuracy por época en la nube:

1. Copia `.env.example` → `.env` y pon tu `WANDB_API_KEY` (`.env` no va a Git).
2. Ejecuta:

```powershell
.\scripts\run_training_wandb.ps1
```

Proyecto en W&B: **har-uci-classification**.  
¿Prefieres ir sin W&B? Deja `--no-wandb` en los pasos 03 y 04, como en el pipeline de arriba.

---

## Interfaz web

**Flask (recomendado en Windows)** — muestra las 9 señales, etiqueta real, predicción y probabilidades sobre **test**:

```powershell
.venv\Scripts\python.exe app/flask_app.py
# o: scripts\run_flask.bat
```

**Streamlit** — misma idea, otro front:

```powershell
.venv\Scripts\python.exe scripts/run_streamlit.py
```

Necesitas `data/processed/har_processed.npz` y `results/models/final_model.pt`. Si solo tienes el modelo en `evidencias/`, mira `evidencias/LEEME.md`.

---

## Demo en Netlify

Netlify no corre PyTorch. La carpeta `site/` sirve una **demo estática**: mismas probabilidades y gráficas, pero predicciones **ya calculadas** sobre test.

Regenerar tras cambiar modelo o datos:

```powershell
.venv\Scripts\python.exe scripts/build_netlify_site.py
```

Despliegue rápido: conecta el repo en [Netlify](https://app.netlify.com/) (lee `netlify.toml`, `publish = site`) o arrastra `site/` en [Netlify Drop](https://app.netlify.com/drop).

---

## Resultados y entrega

Tras el pipeline, lo importante queda en `results/`:

- `metrics/metrics_final.json` — métricas en **val** y referencia en test  
- `figures/confusion_matrix.png` — matriz en **validación**  
- `figures/aciertos_*.png` / `errores_*.png` — ejemplos en **test**  
- `models/final_model.pt` — pesos del Conv1D+LSTM (entrenado solo con train interno)

La carpeta **`evidencias/`** es el paquete para revisión sin reentrenar (`sync_evidencias.py` la actualiza).

| Conjunto | Muestras | Uso |
|----------|----------|-----|
| Train interno | 5551 | Entrenar modelo final |
| Validación | 1801 | Métricas oficiales y matriz |
| Test UCI | 2947 | App web, ejemplos cualitativos, Netlify |

---

## Requisitos

- Python **3.10+**
- GPU NVIDIA opcional (PyTorch con CUDA acelera el entrenamiento)

---

## Créditos de datos

Anguita et al., *Human Activity Recognition Using Smartphones*, UCI ML Repository (ID 240), 2013.
