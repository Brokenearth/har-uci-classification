"""Interfaz web Flask — inferencia con señales, etiquetas y probabilidades."""

import sys
from pathlib import Path

from flask import Flask, render_template_string, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har_classifier.config import ACTIVITY_NAMES, DATA_PROCESSED, MODELS_DIR
from har_classifier.evaluation.metrics import load_model_from_checkpoint
from har_classifier.evaluation.visualization import plot_signals_base64
from har_classifier.preprocessing import load_processed
from har_classifier.training.trainer import Trainer
from har_classifier.utils.device import get_device

app = Flask(__name__)

TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>HAR Classifier — UCI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap" rel="stylesheet"/>
  <style>
    :root {
      --bg: #f0f4f8;
      --surface: #ffffff;
      --surface2: #f8fafc;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --accent: #2563eb;
      --accent-dim: #1d4ed8;
      --ok: #059669;
      --ok-bg: #ecfdf5;
      --ok-border: #6ee7b7;
      --bad: #dc2626;
      --bad-bg: #fef2f2;
      --bad-border: #fca5a5;
      --radius: 12px;
      --shadow: 0 4px 24px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "DM Sans", system-ui, sans-serif;
      background: var(--bg);
      background-image: linear-gradient(180deg, #ffffff 0%, var(--bg) 120px);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
    }
    .wrap { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem 3rem; }
    header { margin-bottom: 1.75rem; }
    header h1 {
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
      margin-bottom: 0.35rem;
    }
    header p { color: var(--muted); font-size: 0.95rem; }
    .badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
    .pill {
      font-size: 0.75rem;
      font-weight: 500;
      padding: 0.3rem 0.75rem;
      border-radius: 999px;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--muted);
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.35rem 1.5rem;
      box-shadow: var(--shadow);
    }
    .card h2 {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 1rem;
    }
    .controls { margin-bottom: 1.25rem; }
    .controls form {
      display: flex;
      flex-wrap: wrap;
      align-items: flex-end;
      gap: 1rem;
    }
    .field { flex: 1; min-width: 200px; }
    .field label {
      display: block;
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--muted);
      margin-bottom: 0.4rem;
    }
    input[type="range"] {
      width: 100%;
      height: 8px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    input[type="number"] {
      width: 100%;
      max-width: 120px;
      padding: 0.55rem 0.75rem;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-family: inherit;
      font-size: 1rem;
    }
    input[type="number"]:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
    }
    button {
      padding: 0.65rem 1.5rem;
      border: none;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-family: inherit;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s, box-shadow 0.15s;
    }
    button:hover {
      background: var(--accent-dim);
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25);
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(300px, 1fr) minmax(480px, 1.65fr);
      gap: 1.25rem;
      align-items: start;
    }
    @media (max-width: 960px) {
      .grid { grid-template-columns: 1fr; }
    }
    .result-banner {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      padding: 1rem 1.15rem;
      border-radius: 10px;
      margin-bottom: 1.25rem;
      border: 1px solid;
    }
    .result-banner.ok { background: var(--ok-bg); border-color: var(--ok-border); }
    .result-banner.bad { background: var(--bad-bg); border-color: var(--bad-border); }
    .status-dot {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }
    .result-banner.ok .status-dot { background: var(--ok); }
    .result-banner.bad .status-dot { background: var(--bad); }
    .result-text .label {
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    .result-text .value { font-size: 1rem; font-weight: 600; margin-top: 0.15rem; }
    .result-banner.ok .value { color: var(--ok); }
    .result-banner.bad .value { color: var(--bad); }
    .labels-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      margin-bottom: 1rem;
    }
    .label-box {
      padding: 0.9rem;
      border-radius: 10px;
      background: var(--surface2);
      border: 1px solid var(--border);
    }
    .label-box .tag {
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 0.35rem;
    }
    .label-box .act { font-size: 0.92rem; font-weight: 600; }
    .label-box.pred.ok { border-color: var(--ok-border); background: var(--ok-bg); }
    .label-box.pred.ok .act { color: var(--ok); }
    .label-box.pred.bad { border-color: var(--bad-border); background: var(--bad-bg); }
    .label-box.pred.bad .act { color: var(--bad); }
    .prob-list { list-style: none; }
    .prob-item { margin-bottom: 0.7rem; }
    .prob-head {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      margin-bottom: 0.35rem;
    }
    .prob-name.top { font-weight: 600; color: var(--text); }
    .prob-pct { color: var(--muted); font-variant-numeric: tabular-nums; }
    .prob-item.top .prob-pct { color: var(--ok); font-weight: 600; }
    .bar-track {
      height: 10px;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 5px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 4px;
      background: linear-gradient(90deg, #93c5fd, var(--accent));
      transition: width 0.35s ease;
    }
    .prob-item.top .bar-fill {
      background: linear-gradient(90deg, #34d399, var(--ok));
    }
    .chart-card {
      padding: 1.25rem;
    }
    .chart-card .chart-wrap {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.75rem;
      min-height: 420px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .chart-card img {
      width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
    }
    footer {
      margin-top: 2rem;
      text-align: center;
      font-size: 0.78rem;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Clasificación de actividades humanas</h1>
      <p>UCI HAR Dataset · Conv1D+LSTM · 9 señales inerciales · ventana 128 pasos</p>
      <div class="badge-row">
        <span class="pill">Muestra {{ idx }} / {{ max_idx }}</span>
        <span class="pill">Conjunto de prueba</span>
      </div>
    </header>

    <div class="card controls">
      <h2>Explorar muestra</h2>
      <form method="get" id="sample-form">
        <div class="field" style="flex: 2; min-width: 280px;">
          <label for="idx-range">Índice (arrastra o escribe)</label>
          <input type="range" id="idx-range" name="idx" min="0" max="{{ max_idx }}"
                 value="{{ idx }}" oninput="document.getElementById('idx-num').value=this.value"/>
        </div>
        <div class="field">
          <label for="idx-num">Índice exacto</label>
          <input type="number" id="idx-num" min="0" max="{{ max_idx }}" value="{{ idx }}"
                 oninput="document.getElementById('idx-range').value=this.value"/>
        </div>
        <button type="submit">Analizar</button>
      </form>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Predicción</h2>
        <div class="result-banner {{ 'ok' if match else 'bad' }}">
          <span class="status-dot"></span>
          <div class="result-text">
            <div class="label">{{ 'Clasificación correcta' if match else 'Error de clasificación' }}</div>
            <div class="value">{{ 'El modelo acertó la actividad' if match else 'El modelo no coincide con la real' }}</div>
          </div>
        </div>
        <div class="labels-row">
          <div class="label-box">
            <div class="tag">Etiqueta real</div>
            <div class="act">{{ true_label }}</div>
          </div>
          <div class="label-box pred {{ 'ok' if match else 'bad' }}">
            <div class="tag">Predicha</div>
            <div class="act">{{ pred_label }}</div>
          </div>
        </div>
        <h2 style="margin-top: 0.25rem;">Probabilidades</h2>
        <ul class="prob-list">
          {% for name, p, is_top in probs %}
          <li class="prob-item {{ 'top' if is_top else '' }}">
            <div class="prob-head">
              <span class="prob-name {{ 'top' if is_top else '' }}">{{ name }}</span>
              <span class="prob-pct">{{ "%.1f"|format(p*100) }}%</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill" style="width: {{ (p*100)|round(1) }}%;"></div>
            </div>
          </li>
          {% endfor %}
        </ul>
      </div>

      <div class="card chart-card">
        <h2>Señales inerciales (9 canales)</h2>
        <div class="chart-wrap">
          <img src="data:image/png;base64,{{ chart_b64 }}" alt="Señales temporales"/>
        </div>
      </div>
    </div>

    <footer>HAR Classifier · http://127.0.0.1:5000 · Detener servidor con Ctrl+C</footer>
  </div>
  <script>
    document.getElementById('sample-form').addEventListener('submit', function() {
      document.getElementById('idx-range').value = document.getElementById('idx-num').value;
    });
  </script>
</body>
</html>
"""

_model = None
_data = None
_device = None

_CKPT = MODELS_DIR / "final_model.pt"
_NPZ = DATA_PROCESSED / "har_processed.npz"


def _preflight() -> None:
    """Comprueba artefactos antes de abrir el servidor."""
    missing = []
    if not _NPZ.exists():
        missing.append(
            f"  - {_NPZ}\n"
            "    Ejecute: .venv\\Scripts\\python.exe scripts\\02_preprocess.py"
        )
    if not _CKPT.exists():
        missing.append(
            f"  - {_CKPT}\n"
            "    Ejecute: .venv\\Scripts\\python.exe scripts\\04_final_training.py"
        )
    if missing:
        print("Faltan archivos para la interfaz:\n" + "\n".join(missing), file=sys.stderr)
        raise SystemExit(1)


def _load():
    global _model, _data, _device
    if _model is None:
        _device = get_device(verbose=False)
        _model = load_model_from_checkpoint(_CKPT, _device)
        _data = load_processed()


@app.route("/")
def index():
    import numpy as np
    import torch

    _load()
    max_idx = len(_data["y_test"]) - 1
    idx = min(max(int(request.args.get("idx", 0)), 0), max_idx)

    x = _data["X_test"][idx]
    y_true = int(_data["y_test"][idx])
    x_t = torch.from_numpy(x.transpose(1, 0).astype(np.float32)).unsqueeze(0)
    _model.eval()
    with torch.no_grad():
        logits = Trainer._forward(_model, x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    y_pred = int(probs.argmax())

    true_label = ACTIVITY_NAMES[y_true]
    pred_label = ACTIVITY_NAMES[y_pred]

    chart_b64 = plot_signals_base64(x, title=f"Muestra {idx}", theme="light")

    prob_items = [
        (ACTIVITY_NAMES[i], float(probs[i]), i == y_pred)
        for i in range(len(ACTIVITY_NAMES))
    ]
    prob_items.sort(key=lambda t: -t[1])

    return render_template_string(
        TEMPLATE,
        idx=idx,
        max_idx=max_idx,
        true_label=true_label,
        pred_label=pred_label,
        match=(y_true == y_pred),
        probs=prob_items,
        chart_b64=chart_b64,
    )


if __name__ == "__main__":
    _preflight()
    print("Cargando modelo y datos de test (una sola vez)...")
    _load()
    print(f"Listo: {len(_data['y_test'])} muestras de test.")
    print("Abre http://127.0.0.1:5000")
    print("Detener con Ctrl+C")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
