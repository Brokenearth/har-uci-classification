"""Interfaz Streamlit para inferencia interactiva sobre UCI HAR."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from har_classifier.config import ACTIVITY_NAMES, INERTIAL_CHANNELS, MODELS_DIR
from har_classifier.preprocessing import load_processed


@st.cache_resource(show_spinner="Cargando modelo y datos…")
def load_model_and_data():
    import torch

    from har_classifier.evaluation.metrics import load_model_from_checkpoint
    from har_classifier.utils.device import get_device

    device = get_device(verbose=False)
    ckpt = MODELS_DIR / "final_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No se encontró {ckpt}. Ejecute scripts\\04_final_training.py"
        )
    model = load_model_from_checkpoint(ckpt, device)
    data = load_processed()
    return model, data, device


def predict(model, x_window: np.ndarray, device) -> np.ndarray:
    import torch

    from har_classifier.training.trainer import Trainer

    x_tensor = torch.from_numpy(x_window.transpose(1, 0).astype(np.float32)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = Trainer._forward(model, x_tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def main() -> None:
    st.set_page_config(page_title="HAR Classifier — UCI", layout="wide")
    st.title("Clasificación de Actividades Humanas (UCI HAR)")

    try:
        model, data, device = load_model_and_data()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    X_test = data["X_test"]
    y_test = data["y_test"]

    st.sidebar.header("Muestra de prueba")
    idx = st.sidebar.slider("Índice", 0, len(X_test) - 1, 0)

    x_window = X_test[idx]
    y_true = int(y_test[idx])
    probs = predict(model, x_window, device)
    y_pred = int(probs.argmax())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Etiquetas")
        st.metric("Real", ACTIVITY_NAMES[y_true])
        st.metric("Predicha", ACTIVITY_NAMES[y_pred])
        st.bar_chart(
            pd.Series(
                {ACTIVITY_NAMES[i]: float(probs[i]) for i in range(len(ACTIVITY_NAMES))}
            )
        )

    with col2:
        st.subheader("Señales inerciales (9 canales)")
        chart_df = pd.DataFrame(x_window, columns=INERTIAL_CHANNELS)
        st.line_chart(chart_df, height=400)
        from har_classifier.evaluation.visualization import plot_signals_base64

        import base64

        b64 = plot_signals_base64(
            x_window,
            title=f"#{idx} {ACTIVITY_NAMES[y_true]} → {ACTIVITY_NAMES[y_pred]}",
        )
        st.image(base64.b64decode(b64), caption="Vista 3×3 (9 canales)")


if __name__ == "__main__":
    main()
