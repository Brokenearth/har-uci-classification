"""Descarga e idempotencia del UCI HAR Dataset."""

import zipfile
from pathlib import Path

import requests

from har_classifier.config import (
    DATA_RAW,
    UCI_HAR_FOLDER_NAMES,
    UCI_HAR_URL,
    UCI_HAR_ZIP_NAME,
)


def _find_dataset_root() -> Path | None:
    """Busca la carpeta raíz del dataset ya extraído."""
    for name in UCI_HAR_FOLDER_NAMES:
        candidate = DATA_RAW / name
        if _is_valid_dataset(candidate):
            return candidate
    for child in DATA_RAW.iterdir() if DATA_RAW.exists() else []:
        if child.is_dir() and _is_valid_dataset(child):
            return child
    return None


def _is_valid_dataset(path: Path) -> bool:
    """Verifica que existan archivos clave del dataset."""
    if not path.exists():
        return False
    required = [
        path / "train" / "subject_train.txt",
        path / "train" / "y_train.txt",
        path / "train" / "Inertial Signals" / "body_acc_x_train.txt",
        path / "test" / "subject_test.txt",
        path / "test" / "y_test.txt",
    ]
    return all(p.exists() for p in required)


def get_dataset_root() -> Path:
    """Devuelve la ruta raíz del dataset; descarga si no existe."""
    existing = _find_dataset_root()
    if existing is not None:
        return existing
    return download_and_extract()


def download_and_extract() -> Path:
    """Descarga el ZIP y lo extrae en data/raw/."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_RAW / UCI_HAR_ZIP_NAME

    if not zip_path.exists():
        print(f"Descargando UCI HAR Dataset desde {UCI_HAR_URL}...")
        response = requests.get(UCI_HAR_URL, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0 and downloaded % (1024 * 1024) < 8192:
                    pct = 100 * downloaded / total
                    print(f"  Progreso: {pct:.1f}%", end="\r")
        print("\nDescarga completada.")

    print(f"Extrayendo {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW)

    for nested in DATA_RAW.glob("*.zip"):
        if nested == zip_path:
            continue
        print(f"Extrayendo ZIP anidado: {nested.name}...")
        with zipfile.ZipFile(nested, "r") as zf:
            zf.extractall(DATA_RAW)

    root = _find_dataset_root()
    if root is None:
        raise FileNotFoundError(
            "No se encontró el dataset tras la extracción. "
            f"Revise el contenido de {DATA_RAW}"
        )
    print(f"Dataset listo en: {root}")
    return root


if __name__ == "__main__":
    root = get_dataset_root()
    print(f"Dataset root: {root}")
