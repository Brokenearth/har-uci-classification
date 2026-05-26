"""Configuración global del proyecto."""

import os
from pathlib import Path

# Rutas base
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = RESULTS_DIR  # alias (plan: results/)
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

# Dataset UCI HAR
UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)
UCI_HAR_ZIP_NAME = "UCI_HAR_Dataset.zip"
UCI_HAR_FOLDER_NAMES = ("UCI HAR Dataset", "UCI-HAR-Dataset")

# Canales inerciales (orden fijo)
INERTIAL_CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

NUM_CHANNELS = len(INERTIAL_CHANNELS)
NUM_CLASSES = 6
WINDOW_SIZE = 128

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

# Hiperparámetros por defecto
RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 30
HIDDEN_SIZE = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
CONV_FILTERS = (32, 64)
CONV_KERNEL_SIZE = 5
N_CV_FOLDS = 2
VAL_SUBJECT_RATIO = 0.2
EARLY_STOPPING_PATIENCE = 5

# W&B (clave en .env o variable WANDB_API_KEY; ver .env.example)
WANDB_PROJECT = "har-uci-classification"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
