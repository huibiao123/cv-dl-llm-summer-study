from pathlib import Path


# =========================
# Project
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================
# Data
# =========================

DATA_DIR = PROJECT_ROOT / "data"

NUM_CLASSES = 10

IMAGE_SIZE = 28


# =========================
# DataLoader参数
# =========================

BATCH_SIZE = 64

NUM_WORKERS = 0


# =========================
# 训练集分割
# =========================

TRAIN_SIZE = 5000

VAL_SIZE = 1000


# =========================
# 训练参数
# =========================

SEED = 0

EPOCHS = 5

LEARNING_RATE = 0.001


# =========================
# Output路径
# =========================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "best_model.pth"

HISTORY_PATH = OUTPUT_DIR / "history.csv"

ACCURACY_CURVE_PATH = OUTPUT_DIR / "accuracy_curve.png"

CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

LOSS_CURVE_PATH = OUTPUT_DIR / "loss_curve.png"

ERROR_EXAMPLES_PATH = OUTPUT_DIR / "error_examples.png"