from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "dnn_model.keras"


FEATURE_ORDER = [
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Europe",
    "Japan",
    "USA",
]


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)


def build_features(cleaned_data: dict[str, float | str]) -> np.ndarray:
    origin = str(cleaned_data["origin"])
    origin_values = {
        "Europe": [1.0, 0.0, 0.0],
        "Japan": [0.0, 1.0, 0.0],
        "USA": [0.0, 0.0, 1.0],
    }[origin]

    features = [
        float(cleaned_data["cylinders"]),
        float(cleaned_data["displacement"]),
        float(cleaned_data["horsepower"]),
        float(cleaned_data["weight"]),
        float(cleaned_data["acceleration"]),
        float(cleaned_data["model_year"]),
        *origin_values,
    ]
    return np.array([features], dtype=np.float32)


def predict_mpg(cleaned_data: dict[str, float | str]) -> float:
    model = load_model()
    features = build_features(cleaned_data)
    prediction = model.predict(features, verbose=0)
    return float(np.squeeze(prediction))
