"""Score review text with the trained star-rating model (models/sentiment_model.joblib)."""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sentiment_model.joblib"


@lru_cache(maxsize=1)
def _model():
    return joblib.load(MODEL_PATH)


def predict_stars(texts: list[str]) -> np.ndarray:
    """Expected star rating (1-5, with decimals) for each text: sum of P(stars=k) * k."""
    if not texts:
        return np.array([])
    model = _model()
    proba = model.predict_proba(texts)
    return (proba * model.classes_).sum(axis=1)
