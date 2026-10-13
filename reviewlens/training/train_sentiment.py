"""Train the star-rating model (1-5 stars from review text).

Data: 42K real product reviews from three very different categories, downloaded from
public GitHub mirrors of well-known Kaggle datasets:
  - Amazon Fine Food Reviews (sample)          -> food
  - Amazon Musical Instruments Reviews          -> music & audio gear
  - Women's E-Commerce Clothing Reviews         -> clothing

Evaluation:
  1. Random 80/20 split
  2. "Unseen category" tests: train on two categories, test on the third,
     to check the model generalizes to product types it has never seen.

Usage:  python training/train_sentiment.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "food": "https://raw.githubusercontent.com/TheAI-Lady/customer_reviews/HEAD/Customer_Reviews.csv",
    "music_audio": "https://raw.githubusercontent.com/zaid7860/Amazon-Musical-Instruments-Reviews/HEAD/Musical_instruments_reviews.csv",
    "clothing": "https://raw.githubusercontent.com/AFAgarap/ecommerce-reviews-analysis/master/Womens%20Clothing%20E-Commerce%20Reviews.csv",
}


def load_data() -> pd.DataFrame:
    food = pd.read_csv(SOURCES["food"])
    music = pd.read_csv(SOURCES["music_audio"])
    cloth = pd.read_csv(SOURCES["clothing"])
    frames = [
        pd.DataFrame({"text": food.Summary.fillna("") + ". " + food.Text.fillna(""),
                      "stars": food.Score, "category": "food"}),
        pd.DataFrame({"text": music.summary.fillna("") + ". " + music.reviewText.fillna(""),
                      "stars": music.overall.astype(int), "category": "music_audio"}),
        pd.DataFrame({"text": cloth.Title.fillna("") + ". " + cloth["Review Text"].fillna(""),
                      "stars": cloth.Rating, "category": "clothing"}),
    ]
    data = pd.concat(frames)
    data = data[data.text.str.len() > 15].drop_duplicates("text").reset_index(drop=True)
    return data


def build_model():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000,
                        sublinear_tf=True, strip_accents="unicode"),
        LogisticRegression(max_iter=300, C=2, class_weight="balanced"),
    )


def polarity(stars):
    return np.where(stars <= 2, "negative", np.where(stars >= 4, "positive", "neutral"))


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)
    classes = model.classes_
    pred = classes[proba.argmax(1)]
    expected = (proba * classes).sum(1)
    return {
        "n": int(len(y)),
        "exact_star_accuracy": round(accuracy_score(y, pred), 3),
        "within_one_star": round(float(np.mean(np.abs(pred - y) <= 1)), 3),
        "mean_abs_error_stars": round(mean_absolute_error(y, expected), 3),
        "macro_f1": round(f1_score(y, pred, average="macro"), 3),
        "polarity_accuracy": round(accuracy_score(polarity(y), polarity(pred)), 3),
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
    }


def main():
    data = load_data()
    print(f"Loaded {len(data):,} reviews:", data.groupby("category").size().to_dict())

    train, test = train_test_split(data, test_size=0.2, stratify=data.stars, random_state=42)
    results = {"random_split": evaluate(build_model().fit(train.text, train.stars),
                                        test.text, test.stars.values)}
    print("Random split:", {k: v for k, v in results["random_split"].items() if k != "confusion_matrix"})

    results["unseen_category"] = {}
    for held_out in data.category.unique():
        tr, te = data[data.category != held_out], data[data.category == held_out]
        r = evaluate(build_model().fit(tr.text, tr.stars), te.text, te.stars.values)
        results["unseen_category"][held_out] = r
        print(f"Unseen '{held_out}':", {k: v for k, v in r.items() if k != "confusion_matrix"})

    final = build_model().fit(data.text, data.stars)
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "models" / "sentiment_model.joblib", compress=3)
    results["training_size"] = int(len(data))
    results["categories"] = data.groupby("category").size().to_dict()
    (ROOT / "models" / "sentiment_metrics.json").write_text(json.dumps(results, indent=2))
    print("Saved models/sentiment_model.joblib")


if __name__ == "__main__":
    main()
