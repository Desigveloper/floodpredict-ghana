"""Train the flood-risk model on Open-Meteo archive data for all Ghana regions.

Usage: python -m scripts.train
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURES, flood_indicator, series_features  # noqa: E402
from app.weather import fetch_archive, load_history, save_history  # noqa: E402

REGIONS_PATH = Path(__file__).resolve().parents[1] / "config" / "regions.json"
MODEL_DIR = Path(__file__).resolve().parents[1] / "model"

START_DATE = "2000-01-01"
ARCHIVE_LAG_DAYS = 5
RAIN_PCT = 0.95
DISCHARGE_PCT = 0.90

END_DATE = (
    __import__("datetime").date.today()
    - __import__("datetime").timedelta(days=ARCHIVE_LAG_DAYS)
).isoformat()


def _fresh_history(region_id):
    cached = load_history(region_id)
    if not cached or not cached[0]:
        return None
    dates, precip, discharge = cached
    if dates[-1] >= END_DATE:
        return cached
    return None


def build_region_dataset(region):
    cached = _fresh_history(region["id"])
    if cached:
        dates, precip, discharge = cached
    else:
        dates, precip, discharge = fetch_archive(region, START_DATE, END_DATE)
        save_history(region["id"], dates, precip, discharge)

    p3 = np.array(
        [
            sum(
                0.0 if precip[j] is None else float(precip[j])
                for j in range(max(0, i - 2), i + 1)
            )
            for i in range(len(dates))
        ]
    )
    rain_threshold = float(np.quantile(p3, RAIN_PCT))

    q_values = [float(v) for v in discharge if v is not None]
    if q_values:
        discharge_threshold = float(np.quantile(q_values, DISCHARGE_PCT))
    else:
        discharge_threshold = None

    if discharge_threshold is None:
        flags = flood_indicator(
            dates, precip, [None] * len(dates), rain_threshold, 1.0
        )
    else:
        flags = flood_indicator(
            dates, precip, discharge, rain_threshold, discharge_threshold
        )

    rows, indices = series_features(dates, precip, discharge)
    X, y = [], []
    for row, i in zip(rows, indices):
        if i + 2 >= len(dates):
            continue
        label = 1 if (flags[i + 1] or flags[i + 2]) else 0
        X.append(row)
        y.append(label)
    thresholds = {"rain_3day": rain_threshold, "discharge": discharge_threshold}
    return np.array(X), np.array(y), thresholds, len(dates)


def main():
    regions = json.loads(REGIONS_PATH.read_text())
    all_X, all_y = [], []
    meta = {}

    for region in regions:
        X, y, thresholds, n_days = build_region_dataset(region)
        all_X.append(X)
        all_y.append(y)
        meta[region["id"]] = {
            "name": region["name"],
            "thresholds": thresholds,
            "days": n_days,
            "positives": int(y.sum()),
        }
        print(f"{region['name']:15s} days={n_days} floods={int(y.sum())} thresholds={thresholds}")
        time.sleep(3)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    split = int(len(X) * 0.8)  # temporal split: train older, test newer
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "average_precision": round(float(average_precision_score(y_test, proba)), 3),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 3),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
    }
    print("metrics:", metrics)

    MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "regions": meta,
        "metrics": metrics,
        "trained": __import__("time").strftime("%Y-%m-%d"),
        "data_source": "Open-Meteo ERA5 archive + GloFAS river discharge (2000-2025)",
    }
    joblib.dump(bundle, MODEL_DIR / "flood_model.joblib", compress=3)
    print(f"saved -> {MODEL_DIR / 'flood_model.joblib'}")


if __name__ == "__main__":
    main()
