"""Offline smoke tests: no network required, safe for CI."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.features import FEATURES, flood_indicator, rolling_features, series_features  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "flood_model.joblib"


@pytest.fixture(scope="module")
def bundle():
    import joblib

    return joblib.load(MODEL_PATH)


def test_features_present():
    assert len(FEATURES) == 9


def test_rolling_features_values():
    row = rolling_features([1, 2, 3, 4, 5, 6, 7], [1, 1, 1, 1, 1, 2, 4], 6)
    assert row[0] == 7.0
    assert row[1] == pytest.approx(6.0)
    assert row[2] == pytest.approx(4.0)
    assert row[3] == 7.0
    assert row[4] == 4.0
    assert row[5] == pytest.approx(7.0 / 3.0)
    assert row[6] == pytest.approx(11.0 / 7.0)
    assert row[7] == pytest.approx(math.sin(2 * math.pi * 6 / 12))
    assert row[8] == pytest.approx(math.cos(2 * math.pi * 6 / 12))


def test_rolling_features_handles_none():
    row = rolling_features([0, None, 2, None, 4, None, 6], [None] * 7, 1)
    assert row[0] == 6.0
    assert all(v == 0.0 for v in row[4:7])


def test_series_features_skips_warmup():
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    rows, idx = series_features(dates, [0.0] * 10, [0.0] * 10)
    assert len(rows) == len(idx) == 4
    assert idx[0] == 6


def test_flood_indicator_rain_and_discharge():
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    flags = flood_indicator(dates, [0, 5, 40, 0, 0, 0, 0, 0, 0, 0], [0] * 10, 30.0, 10.0)
    assert flags[2] == 1
    assert flags[0] == 0

    flags_q = flood_indicator(dates, [0] * 10, [0, 0, 11, 0, 0, 0, 0, 0, 0, 0], 30.0, 10.0)
    assert flags_q[2] == 1


def test_model_bundle_structure(bundle):
    assert hasattr(bundle["model"], "predict_proba")
    assert bundle["features"] == FEATURES
    assert len(bundle["regions"]) == 16
    assert "roc_auc" in bundle["metrics"]


def test_model_predicts_valid_probability(bundle):
    row = rolling_features([0, 10, 40, 30, 5, 1, 0], [0] * 7, 6)
    prob = bundle["model"].predict_proba([row])[0][1]
    assert 0.0 <= prob <= 1.0
