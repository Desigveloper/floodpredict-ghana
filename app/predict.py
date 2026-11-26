"""Inference: combine cached history with live forecasts to score flood risk."""

import numpy as np

from .features import rolling_features
from .weather import fetch_forecast, load_history

_bundle = None

RISK_LEVELS = [(0.25, "low"), (0.5, "moderate"), (0.75, "high"), (2.0, "severe")]


def _load():
    global _bundle
    if _bundle is None:
        import joblib
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "model" / "flood_model.joblib"
        _bundle = joblib.load(path)
    return _bundle


def risk_level(prob):
    for threshold, level in RISK_LEVELS:
        if prob < threshold:
            return level
    return "severe"


def _median_discharge(discharge):
    vals = [float(v) for v in discharge if v is not None]
    return float(np.median(vals)) if vals else 1.0


def predict_region(region_id, region_obj, forecast_days=7):
    bundle = _load()
    model = bundle["model"]
    thresholds = bundle["regions"][region_id]["thresholds"]

    hist = load_history(region_id)
    if hist is None:
        raise FileNotFoundError(
            f"no history cache for {region_id}; run `python -m scripts.train` first"
        )
    h_dates, h_precip, h_discharge = hist

    f_dates, f_precip, f_discharge = fetch_forecast(
        region_obj, past_days=7, forecast_days=forecast_days
    )

    series = {}
    for i, d in enumerate(h_dates):
        if d < f_dates[0]:
            series[d] = [h_precip[i], h_discharge[i]]
    for d, p, q in zip(f_dates, f_precip, f_discharge):
        series.setdefault(d, [p, q])

    dates = sorted(d for d in series if d >= f_dates[0])
    all_dates = sorted(series)
    obs_discharge = [series[d][1] for d in all_dates if series[d][1] is not None]
    fallback_q = _median_discharge(obs_discharge)

    rows, meta = [], []
    for i, d in enumerate(dates):
        j0 = all_dates.index(d) - 6
        y, m, _ = (int(x) for x in d.split("-"))
        precip_win = [series[all_dates[k]][0] for k in range(j0, j0 + 7)]
        disch_win = [series[all_dates[k]][1] for k in range(j0, j0 + 7)]
        disch_win = [fallback_q if v is None else v for v in disch_win]
        rows.append(rolling_features(precip_win, disch_win, m))
        meta.append((d, precip_win[-1], disch_win[-1]))

    probs = model.predict_proba(rows)[:, 1] if rows else []
    results = [
        {
            "date": d,
            "precip": round(p or 0.0, 1),
            "discharge": round(q, 2),
            "probability": round(float(prob), 3),
            "level": risk_level(float(prob)),
        }
        for (d, p, q), prob in zip(meta, probs)
    ]
    return results


def peak_risk(results):
    return max(results, key=lambda r: r["probability"], default=None)
