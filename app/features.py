"""Shared feature engineering used by both training and inference."""

import math

FEATURES = [
    "precip_1d",
    "precip_3d",
    "precip_7d",
    "precip_max_3d",
    "discharge",
    "discharge_3d_mean",
    "discharge_7d_mean",
    "month_sin",
    "month_cos",
]


def rolling_features(precip, discharge, month):
    """Build one feature row from trailing daily series (oldest -> newest).

    precip and discharge must have the same length (>= 7 entries) and end on
    the day the prediction is made for. `month` is the 1-12 month of that day.
    """
    p = [0.0 if v is None else float(v) for v in precip]
    q = [0.0 if v is None else float(v) for v in discharge]
    return [
        p[-1],
        sum(p[-3:]) / 3.0,
        sum(p[-7:]) / 7.0,
        max(p[-3:]),
        q[-1],
        sum(q[-3:]) / 3.0,
        sum(q[-7:]) / 7.0,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
    ]


def series_features(dates, precip, discharge):
    """Feature rows for every day of a daily series, plus the raw trailing values.

    Returns (rows, indices) where rows[i] corresponds to dates[indices[i]].
    """
    rows, indices = [], []
    for i in range(len(dates)):
        if i < 6:
            continue
        y, m, _ = (int(x) for x in dates[i].split("-"))
        rows.append(rolling_features(precip[max(0, i - 6) : i + 1], discharge[max(0, i - 6) : i + 1], m))
        indices.append(i)
    return rows, indices


def flood_indicator(dates, precip, discharge, rain_threshold, discharge_threshold):
    """Daily proxy label: extreme 3-day rainfall or extreme river discharge."""
    n = len(dates)
    flags = [0] * n
    for i in range(2, n):
        p3 = sum(
            0.0 if precip[j] is None else float(precip[j]) for j in range(i - 2, i + 1)
        )
        q = 0.0 if discharge[i] is None else float(discharge[i])
        if p3 >= rain_threshold or q >= discharge_threshold:
            flags[i] = 1
    return flags
