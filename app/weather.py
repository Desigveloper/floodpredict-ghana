"""Open-Meteo clients (archive + forecast) and simple JSON caching."""

import json
import time
from pathlib import Path

import httpx

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FLOODS_URL = "https://flood-api.open-meteo.com/v1/floods"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HISTORY_DIR = DATA_DIR / "history"


def _get(url, params=None, retries=6):
    for attempt in range(retries):
        try:
            r = httpx.get(url, params=params, timeout=90)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"{r.status_code}", request=r.request, response=r
                )
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, httpx.TransportError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt * 3)


def fetch_archive(region, start_date, end_date):
    """Historical daily precipitation and river discharge for one region."""
    payload = _get(
        ARCHIVE_URL,
        {
            "latitude": region["lat"],
            "longitude": region["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum,river_discharge",
            "timezone": "Africa/Accra",
        },
    )
    daily = payload["daily"]
    return daily["time"], daily["precipitation_sum"], daily["river_discharge"]


def fetch_forecast(region, past_days=7, forecast_days=7):
    """Forecast API: 7 days of future precipitation (plus trailing observed days)."""
    payload = _get(
        FORECAST_URL,
        {
            "latitude": region["lat"],
            "longitude": region["lon"],
            "daily": "precipitation_sum,river_discharge",
            "past_days": past_days,
            "forecast_days": forecast_days,
            "timezone": "Africa/Accra",
        },
    )
    daily = payload["daily"]
    return daily["time"], daily["precipitation_sum"], daily["river_discharge"]


def save_history(region_id, dates, precip, discharge):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / f"{region_id}.json"
    payload = {
        "updated": time.strftime("%Y-%m-%d"),
        "dates": dates,
        "precip": precip,
        "discharge": discharge,
    }
    path.write_text(json.dumps(payload))


def load_history(region_id):
    path = HISTORY_DIR / f"{region_id}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload["dates"], payload["precip"], payload["discharge"]
