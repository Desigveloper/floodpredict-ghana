# FloodPredict Ghana

An AI-powered system that predicts and forecasts flood risk for all 16 regions of Ghana, combining a machine-learning risk model with a live map dashboard.

## How it works

1. **Data** — Historical daily precipitation (ERA5) and river discharge (GloFAS) for each regional capital come from the free [Open-Meteo](https://open-meteo.com/) archive API.
2. **Labeling** — Because a verified national flood-event dataset is not yet integrated, days are labeled as flood proxies when the 3-day cumulative rainfall exceeds the region's 95th percentile or river discharge exceeds its 90th percentile. A training label means *flooding very likely within the following 2 days*.
3. **Model** — A `RandomForestClassifier` on rolling rainfall/discharge features plus seasonal encoding, trained on 2000–2025 data with a temporal train/test split and class weighting for rare events.
4. **Serving** — A FastAPI backend loads the trained model, merges the cached history with live 7-day weather + GloFAS discharge forecasts, and scores the flood probability per region per day.
5. **Dashboard** — A single-page Leaflet + Chart.js app shows a color-coded risk map, a ranked region list, and a 7-day probability chart per region.

## Project layout

```
app/
  main.py        FastAPI app and API routes
  predict.py     inference: merges history + live forecasts, scores risk
  features.py    shared rolling feature engineering (train + inference)
  weather.py     Open-Meteo archive/forecast/floods clients, local caching
config/
  regions.json   the 16 regions with capital coordinates
scripts/
  train.py       data ingestion + labeling + training
static/          dashboard (HTML/CSS/JS)
model/           trained model bundle (committed)
data/history/    cached archive data per region (generated, not committed)
```

## Getting started

```bash
pip install -r requirements.txt
python -m scripts.train      # fetch data and train (first time: ~1–2 min, needs internet)
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 — the API is documented at `/docs`.

## API

| Endpoint | Description |
| --- | --- |
| `GET /api/regions` | All 16 regions with coordinates |
| `GET /api/risk` | 7-day forecast summary for every region (cached 30 min) |
| `GET /api/forecast/{region_id}?days=7` | Day-by-day probability, rainfall and discharge |
| `GET /health` | Liveness check |

## Limitations & roadmap

- Flood labels are statistical extremes (proxy), not verified NADMO/EM-DAT event records — swap in real event data as labels when available.
- Features are limited to rainfall, discharge and seasonality; adding soil moisture, elevation/drainage density and urban exposure would improve skill.
- Regional capitals are proxies for whole regions; gridded multi-point ingestion is a natural next step.
- Risk thresholds (low/moderate/high/severe) are heuristic cuts of model probability.
