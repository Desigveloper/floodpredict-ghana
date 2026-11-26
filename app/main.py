"""FloodPredict Ghana API."""

import asyncio
import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .predict import peak_risk, predict_region

BASE_DIR = Path(__file__).resolve().parents[1]
REGIONS = [
    {
        "id": r["id"],
        "name": r["name"],
        "capital": r["capital"],
        "lat": r["lat"],
        "lon": r["lon"],
    }
    for r in json.loads((BASE_DIR / "config" / "regions.json").read_text())
]
REGIONS_BY_ID = {r["id"]: r for r in REGIONS}

app = FastAPI(title="FloodPredict Ghana", version="0.1.0")

_cache: dict = {}
CACHE_TTL = 30 * 60
REGION_TTL = 6 * 60 * 60

_fetch_sem = asyncio.Semaphore(2)
_warming = False


def _cache_get(key, ttl):
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < ttl:
        return entry[1]
    return None


def _region_payload(region, results):
    return {
        "region": region,
        "peak": peak_risk(results),
        "daily": results[-7:],
    }


async def _warm_missing(region_ids):
    global _warming
    if _warming:
        return
    _warming = True
    try:
        for region_id in region_ids:
            if _cache_get(f"risk:{region_id}", REGION_TTL):
                continue
            region = REGIONS_BY_ID.get(region_id)
            if not region:
                continue
            async with _fetch_sem:
                try:
                    results = await asyncio.to_thread(predict_region, region_id, region)
                except Exception:
                    continue
            _cache[f"risk:{region_id}"] = ((time.time()), _region_payload(region, results))
    finally:
        _warming = False


@app.get("/api/regions")
def regions():
    return REGIONS


@app.get("/api/risk")
async def risk():
    ready, pending = [], []
    for region in REGIONS:
        hit = _cache_get(f"risk:{region['id']}", REGION_TTL)
        if hit:
            ready.append(hit)
        else:
            pending.append(region["id"])
    if pending:
        asyncio.create_task(_warm_missing(pending))
    return {"regions": ready, "pending": pending}


@app.get("/api/forecast/{region_id}")
async def forecast(region_id: str, days: int = 7):
    region = REGIONS_BY_ID.get(region_id)
    if not region:
        raise HTTPException(404, f"unknown region: {region_id}")
    days = max(1, min(days, 7))
    hit = _cache_get(f"risk:{region_id}", REGION_TTL)
    if hit and days == 7:
        return {"region": region, "daily": hit["daily"]}
    try:
        results = await asyncio.to_thread(predict_region, region_id, region, days)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(503, str(exc))
    return {"region": region, "daily": results}


@app.get("/health")
def health():
    return {"status": "ok", "cached_regions": len(_cache)}


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
