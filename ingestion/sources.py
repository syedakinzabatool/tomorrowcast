"""Phase 2: live data sources for the hourly ingestion run.

Open-Meteo (weather + air quality) is the primary source: free, no key,
hourly resolution. OpenWeather's free "current weather" endpoint is the
secondary source, used only to fill gaps when Open-Meteo is missing a
value for an hour — see ingestion/fuse.py for the priority rule.

All timestamps are kept in UTC throughout the pipeline (never a local
Asia/Karachi offset) so comparisons and storage stay unambiguous; local
time is only something the dashboard/report layer converts to for display.
"""
from __future__ import annotations

import datetime as dt
import time

import requests

from common.config import CITY, optional_env

OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENWEATHER_CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"

# Hourly runs happen unattended in GitHub Actions, where a single transient
# DNS/connect timeout shouldn't fail the whole run. Short exponential backoff
# covers that without masking a genuinely dead endpoint (still fails after 3 tries).
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2


def _get_with_retries(url: str, params: dict) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_error = exc
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise last_error

WEATHER_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "uv_index",
]
AIR_QUALITY_HOURLY_VARS = ["us_aqi"]


def _latest_complete_hour_index(timestamps: list[str]) -> int:
    """Open-Meteo's hourly array always includes the current, still
    in-progress hour. We want the last hour that has actually finished, so
    we pick the latest timestamp that isn't in the future relative to the
    real UTC clock, rather than trusting any fixed offset into the array.
    """
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    parsed = [dt.datetime.fromisoformat(t) for t in timestamps]
    candidates = [i for i, t in enumerate(parsed) if t <= now]
    if not candidates:
        raise RuntimeError("Open-Meteo returned no timestamp at or before now")
    return candidates[-1]


def fetch_open_meteo_weather() -> dict:
    params = {
        "latitude": CITY.latitude,
        "longitude": CITY.longitude,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "wind_speed_unit": "kmh",
        "past_hours": 3,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    resp = _get_with_retries(OPEN_METEO_WEATHER_URL, params)
    hourly = resp.json()["hourly"]
    idx = _latest_complete_hour_index(hourly["time"])
    return {
        "timestamp": hourly["time"][idx],
        "temperature": hourly["temperature_2m"][idx],
        "humidity": hourly["relative_humidity_2m"][idx],
        "precipitation": hourly["precipitation"][idx],
        "wind_speed": hourly["wind_speed_10m"][idx],
        "wind_direction": hourly["wind_direction_10m"][idx],
        "uv_index": hourly["uv_index"][idx],
    }


def fetch_open_meteo_air_quality() -> dict:
    params = {
        "latitude": CITY.latitude,
        "longitude": CITY.longitude,
        "hourly": ",".join(AIR_QUALITY_HOURLY_VARS),
        "past_hours": 3,
        "forecast_days": 1,
        "timezone": "UTC",
    }
    resp = _get_with_retries(OPEN_METEO_AIR_QUALITY_URL, params)
    hourly = resp.json()["hourly"]
    idx = _latest_complete_hour_index(hourly["time"])
    return {"timestamp": hourly["time"][idx], "aqi": hourly["us_aqi"][idx]}


def fetch_openweather_current() -> dict | None:
    """Secondary source. Returns None (never raises) if no API key is
    configured yet, or if the call fails for any reason — a missing
    secondary source must never take down the hourly run.
    """
    api_key = optional_env("OPENWEATHER_API_KEY")
    if not api_key:
        return None
    params = {
        "lat": CITY.latitude,
        "lon": CITY.longitude,
        "appid": api_key,
        "units": "metric",
    }
    try:
        data = _get_with_retries(OPENWEATHER_CURRENT_URL, params).json()
    except requests.RequestException:
        return None
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        # m/s -> km/h, to match the unit Open-Meteo is configured to return.
        "wind_speed": data["wind"]["speed"] * 3.6,
        "wind_direction": data["wind"].get("deg"),
    }
