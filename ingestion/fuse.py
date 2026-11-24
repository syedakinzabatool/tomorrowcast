"""Phase 2: fuse the primary (Open-Meteo) and secondary (OpenWeather)
observations into one row, following a documented, non-silent imputation
strategy. For each fused field, in order:

1. Open-Meteo has a value -> use it (source = "open-meteo").
2. Open-Meteo is missing it, OpenWeather has it -> use OpenWeather
   (source = "openweather"). Only fields OpenWeather actually reports
   (temperature, humidity, wind) are eligible for this step.
3. Still missing, and we have a previous observation no more than
   MAX_FORWARD_FILL_HOURS old -> forward-fill that last known value
   (source = "forward-fill", imputed = True).
4. Still missing -> leave it as None and mark imputed = True. We never
   silently guess a value with no basis for it.
"""
from __future__ import annotations

import pandas as pd

MAX_FORWARD_FILL_HOURS = 3

FUSED_FIELDS = [
    "temperature",
    "humidity",
    "precipitation",
    "wind_speed",
    "wind_direction",
    "uv_index",
    "aqi",
]

# Only fields OpenWeather's free "current weather" endpoint actually supplies.
OPENWEATHER_FALLBACK_FIELDS = {"temperature", "humidity", "wind_speed", "wind_direction"}


def fuse_observation(
    weather: dict,
    air_quality: dict,
    openweather: dict | None,
    last_row: dict | None,
    hours_since_last: float | None,
) -> dict:
    fused: dict = {"timestamp": pd.Timestamp(weather["timestamp"])}
    merged_primary = {**weather, **air_quality}

    for field in FUSED_FIELDS:
        value = merged_primary.get(field)
        source = "open-meteo"
        imputed = False

        if value is None and openweather and field in OPENWEATHER_FALLBACK_FIELDS:
            value = openweather.get(field)
            source = "openweather"

        if (
            value is None
            and last_row is not None
            and hours_since_last is not None
            and hours_since_last <= MAX_FORWARD_FILL_HOURS
        ):
            value = last_row.get(field)
            source = "forward-fill"
            imputed = True

        if value is None:
            source = "none"
            imputed = True

        fused[field] = value
        fused[f"{field}_source"] = source
        fused[f"{field}_imputed"] = imputed

    return fused
