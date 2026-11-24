"""Phase 2: the hourly ingestion run.

fetch (sources.py) -> fuse (fuse.py) -> engineer features (features.py)
-> write one row to each of two Hopsworks feature groups:
  - lahore_raw_observations:     the fused raw values, source + imputed flags
  - lahore_engineered_features:  everything derived from the raw row

Run directly: python -m ingestion.run_ingestion
In Phase 7 this becomes the first step of the hourly GitHub Actions workflow.
"""
from __future__ import annotations

import pandas as pd

from common.hopsworks_client import get_or_create_feature_group
from ingestion.features import engineer_features
from ingestion.fuse import fuse_observation
from ingestion.sources import (
    fetch_open_meteo_air_quality,
    fetch_open_meteo_weather,
    fetch_openweather_current,
)

RAW_FG_NAME = "lahore_raw_observations"
FEATURES_FG_NAME = "lahore_engineered_features"
FG_VERSION = 1


def _to_naive_utc(timestamp) -> pd.Timestamp:
    """Hopsworks hands back timezone-aware (UTC) timestamps on read, while
    Open-Meteo's raw strings are naive. Normalize both to naive-UTC so they
    can be compared/subtracted without pandas raising on the mismatch.
    """
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _hours_since(last_timestamp, now_timestamp) -> float | None:
    if last_timestamp is None:
        return None
    return (_to_naive_utc(now_timestamp) - _to_naive_utc(last_timestamp)).total_seconds() / 3600


def run() -> dict:
    weather = fetch_open_meteo_weather()
    air_quality = fetch_open_meteo_air_quality()
    openweather = fetch_openweather_current()
    print(f"OpenWeather secondary source: {'available' if openweather else 'skipped (no key / call failed)'}")

    raw_fg = get_or_create_feature_group(
        name=RAW_FG_NAME,
        version=FG_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly fused raw observations for Lahore (Open-Meteo primary + OpenWeather secondary).",
    )
    features_fg = get_or_create_feature_group(
        name=FEATURES_FG_NAME,
        version=FG_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="Hourly engineered features derived from lahore_raw_observations.",
    )

    try:
        history = raw_fg.read()
    except Exception:
        history = pd.DataFrame()

    try:
        features_history = features_fg.read()
    except Exception:
        features_history = pd.DataFrame()

    # Strictly-earlier history only. A retry after a partial failure (raw
    # written, features write failed) finds this hour's own row already in
    # raw_fg; without this filter it would compare the new observation
    # against itself as its own "previous hour" and roll into itself.
    new_ts = _to_naive_utc(weather["timestamp"])
    if not history.empty:
        history = history.sort_values("timestamp")
        history = history[history["timestamp"].apply(_to_naive_utc) < new_ts]

    last_row = None
    hours_since_last = None
    if not history.empty:
        last_row = history.iloc[-1].to_dict()
        hours_since_last = _hours_since(last_row["timestamp"], weather["timestamp"])

    fused = fuse_observation(weather, air_quality, openweather, last_row, hours_since_last)

    # Check the *features* table, not the raw one: if a previous run wrote
    # the raw row but failed before writing features (as happened on the
    # very first run here), we must retry rather than skip.
    if not features_history.empty and fused["timestamp"] <= _to_naive_utc(features_history["timestamp"].max()):
        print(f"Already have a fully-ingested observation for {fused['timestamp']} UTC; skipping duplicate write.")
        return {"skipped": True, "timestamp": fused["timestamp"]}

    features = engineer_features(history, fused)

    # kafka_timeout: the client's default (6s) can be too tight for the SSL
    # handshake to the Kafka brokers Hopsworks writes land in; this only
    # affects how long we wait for topic metadata, not data correctness.
    write_options = {"kafka_timeout": 30}
    raw_fg.insert(pd.DataFrame([fused]), write_options=write_options)
    features_fg.insert(pd.DataFrame([features]), write_options=write_options)

    print(f"Ingested observation for {fused['timestamp']} UTC.")
    return {"skipped": False, "raw": fused, "features": features}


if __name__ == "__main__":
    run()
