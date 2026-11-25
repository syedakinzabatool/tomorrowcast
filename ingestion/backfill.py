"""Phase 3: one-time historical backfill.

Builds 1-2 years of hourly history for Lahore by reusing the same
primary/secondary-source philosophy as the live hourly ingestion
(ingestion/run_ingestion.py), but as one big vectorized pandas job instead
of a per-row loop -- a per-row loop re-sorting a growing history frame
17,000+ times would be O(n^2) and painfully slow for a job this size.

Data sources:
- Open-Meteo **archive-api** (ERA5 reanalysis): temperature, humidity,
  precipitation, wind speed/direction, air quality (us_aqi). Confirmed via
  a live test fetch to have complete (zero-gap) 2-year coverage for Lahore.
- Open-Meteo's archive-api has NO uv_index variable at all (confirmed:
  every value comes back null, even for recent dates) -- ERA5 reanalysis
  simply doesn't model it. Two-part workaround:
  1. For the last ~92 days, Open-Meteo's *forecast* API can return real
     historical uv_index via `past_days` -- we use the genuine value there.
  2. For anything older, we derive uv_index from `shortwave_radiation`
     (which the archive-api *does* have) using a ratio empirically
     calibrated against those same ~92 real days for this exact location:
     median(uv_index / shortwave_radiation) = 0.00789 across 586 daylight
     hourly samples. This is a documented approximation, not a measurement
     -- every derived row is flagged uv_index_source="derived-shortwave",
     uv_index_imputed=True, same "never silently guess" rule as fuse.py.
- No OpenWeather here: its free tier has no historical endpoint, so the
  secondary-source fallback that exists for live ingestion simply doesn't
  apply to backfill.

Run directly: python -m ingestion.backfill
"""
from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np
import pandas as pd

from common.config import CITY, METRICS
from common.hopsworks_client import get_feature_store, get_or_create_feature_group
from ingestion.features import METRIC_TO_RAW_FIELD, ROLLING_WINDOWS_HOURS
from ingestion.fuse import FUSED_FIELDS, MAX_FORWARD_FILL_HOURS
from ingestion.run_ingestion import FEATURES_FG_NAME, FG_VERSION, RAW_FG_NAME
from ingestion.sources import _get_with_retries  # retry wrapper, reused as-is

ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
ARCHIVE_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ARCHIVE_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
]

# Calibrated once against 586 real (uv_index, shortwave_radiation) pairs
# from Open-Meteo's forecast API for this location (see module docstring).
UV_FROM_SHORTWAVE_RATIO = 0.00789

# Open-Meteo reports these three fields as JSON integers (not floats), and
# Phase 2's very first live row into lahore_raw_observations happened to
# carry them as whole numbers too -- so Hopsworks locked this feature
# group's schema with these columns as 'bigint'. Our vectorized backfill
# computes everything as float64, so these three need to be cast back to
# integer right before the raw_fg write, specifically, to match that
# already-locked production schema (the features_fg schema is unaffected:
# engineer_history_features always casts to float, matching how Phase 2's
# per-row features.py already wrote 'double' columns there).
RAW_INTEGER_FIELDS = ["humidity", "wind_direction", "aqi"]

TARGETS_FG_NAME = "lahore_targets"
TRAINING_FV_NAME = "lahore_training_view"
TARGET_HORIZON_HOURS = 24

# Each insert() call launches its own Spark materialization job on the
# (shared, free-tier) Hopsworks cluster -- that job, not the data upload
# itself, is what's flaky and resource-sensitive. Smaller chunks mean more
# total job launches, which costs more shared compute, not less: a 4000-
# row chunk size here previously triggered 12 job executions (5 failed)
# for one 17.5k-row feature group alone. So we now default to ONE insert
# per feature group (chunk size >= any realistic row count) to minimize
# job launches, and keep the chunking machinery only as a manual fallback
# if a single big insert can't get through.
INSERT_CHUNK_ROWS = 25000
CHUNK_RETRY_ATTEMPTS = 2
CHUNK_RETRY_BACKOFF_SECONDS = 20

# Pause between feature groups' inserts so one job's resources are fully
# released on the Hopsworks cluster before the next job launches, instead
# of firing back-to-back and risking resource contention between them.
BETWEEN_FG_PAUSE_SECONDS = 30


def _existing_timestamps(fg) -> set:
    """Timestamps already committed to this feature group, if any. Used to
    keep the backfill strictly additive -- it must never upsert over rows
    that are already there, whether from Phase 2's live hourly ingestion or
    an earlier backfill run. This also sidesteps a real failure we hit
    while testing: the one chunk whose range overlapped already-ingested
    live rows had its materialization job fail 3/3 times (every other,
    non-overlapping chunk succeeded), which points at Hudi upsert-over-
    existing-keys being the trigger on this free-tier cluster -- plain
    inserts of genuinely new rows don't hit it.
    """
    try:
        existing = fg.read()
    except Exception:
        return set()
    if existing.empty:
        return set()
    return set(pd.to_datetime(existing["timestamp"]).dt.tz_localize(None))


def _drop_existing(df: pd.DataFrame, existing_timestamps: set, label: str) -> pd.DataFrame:
    if not existing_timestamps:
        return df
    mask = df["timestamp"].isin(existing_timestamps)
    skipped = int(mask.sum())
    if skipped:
        print(f"{label}: {skipped} row(s) already present (e.g. from live ingestion) -- skipping, not overwriting.")
    return df[~mask].reset_index(drop=True)


def _insert_in_chunks(fg, df: pd.DataFrame, write_options: dict, label: str) -> None:
    total = len(df)
    if total == 0:
        print(f"{label}: nothing new to insert.")
        return
    for start in range(0, total, INSERT_CHUNK_ROWS):
        chunk = df.iloc[start : start + INSERT_CHUNK_ROWS]
        chunk_num = start // INSERT_CHUNK_ROWS + 1
        total_chunks = (total + INSERT_CHUNK_ROWS - 1) // INSERT_CHUNK_ROWS
        last_error: Exception | None = None
        for attempt in range(CHUNK_RETRY_ATTEMPTS):
            try:
                # wait=True: block until this chunk's materialization job
                # finishes before returning. Without it, insert() is async
                # (fire-and-forget) -- firing the next chunk immediately
                # launches a second Spark job against the same Hudi table
                # while the first is still running, and the two collide
                # (this is exactly what happened in testing: two concurrent
                # jobs on one table, one SUCCEEDED and one FAILED).
                fg.insert(chunk, write_options=write_options, wait=True)
                print(f"{label}: chunk {chunk_num}/{total_chunks} ({len(chunk)} rows) inserted.")
                break
            except Exception as exc:
                last_error = exc
                print(f"{label}: chunk {chunk_num}/{total_chunks} failed (attempt {attempt + 1}): {exc}")
                if attempt < CHUNK_RETRY_ATTEMPTS - 1:
                    time.sleep(CHUNK_RETRY_BACKOFF_SECONDS * (attempt + 1))
        else:
            raise RuntimeError(f"{label}: chunk {chunk_num}/{total_chunks} failed after {CHUNK_RETRY_ATTEMPTS} attempts") from last_error


def fetch_archive_weather(start_date: str, end_date: str) -> pd.DataFrame:
    resp = _get_with_retries(
        ARCHIVE_WEATHER_URL,
        {
            "latitude": CITY.latitude,
            "longitude": CITY.longitude,
            "hourly": ",".join(ARCHIVE_WEATHER_VARS),
            "wind_speed_unit": "kmh",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        },
    )
    hourly = resp.json()["hourly"]
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "humidity": hourly["relative_humidity_2m"],
            "precipitation": hourly["precipitation"],
            "wind_speed": hourly["wind_speed_10m"],
            "wind_direction": hourly["wind_direction_10m"],
            "shortwave_radiation": hourly["shortwave_radiation"],
        }
    )


def fetch_archive_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    resp = _get_with_retries(
        ARCHIVE_AIR_QUALITY_URL,
        {
            "latitude": CITY.latitude,
            "longitude": CITY.longitude,
            "hourly": "us_aqi",
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    hourly = resp.json()["hourly"]
    return pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"]), "aqi": hourly["us_aqi"]})


def fetch_recent_real_uv_index(past_days: int = 92) -> pd.DataFrame:
    """Real (not derived) uv_index for roughly the last `past_days` days,
    via the forecast API's `past_days` param -- the only Open-Meteo
    endpoint that actually models this variable historically.
    """
    resp = _get_with_retries(
        FORECAST_URL,
        {
            "latitude": CITY.latitude,
            "longitude": CITY.longitude,
            "hourly": "uv_index",
            "past_days": past_days,
            "forecast_days": 1,
            "timezone": "UTC",
        },
    )
    hourly = resp.json()["hourly"]
    return pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"]), "uv_index_real": hourly["uv_index"]})


def fuse_history(weather: pd.DataFrame, air_quality: pd.DataFrame, real_uv: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of fuse.fuse_observation applied to an entire
    historical range at once: same priority rule (measured value -> bounded
    forward-fill -> none), same *_source / *_imputed columns per field.
    """
    df = weather.merge(air_quality, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)

    # uv_index isn't fetched from the archive at all (see module docstring);
    # it starts as all-missing here and gets filled in just below from the
    # real/derived overlay, before going through the same forward-fill rule
    # as every other field.
    df["uv_index"] = np.where(
        df["shortwave_radiation"].notna(),
        df["shortwave_radiation"] * UV_FROM_SHORTWAVE_RATIO,
        np.nan,
    )
    df = df.merge(real_uv, on="timestamp", how="left")
    uv_is_real = df["uv_index_real"].notna()
    df["uv_index"] = np.where(uv_is_real, df["uv_index_real"], df["uv_index"])

    fused = pd.DataFrame({"timestamp": df["timestamp"]})
    for field in FUSED_FIELDS:
        raw = df[field].astype(float)
        source = pd.Series("open-meteo", index=df.index, dtype=object)
        if field == "uv_index":
            source = np.where(uv_is_real, "open-meteo", "derived-shortwave")
            source = pd.Series(source, index=df.index, dtype=object)

        filled = raw.ffill(limit=MAX_FORWARD_FILL_HOURS)
        was_filled = raw.isna() & filled.notna()
        source = source.where(~was_filled, "forward-fill")

        still_missing = filled.isna()
        source = source.where(~still_missing, "none")
        imputed = was_filled | still_missing | (field == "uv_index")

        fused[field] = filled
        fused[f"{field}_source"] = source
        fused[f"{field}_imputed"] = imputed

    return fused


def engineer_history_features(fused: pd.DataFrame) -> pd.DataFrame:
    """Vectorized equivalent of features.engineer_features applied to an
    entire historical range -- same formulas (cyclical encodings, 1h
    rate-of-change, 3h/24h rolling means), computed with pandas' native
    diff()/rolling() instead of a per-row Python loop.
    """
    ts = fused["timestamp"]
    features = pd.DataFrame({"timestamp": ts})

    def cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
        angle = 2 * math.pi * values / period
        return np.sin(angle), np.cos(angle)

    features["hour_sin"], features["hour_cos"] = cyclical(ts.dt.hour, 24)
    features["dow_sin"], features["dow_cos"] = cyclical(ts.dt.dayofweek, 7)
    features["month_sin"], features["month_cos"] = cyclical(ts.dt.month - 1, 12)
    features["wind_direction_sin"], features["wind_direction_cos"] = cyclical(fused["wind_direction"], 360)

    for metric in METRICS:
        field = METRIC_TO_RAW_FIELD[metric]
        series = fused[field].astype(float)
        features[f"{metric}_value"] = series
        features[f"{metric}_roc_1h"] = series.diff()
        for window in ROLLING_WINDOWS_HOURS:
            features[f"{metric}_roll_{window}h"] = series.rolling(window=window, min_periods=1).mean()

    return features


def build_targets(fused: pd.DataFrame) -> pd.DataFrame:
    """target_<metric>[t] = raw observation of that metric at t+24h. The
    newest TARGET_HORIZON_HOURS rows have no future observation yet within
    this backfilled range, so target_available=False for them -- flagged,
    not dropped, so the feature view can filter them out explicitly rather
    than them silently vanishing.
    """
    targets = pd.DataFrame({"timestamp": fused["timestamp"]})
    for metric in METRICS:
        field = METRIC_TO_RAW_FIELD[metric]
        targets[f"target_{metric}"] = fused[field].astype(float).shift(-TARGET_HORIZON_HOURS)

    last_available_ts = fused["timestamp"].max() - pd.Timedelta(hours=TARGET_HORIZON_HOURS)
    targets["target_available"] = fused["timestamp"] <= last_available_ts
    return targets


def run(start_date: str, end_date: str) -> dict:
    print(f"Fetching Open-Meteo archive weather + air quality for {start_date}..{end_date}...")
    weather = fetch_archive_weather(start_date, end_date)
    air_quality = fetch_archive_air_quality(start_date, end_date)
    real_uv = fetch_recent_real_uv_index(past_days=92)
    print(f"Fetched {len(weather)} weather rows, {len(air_quality)} air-quality rows.")

    fused = fuse_history(weather, air_quality, real_uv)
    features = engineer_history_features(fused)
    targets = build_targets(fused)
    print(f"Fused + engineered {len(fused)} hourly rows.")

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
    targets_fg = get_or_create_feature_group(
        name=TARGETS_FG_NAME,
        version=FG_VERSION,
        primary_key=["timestamp"],
        event_time="timestamp",
        description="24h-ahead training targets per metric, shifted from lahore_raw_observations.",
    )

    fused_for_raw = fused.copy()
    for field in RAW_INTEGER_FIELDS:
        fused_for_raw[field] = fused_for_raw[field].round().astype("Int64")

    raw_existing = _existing_timestamps(raw_fg)
    features_existing = _existing_timestamps(features_fg)
    targets_existing = _existing_timestamps(targets_fg)

    fused_for_raw = _drop_existing(fused_for_raw, raw_existing, RAW_FG_NAME)
    features = _drop_existing(features, features_existing, FEATURES_FG_NAME)
    targets = _drop_existing(targets, targets_existing, TARGETS_FG_NAME)

    write_options = {"kafka_timeout": 30}
    _insert_in_chunks(raw_fg, fused_for_raw, write_options, RAW_FG_NAME)
    if len(fused_for_raw) and len(features):
        time.sleep(BETWEEN_FG_PAUSE_SECONDS)
    _insert_in_chunks(features_fg, features, write_options, FEATURES_FG_NAME)
    if len(features) and len(targets):
        time.sleep(BETWEEN_FG_PAUSE_SECONDS)
    _insert_in_chunks(targets_fg, targets, write_options, TARGETS_FG_NAME)
    print(f"Inserted {len(fused_for_raw)}/{len(features)}/{len(targets)} new rows into {RAW_FG_NAME}/{FEATURES_FG_NAME}/{TARGETS_FG_NAME}.")

    fs = get_feature_store()
    query = features_fg.select_all().join(
        targets_fg.select_except(["timestamp"]),
        on=["timestamp"],
    )
    fs.get_or_create_feature_view(
        name=TRAINING_FV_NAME,
        version=FG_VERSION,
        query=query,
        description="Engineered features joined with 24h-ahead targets, for Phase 5 training.",
        labels=[f"target_{metric}" for metric in METRICS],
    )
    print(f"Feature view '{TRAINING_FV_NAME}' ready.")

    return {"rows": len(fused), "fused": fused, "features": features, "targets": targets}


if __name__ == "__main__":
    end = (dt.date.today() - dt.timedelta(days=2)).isoformat()
    start = (dt.date.today() - dt.timedelta(days=730)).isoformat()
    run(start, end)
