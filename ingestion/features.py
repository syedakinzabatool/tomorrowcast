"""Phase 2: feature engineering for one new hourly observation, using the
history already written to the raw feature group.

- Cyclical time encodings (hour / day-of-week / month) as sin/cos pairs, so
  a model sees time as a circle (hour 23 and hour 0 are neighbors) instead
  of a number with a cliff in it.
- Wind direction is circular too (0 deg == 360 deg) and gets the same
  sin/cos treatment.
- Rate-of-change vs. the previous hour, per metric.
- Rolling means over the last 3h and 24h, per metric.
"""
from __future__ import annotations

import math

import pandas as pd

from common.config import METRICS

ROLLING_WINDOWS_HOURS = [3, 24]

# Our six brief-defined metric names -> the raw fused-observation field
# that carries that metric's value.
METRIC_TO_RAW_FIELD = {
    "aqi": "aqi",
    "temperature": "temperature",
    "precipitation": "precipitation",
    "humidity": "humidity",
    "uv_index": "uv_index",
    "wind": "wind_speed",
}


def _cyclical(value: float, period: int) -> tuple[float, float]:
    angle = 2 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def engineer_features(history: pd.DataFrame, new_row: dict) -> dict:
    """`history` = every past fused observation already in Hopsworks,
    (may be empty on the very first-ever run). `new_row` = this hour's
    fused observation dict, as produced by fuse.fuse_observation.
    """
    ts = pd.Timestamp(new_row["timestamp"])
    features: dict = {"timestamp": ts}

    hour_sin, hour_cos = _cyclical(ts.hour, 24)
    dow_sin, dow_cos = _cyclical(ts.dayofweek, 7)
    month_sin, month_cos = _cyclical(ts.month - 1, 12)
    features.update(
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        dow_sin=dow_sin,
        dow_cos=dow_cos,
        month_sin=month_sin,
        month_cos=month_cos,
    )

    # NaN, not None: a column that is None for every row gets inferred as
    # pandas/pyarrow's "null" dtype, which Hopsworks' schema mapper rejects.
    # NaN keeps the column a proper float64, with the same "missing" meaning.
    wind_dir = new_row.get("wind_direction")
    if wind_dir is not None:
        wd_sin, wd_cos = _cyclical(wind_dir, 360)
    else:
        wd_sin, wd_cos = math.nan, math.nan
    features["wind_direction_sin"] = wd_sin
    features["wind_direction_cos"] = wd_cos

    sorted_history = history.sort_values("timestamp") if not history.empty else history

    for metric in METRICS:
        field = METRIC_TO_RAW_FIELD[metric]
        current_value = new_row.get(field)
        # Always float, even when the arithmetic happens to land on a whole
        # number (e.g. a 0 roc on the very first row): Hopsworks infers and
        # locks a feature group's column type from its first insert, so a
        # column that's sometimes-int/sometimes-float would get stuck as
        # 'bigint' and later reject a genuine float/NaN value.
        features[f"{metric}_value"] = float(current_value) if current_value is not None else math.nan

        if sorted_history.empty or field not in sorted_history.columns:
            features[f"{metric}_roc_1h"] = math.nan
            for window in ROLLING_WINDOWS_HOURS:
                features[f"{metric}_roll_{window}h"] = math.nan
            continue

        series = sorted_history[field]
        prev_value = series.iloc[-1] if len(series) else None
        features[f"{metric}_roc_1h"] = (
            float(current_value - prev_value)
            if current_value is not None and prev_value is not None
            else math.nan
        )

        for window in ROLLING_WINDOWS_HOURS:
            tail = series.tail(window - 1).tolist()
            if current_value is not None:
                tail = tail + [current_value]
            tail = [v for v in tail if v is not None]
            features[f"{metric}_roll_{window}h"] = sum(tail) / len(tail) if tail else math.nan

    return features
