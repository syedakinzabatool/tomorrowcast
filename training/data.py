"""Phase 5: load training data for all 6 metrics from the Phase 3 feature view.

`lahore_training_view` joins lahore_engineered_features (X) to lahore_targets
(y = target_<metric>, 24h-ahead raw values) on timestamp. We ask Hopsworks for
an explicit time-based train/validation/test split (not random) so a model
never trains on data from after the point it's evaluated on -- shuffling would
leak future information into training for a time series.
"""
from __future__ import annotations

import pandas as pd

from common.config import METRICS
from common.hopsworks_client import get_feature_store

FEATURE_VIEW_NAME = "lahore_training_view"
FEATURE_VIEW_VERSION = 1

# Train on the older ~19 months, validate on 3, test on the most recent ~3.
# Boundaries line up exactly (one split's end == the next split's start) so
# no hour is silently dropped or duplicated between splits.
TRAIN_END = "2026-06-01"
VALIDATION_END = "2026-09-01"


def _split_X_y(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with no real target yet (the newest 24h of the full range,
    flagged by Phase 3's target_available column) and tidy up types/order.
    """
    combined = X.join(y)
    combined = combined[combined["target_available"].fillna(False)].copy()
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    X_clean = combined.drop(columns=["target_available"] + [f"target_{m}" for m in METRICS])
    y_clean = combined[[f"target_{m}" for m in METRICS]]
    return X_clean, y_clean


def load_train_val_test() -> dict[str, pd.DataFrame]:
    fs = get_feature_store()
    fv = fs.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)

    X_train, X_val, X_test, y_train, y_val, y_test = fv.train_validation_test_split(
        train_start="",
        train_end=TRAIN_END,
        validation_start=TRAIN_END,
        validation_end=VALIDATION_END,
        test_start=VALIDATION_END,
        test_end="",
        event_time=True,
        description="Phase 5 time-based split: train < 2026-06-01 <= val < 2026-09-01 <= test",
    )

    X_train, y_train = _split_X_y(X_train, y_train)
    X_val, y_val = _split_X_y(X_val, y_val)
    X_test, y_test = _split_X_y(X_test, y_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


if __name__ == "__main__":
    data = load_train_val_test()
    for split in ["train", "val", "test"]:
        X, y = data[f"X_{split}"], data[f"y_{split}"]
        print(f"{split}: {len(X)} rows, {X['timestamp'].min()} to {X['timestamp'].max()}")
