from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

DatasetPreset = Literal["generic", "metropt", "metropt_raw", "chiller", "swat"]


METROPT_FAILURE_WINDOWS = (
    ("2020-04-18 00:00:00", "2020-04-18 23:59:59"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
)


def load_dataset(path: Path, preset: DatasetPreset) -> pd.DataFrame:
    """Load a supported telemetry file into the pipeline's standard schema."""
    if preset == "generic":
        return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)

    frame = pd.read_csv(path)
    if preset == "metropt":
        if "timestamp" not in frame or "label" not in frame:
            raise ValueError("metropt input must contain timestamp and label columns")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], dayfirst=True, errors="raise", utc=True)
        return frame
    if preset == "metropt_raw":
        if "timestamp" not in frame:
            raise ValueError("metropt_raw input must contain a timestamp column")
        # The public CSV includes a saved row-index column.  It is not telemetry
        # and would otherwise create a spurious time-trend feature.
        frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
        label = pd.Series(False, index=frame.index)
        for start, end in METROPT_FAILURE_WINDOWS:
            label |= frame["timestamp"].between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
        frame["label"] = label.astype(int)
        return frame
    if preset == "chiller":
        if "Date" not in frame or "label" not in frame:
            raise ValueError("chiller input must contain Date and label columns")
        frame = frame.rename(columns={"Date": "timestamp"})
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], format="mixed", dayfirst=True, errors="raise", utc=True
        )
        return frame
    if preset == "swat":
        frame.columns = frame.columns.str.strip()
        if "Timestamp" not in frame or "Normal/Attack" not in frame:
            raise ValueError("swat input must contain Timestamp and Normal/Attack columns")
        frame = frame.rename(columns={"Timestamp": "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"].str.strip(), dayfirst=True, errors="raise", utc=True)
        frame["label"] = frame["Normal/Attack"].astype(str).str.strip().eq("Attack").astype(int)
        frame = frame.drop(columns=["Normal/Attack"])
        for column in frame.columns.drop("timestamp"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        # This public mirror concatenates overlapping normal-operation chunks.
        # Canonicalize the time series before any baseline or calibration split.
        return frame.sort_values("timestamp").drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    raise ValueError(f"unsupported dataset preset: {preset}")
