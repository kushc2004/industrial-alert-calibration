"""Canonical, label-preserving preparation of public SWaT telemetry."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path, header=1)
    return pd.read_csv(path)


def canonicalize_swat(path: Path) -> pd.DataFrame:
    """Return sorted SWaT rows with numeric sensors and a binary outcome label.

    Labels are retained solely for later offline evaluation.  This function does
    not use them to transform, impute, or score any sensor values.
    """
    frame = _read_table(path)
    frame.columns = frame.columns.astype(str).str.strip()
    if "Timestamp" not in frame or "Normal/Attack" not in frame:
        raise ValueError("SWaT input must contain Timestamp and Normal/Attack columns")
    frame = frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()
    frame["Timestamp"] = pd.to_datetime(
        frame["Timestamp"].astype(str).str.strip(), dayfirst=True, errors="raise", utc=True
    )
    status = frame["Normal/Attack"].astype(str).str.replace(r"\s+", "", regex=True).str.casefold()
    unknown = ~status.isin({"normal", "attack"})
    if unknown.any():
        examples = sorted(status.loc[unknown].unique())[:3]
        raise ValueError(f"unrecognised SWaT Normal/Attack values: {examples}")
    frame["label"] = status.eq("attack").astype("int8")
    frame = frame.drop(columns=["Normal/Attack"])
    sensors = [column for column in frame.columns if column not in {"Timestamp", "label"}]
    for column in sensors:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    all_missing = [column for column in sensors if frame[column].isna().all()]
    if all_missing:
        frame = frame.drop(columns=all_missing)
        sensors = [column for column in sensors if column not in all_missing]
    if not sensors:
        raise ValueError("SWaT input contains no numeric sensor columns")
    # Mirrors can contain repeated rows after normal/attack files are combined.
    # Repeated timestamps should agree; if not, retain an attack label so an
    # episode cannot disappear during canonicalisation.
    frame = frame.sort_values("Timestamp")
    aggregated = {column: "mean" for column in sensors}
    aggregated["label"] = "max"
    return frame.groupby("Timestamp", as_index=False, sort=True).agg(aggregated)


def prepare_minute_swat(path: Path, cadence: str = "1min") -> tuple[pd.DataFrame, dict]:
    """Aggregate SWaT telemetry to a fixed cadence without diluting attack labels."""
    frame = canonicalize_swat(path)
    sensors = [column for column in frame.columns if column not in {"Timestamp", "label"}]
    indexed = frame.set_index("Timestamp")
    aggregations = {column: "mean" for column in sensors}
    aggregations["label"] = "max"
    prepared = indexed.resample(cadence).agg(aggregations).dropna(subset=sensors, how="all").reset_index()
    # Forward-fill uses past information only.  It makes the input rectangular
    # for model inference while avoiding future-value imputation.
    prepared[sensors] = prepared[sensors].ffill()
    prepared = prepared.dropna(subset=sensors).reset_index(drop=True)
    prepared["label"] = prepared["label"].astype("int8")
    metadata = {
        "cadence": cadence,
        "raw_rows_after_deduplication": int(len(frame)),
        "prepared_rows": int(len(prepared)),
        "sensor_count": len(sensors),
        "start": prepared["Timestamp"].iloc[0].isoformat(),
        "end": prepared["Timestamp"].iloc[-1].isoformat(),
        "attack_minutes": int(prepared["label"].sum()),
        "label_use": "offline evaluation only; no label-derived feature transformation or scoring",
    }
    return prepared, metadata


def write_prepared_swat(source: Path, output: Path, cadence: str = "1min") -> dict:
    prepared, metadata = prepare_minute_swat(source, cadence)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
