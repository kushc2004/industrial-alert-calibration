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


def _resample_swat(frame: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Aggregate one uninterrupted telemetry session without diluting labels."""
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
    return prepared


def _metadata(
    prepared: pd.DataFrame, raw_rows: int, cadence: str, session_mode: str, **extra: object
) -> dict:
    sensors = [column for column in prepared.columns if column not in {"Timestamp", "label"}]
    metadata = {
        "cadence": cadence,
        "session_mode": session_mode,
        "raw_rows_after_deduplication": raw_rows,
        "prepared_rows": int(len(prepared)),
        "sensor_count": len(sensors),
        "start": prepared["Timestamp"].iloc[0].isoformat(),
        "end": prepared["Timestamp"].iloc[-1].isoformat(),
        "attack_minutes": int(prepared["label"].sum()),
        "label_use": "offline evaluation only; no label-derived feature transformation or scoring",
    }
    metadata.update(extra)
    return metadata


def prepare_minute_swat(path: Path, cadence: str = "1min") -> tuple[pd.DataFrame, dict]:
    """Aggregate one public SWaT file to a fixed cadence."""
    frame = canonicalize_swat(path)
    prepared = _resample_swat(frame, cadence)
    return prepared, _metadata(prepared, len(frame), cadence, "single_file")


def prepare_minute_swat_sessions(normal_path: Path, attack_path: Path, cadence: str = "1min") -> tuple[pd.DataFrame, dict]:
    """Build a deployment-style normal-then-attack replay from separate sessions.

    The public release's merged mirror is not guaranteed to preserve this
    session order.  Each source is resampled independently, then concatenated
    as the intended operating sequence: known healthy commissioning data,
    followed by a later attack evaluation session.
    """
    normal = canonicalize_swat(normal_path)
    attack = canonicalize_swat(attack_path)
    if normal["label"].ne(0).any():
        raise ValueError("normal session contains attack labels")
    normal_prepared = _resample_swat(normal, cadence)
    attack_prepared = _resample_swat(attack, cadence)
    # The two public files are independent collection sessions and their
    # wall-clock ranges may overlap.  For an offline deployment replay their
    # sequence, rather than their unrelated absolute clocks, is causal: train
    # and calibrate on the completed healthy session, then evaluate the later
    # attack session.  Place the latter immediately after the former so alert
    # rates are measured over observed telemetry, not an arbitrary calendar
    # gap.  Sensor values and labels are unchanged.
    cadence_delta = pd.to_timedelta(cadence)
    original_attack_start = attack_prepared["Timestamp"].iloc[0]
    original_attack_end = attack_prepared["Timestamp"].iloc[-1]
    replay_attack_start = normal_prepared["Timestamp"].iloc[-1] + cadence_delta
    offset = replay_attack_start - original_attack_start
    attack_prepared = attack_prepared.copy()
    attack_prepared["Timestamp"] = attack_prepared["Timestamp"] + offset
    prepared = pd.concat([normal_prepared, attack_prepared], ignore_index=True)
    if not prepared["Timestamp"].is_monotonic_increasing:
        raise ValueError("session replay clock is not monotonic after normalization")
    return prepared, _metadata(
        prepared,
        len(normal) + len(attack),
        cadence,
        "normal_then_attack_replay_clock",
        normal_session_start=normal_prepared["Timestamp"].iloc[0].isoformat(),
        normal_session_end=normal_prepared["Timestamp"].iloc[-1].isoformat(),
        attack_session_start=original_attack_start.isoformat(),
        attack_session_end=original_attack_end.isoformat(),
        replay_attack_start=replay_attack_start.isoformat(),
        attack_timestamp_offset_seconds=float(offset.total_seconds()),
        replay_clock_note=(
            "Attack-session timestamps were shifted only to create a contiguous "
            "normal-then-attack offline replay; sensor values and labels were unchanged."
        ),
    )


def write_prepared_swat(source: Path, output: Path, cadence: str = "1min") -> dict:
    prepared, metadata = prepare_minute_swat(source, cadence)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def write_prepared_swat_sessions(normal_source: Path, attack_source: Path, output: Path, cadence: str = "1min") -> dict:
    prepared, metadata = prepare_minute_swat_sessions(normal_source, attack_source, cadence)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(output, index=False)
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
