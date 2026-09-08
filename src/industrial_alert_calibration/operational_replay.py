"""Chronological, alert-budget-matched replay for saved anomaly score streams.

The module deliberately evaluates *configurations*, not a fictional collection
of unrelated models.  A configuration can be a forecasting, reconstruction or
few-shot mode of the same underlying model.  Scores are aligned to the common
timestamps first, then all alert-policy choices use only chronology before the
final labelled incidents.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .events import evaluate_events, group_positive_runs, persistent_alerts


@dataclass(frozen=True)
class ReplaySplit:
    """The final complete incidents reserved for one untouched evaluation."""

    test_start: int
    calibration_incidents: int
    test_incidents: int


def observed_days(timestamps: pd.Series) -> float:
    """Observed duration without counting unobserved gaps as normal operation."""
    delta = timestamps.diff().dt.total_seconds()
    cadence = delta[delta > 0].median()
    if pd.isna(cadence):
        raise ValueError("need positive timestamp intervals")
    return float(delta.clip(lower=0, upper=cadence).sum() + cadence) / 86400


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_labels(path: Path, timestamp_column: str, label_column: str) -> pd.DataFrame:
    frame = _read_table(path)
    missing = {timestamp_column, label_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"label file is missing columns: {sorted(missing)}")
    output = frame[[timestamp_column, label_column]].rename(
        columns={timestamp_column: "timestamp", label_column: "label"}
    )
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="raise")
    output["label"] = pd.to_numeric(output["label"], errors="raise").astype(int).ne(0).astype(int)
    if output.timestamp.duplicated().any():
        raise ValueError("label file has duplicate timestamps")
    return output.sort_values("timestamp").reset_index(drop=True)


def _normalise_scores(path: Path, score_column: str) -> pd.DataFrame:
    frame = _read_table(path)
    missing = {"Timestamp", score_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"score file {path} is missing columns: {sorted(missing)}")
    output = frame[["Timestamp", score_column]].rename(columns={"Timestamp": "timestamp", score_column: "score"})
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="raise")
    output["score"] = pd.to_numeric(output["score"], errors="raise")
    if output.timestamp.duplicated().any():
        raise ValueError(f"score file has duplicate timestamps: {path}")
    return output.sort_values("timestamp").reset_index(drop=True)


def align_score_streams(
    labels_path: Path,
    score_paths: dict[str, Path],
    timestamp_column: str = "Timestamp",
    label_column: str = "label",
    score_column: str = "Overall_AnomalyScore",
) -> pd.DataFrame:
    """Return labels and every score stream on exactly the same timestamps."""
    if len(score_paths) < 2:
        raise ValueError("need at least two score configurations for a comparison")
    aligned = _normalise_labels(labels_path, timestamp_column, label_column)
    for name, path in score_paths.items():
        if not name.replace("_", "").isalnum():
            raise ValueError(f"invalid configuration name: {name!r}")
        stream = _normalise_scores(path, score_column).rename(columns={"score": name})
        aligned = aligned.merge(stream, on="timestamp", how="inner", validate="one_to_one")
    if len(aligned) < 200:
        raise ValueError("fewer than 200 common timestamped rows after alignment")
    if aligned.label.nunique() < 2:
        raise ValueError("common timestamps must contain both normal and labelled rows")
    return aligned.sort_values("timestamp").reset_index(drop=True)


def chronological_event_split(frame: pd.DataFrame, holdout_fraction: float = .30) -> ReplaySplit:
    """Keep the final fraction of *whole* labelled incidents for test."""
    if not .05 <= holdout_fraction <= .50:
        raise ValueError("holdout_fraction must be between .05 and .50")
    incidents = group_positive_runs(frame.label.astype(bool), 0, 1, frame.timestamp)
    if len(incidents) < 4:
        raise ValueError("need at least four labelled incidents after score alignment")
    test_incidents = max(1, int(np.ceil(len(incidents) * holdout_fraction)))
    test_start = incidents[-test_incidents].start
    if frame.iloc[:test_start].label.eq(0).sum() < 100:
        raise ValueError("need at least 100 earlier normal rows for alert calibration")
    return ReplaySplit(test_start, len(incidents) - test_incidents, test_incidents)


def _policy_metrics(
    scores: pd.Series, labels: pd.Series, timestamps: pd.Series, threshold: float, persistence_points: int
) -> dict[str, float | int | None]:
    alerts = persistent_alerts(scores.ge(threshold), timestamps, persistence_points)
    actual = group_positive_runs(labels.astype(bool), 0, 1, timestamps)
    predicted = group_positive_runs(alerts, 0, 1, timestamps)
    metrics = evaluate_events(predicted, actual, timestamps)
    return metrics | {
        "threshold": float(threshold),
        "persistence_points": int(persistence_points),
        "alert_points": int(alerts.sum()),
        "normal_alert_fraction": float(alerts[labels.eq(0)].mean()),
        "observed_days": observed_days(timestamps),
        "false_alerts_per_day": metrics["false_alert_events"]
        / observed_days(timestamps),
    }


def select_alert_policy(
    scores: pd.Series,
    calibration: pd.DataFrame,
    min_onset_recall: float = .50,
    max_false_alerts_per_day: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Select a causal policy with earlier labels and a fixed alert budget."""
    normal_scores = scores.loc[calibration.label.eq(0)]
    if len(normal_scores) < 100:
        raise ValueError("need at least 100 normal scores to select a policy")
    thresholds = sorted({float(normal_scores.quantile(q)) for q in (.90, .95, .975, .99, .995, .999)})
    candidates = [
        _policy_metrics(scores, calibration.label, calibration.timestamp, threshold, points)
        for threshold in thresholds
        for points in (1, 3, 5, 10, 30, 60)
    ]
    eligible = [
        item for item in candidates
        if item["onset_event_recall"] >= min_onset_recall
        and item["false_alerts_per_day"] <= max_false_alerts_per_day
    ]
    # Policies that miss the selection gate remain auditable but cannot be
    # described as deployed or included in the held-out winner ranking.
    pool = eligible or candidates
    selected = min(
        pool,
        key=lambda item: (
            -item["onset_event_recall"], item["false_alerts_per_day"],
            item["median_detection_delay_seconds"] if item["median_detection_delay_seconds"] is not None else float("inf"),
        ),
    )
    return selected | {
        "validation_gate_met": bool(eligible),
        "selection_reason": "met_recall_and_alert_budget" if eligible else "no_policy_met_validation_gate",
        "minimum_validation_onset_recall": min_onset_recall,
        "maximum_validation_false_alerts_per_day": max_false_alerts_per_day,
    }, candidates


def run_operational_replay(
    frame: pd.DataFrame,
    holdout_fraction: float = .30,
    min_onset_recall: float = .50,
    max_false_alerts_per_day: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Calibrate every configuration on earlier data and evaluate it once later."""
    split = chronological_event_split(frame, holdout_fraction)
    calibration = frame.iloc[:split.test_start].reset_index(drop=True)
    test = frame.iloc[split.test_start:].reset_index(drop=True)
    rows: list[dict] = []
    candidates: dict[str, list[dict]] = {}
    configurations = [column for column in frame.columns if column not in {"timestamp", "label"}]
    for configuration in configurations:
        selected, all_candidates = select_alert_policy(
            calibration[configuration], calibration, min_onset_recall, max_false_alerts_per_day
        )
        held_out = _policy_metrics(
            test[configuration], test.label, test.timestamp,
            selected["threshold"], selected["persistence_points"],
        )
        held_out |= {
            "configuration": configuration,
            "validation_gate_met": selected["validation_gate_met"],
            "selection_reason": selected["selection_reason"],
            "selection_threshold": selected["threshold"],
            "selection_persistence_points": selected["persistence_points"],
            "validation_onset_event_recall": selected["onset_event_recall"],
            "validation_false_alerts_per_day": selected["false_alerts_per_day"],
        }
        rows.append(held_out)
        candidates[configuration] = all_candidates
    comparison = pd.DataFrame(rows).sort_values(
        ["validation_gate_met", "onset_event_recall", "false_alerts_per_day"], ascending=[False, False, True]
    ).reset_index(drop=True)
    summary = {
        "protocol": "chronological_alert_budget_matched_operational_replay",
        "configuration_count": len(configurations),
        "common_timestamp_rows": len(frame),
        "common_timestamp_start": str(frame.timestamp.iloc[0]),
        "common_timestamp_end": str(frame.timestamp.iloc[-1]),
        "test_starts_at": str(test.timestamp.iloc[0]),
        "calibration_incidents": split.calibration_incidents,
        "held_out_incidents": split.test_incidents,
        "minimum_validation_onset_recall": min_onset_recall,
        "maximum_validation_false_alerts_per_day": max_false_alerts_per_day,
        "winner": comparison.iloc[0].configuration if bool(comparison.iloc[0].validation_gate_met) else None,
        "winner_requires_validation_gate": True,
        "candidate_policies": candidates,
    }
    return comparison, summary


def _parse_score_paths(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("score paths must use CONFIGURATION=/path/to/scores.csv")
        name, raw_path = value.split("=", 1)
        if name in parsed:
            raise ValueError(f"configuration was supplied twice: {name}")
        parsed[name] = Path(raw_path)
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay timestamped anomaly score configurations fairly.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--score", action="append", required=True, help="NAME=/path/to/score.csv; repeat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp-column", default="Timestamp")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="Overall_AnomalyScore")
    parser.add_argument("--holdout-fraction", type=float, default=.30)
    parser.add_argument("--min-onset-recall", type=float, default=.50)
    parser.add_argument("--max-false-alerts-per-day", type=float, default=1.0)
    args = parser.parse_args()
    frame = align_score_streams(
        args.labels, _parse_score_paths(args.score), args.timestamp_column, args.label_column, args.score_column
    )
    comparison, summary = run_operational_replay(
        frame, args.holdout_fraction, args.min_onset_recall, args.max_false_alerts_per_day
    )
    args.output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output / "comparison.csv", index=False)
    write_json(args.output / "metrics.json", summary | {"configurations": comparison.to_dict(orient="records")})
    print(comparison.to_string(index=False))
    print(f"Wrote replay artifacts to {args.output}")


if __name__ == "__main__":
    main()
