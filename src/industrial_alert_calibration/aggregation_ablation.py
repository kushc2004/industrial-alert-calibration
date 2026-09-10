"""Compare diagonal and full-covariance aggregation of saved residuals.

The two streams in this module are deliberately derived from the *same* model
residual matrix and the same initial healthy calibration prefix.  This makes
the experiment an aggregation ablation, rather than a comparison between
different models, windows, or label splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .events import evaluate_events, group_positive_runs, persistent_alerts
from .operational_replay import align_score_streams, observed_days, run_operational_replay


def aggregate_residuals(residuals: np.ndarray, baseline_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal-covariance and Mahalanobis scores from one residual matrix.

    ``independent_standardized_rms`` is the root-mean-square standardized
    residual, equivalent to a diagonal-covariance distance up to a constant
    factor.  Mahalanobis additionally accounts for correlations observed only
    in the initial healthy prefix.
    """
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("residuals must be a non-empty two-dimensional matrix")
    if not 1 < baseline_rows <= len(values):
        raise ValueError("baseline_rows must be between 2 and the residual row count")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = values[:baseline_rows]
    center = baseline.mean(axis=0)
    scale = baseline.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (values - center) / scale
    independent_standardized_rms = np.sqrt(np.mean(standardized**2, axis=1))

    covariance = np.cov(baseline, rowvar=False) + np.eye(baseline.shape[1]) * 1e-6
    precision = np.linalg.pinv(covariance)
    delta = values - center
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", delta, precision, delta))
    return independent_standardized_rms, mahalanobis


def _policy_on_partition(
    scores: pd.Series, partition: pd.DataFrame, threshold: float, persistence_points: int
) -> dict:
    """Score one labelled session without treating missing inter-event time as normal."""
    alerts = persistent_alerts(scores.ge(threshold), partition.timestamp, persistence_points)
    actual = group_positive_runs(partition.label.astype(bool), 0, 1, partition.timestamp)
    predicted = group_positive_runs(alerts, 0, 1, partition.timestamp)
    event_metrics = evaluate_events(predicted, actual, partition.timestamp)
    normal_mask = partition.label.eq(0)
    normal = partition.loc[normal_mask].reset_index(drop=True)
    normal_alerts = alerts.loc[normal_mask].reset_index(drop=True)
    normal_predicted = group_positive_runs(normal_alerts, 0, 1, normal.timestamp)
    return event_metrics | {
        "threshold": float(threshold),
        "persistence_points": int(persistence_points),
        "normal_alert_fraction": float(normal_alerts.mean()) if len(normal_alerts) else None,
        "normal_observed_days": observed_days(normal.timestamp) if len(normal) > 1 else None,
        "normal_false_alert_events": len(normal_predicted),
        "normal_false_alerts_per_day": (
            len(normal_predicted) / observed_days(normal.timestamp) if len(normal) > 1 else None
        ),
    }


def _split_complete_incidents(frame: pd.DataFrame, holdout_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    incidents = group_positive_runs(frame.label.astype(bool), 0, 1, frame.timestamp)
    if len(incidents) < 4:
        raise ValueError("need at least four labelled incidents in the attack session")
    held_out = max(1, int(np.ceil(len(incidents) * holdout_fraction)))
    split_at = incidents[-held_out].start
    return frame.iloc[:split_at].reset_index(drop=True), frame.iloc[split_at:].reset_index(drop=True), len(incidents) - held_out, held_out


def run_session_aware_replay(
    frame: pd.DataFrame,
    baseline_rows: int,
    holdout_fraction: float = .50,
    min_onset_recall: float = .50,
    max_false_alerts_per_day: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate separate SWaT normal and attack sessions without inventing normal gaps.

    The public mirror omits normal samples between many labelled attack windows.
    Thus false alerts are measured on a chronologically held-out clean-normal
    block, while recall and delay are measured on later complete attack events.
    Thresholds and persistence are selected only from their earlier counterparts.
    """
    if not .05 <= holdout_fraction <= .50:
        raise ValueError("holdout_fraction must be between .05 and .50")
    first_attack = np.flatnonzero(frame.label.to_numpy(dtype=int) != 0)
    if not len(first_attack):
        raise ValueError("session-aware replay requires a labelled attack session")
    attack_start = int(first_attack[0])
    if frame.label.iloc[attack_start:].eq(0).any():
        raise ValueError("session-aware replay requires a normal-prefix then attack-session layout")
    if not 100 <= baseline_rows < attack_start - 200:
        raise ValueError("need at least 200 post-baseline normal rows for session-aware replay")

    # The initial normal prefix fits residual moments.  Split the remaining
    # normal telemetry chronologically into selection and untouched test blocks.
    available_normal = frame.iloc[baseline_rows:attack_start].reset_index(drop=True)
    normal_test_rows = max(100, int(np.ceil(len(available_normal) * holdout_fraction)))
    validation_normal = available_normal.iloc[:-normal_test_rows].reset_index(drop=True)
    held_out_normal = available_normal.iloc[-normal_test_rows:].reset_index(drop=True)
    if len(validation_normal) < 100 or len(held_out_normal) < 100:
        raise ValueError("need at least 100 normal rows in both selection and held-out normal blocks")
    validation_attack, held_out_attack, calibration_incidents, held_out_incidents = _split_complete_incidents(
        frame.iloc[attack_start:].reset_index(drop=True), holdout_fraction
    )

    configurations = [column for column in frame.columns if column not in {"timestamp", "label"}]
    rows: list[dict] = []
    candidates: dict[str, list[dict]] = {}
    for configuration in configurations:
        normal_scores = validation_normal[configuration]
        thresholds = sorted({float(normal_scores.quantile(q)) for q in (.90, .95, .975, .99, .995, .999)})
        all_candidates: list[dict] = []
        for threshold in thresholds:
            for points in (1, 3, 5, 10, 30, 60):
                normal_metrics = _policy_on_partition(
                    validation_normal[configuration], validation_normal, threshold, points
                )
                attack_metrics = _policy_on_partition(
                    validation_attack[configuration], validation_attack, threshold, points
                )
                all_candidates.append({
                    "threshold": threshold,
                    "persistence_points": points,
                    "validation_onset_event_recall": attack_metrics["onset_event_recall"],
                    "validation_onset_matched_events": attack_metrics["onset_matched_events"],
                    "validation_attack_events": attack_metrics["actual_events"],
                    "validation_median_detection_delay_seconds": attack_metrics["median_detection_delay_seconds"],
                    "validation_normal_false_alert_events": normal_metrics["normal_false_alert_events"],
                    "validation_normal_false_alerts_per_day": normal_metrics["normal_false_alerts_per_day"],
                })
        eligible = [item for item in all_candidates if (
            item["validation_onset_event_recall"] >= min_onset_recall
            and item["validation_normal_false_alerts_per_day"] <= max_false_alerts_per_day
        )]
        selected = min(
            eligible or all_candidates,
            key=lambda item: (
                -item["validation_onset_event_recall"],
                item["validation_normal_false_alerts_per_day"],
                item["validation_median_detection_delay_seconds"]
                if item["validation_median_detection_delay_seconds"] is not None else float("inf"),
            ),
        )
        normal_test = _policy_on_partition(
            held_out_normal[configuration], held_out_normal,
            selected["threshold"], selected["persistence_points"],
        )
        attack_test = _policy_on_partition(
            held_out_attack[configuration], held_out_attack,
            selected["threshold"], selected["persistence_points"],
        )
        rows.append({
            "configuration": configuration,
            "validation_gate_met": bool(eligible),
            "selection_reason": "met_recall_and_alert_budget" if eligible else "no_policy_met_validation_gate",
            **selected,
            "held_out_attack_events": attack_test["actual_events"],
            "held_out_attack_onset_matched_events": attack_test["onset_matched_events"],
            "held_out_attack_onset_event_recall": attack_test["onset_event_recall"],
            "held_out_attack_median_detection_delay_seconds": attack_test["median_detection_delay_seconds"],
            "held_out_normal_false_alert_events": normal_test["normal_false_alert_events"],
            "held_out_normal_observed_days": normal_test["normal_observed_days"],
            "held_out_normal_false_alerts_per_day": normal_test["normal_false_alerts_per_day"],
        })
        candidates[configuration] = all_candidates
    comparison = pd.DataFrame(rows).sort_values(
        ["validation_gate_met", "held_out_attack_onset_event_recall", "held_out_normal_false_alerts_per_day"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary = {
        "protocol": "session_aware_alert_budget_matched_replay",
        "protocol_note": "False-alert rates use a held-out normal session block; attack recall and delay use later complete attack events because the public attack mirror omits normal telemetry between many events.",
        "configuration_count": len(configurations),
        "baseline_normal_rows": baseline_rows,
        "validation_normal_rows": len(validation_normal),
        "held_out_normal_rows": len(held_out_normal),
        "calibration_incidents": calibration_incidents,
        "held_out_incidents": held_out_incidents,
        "minimum_validation_onset_recall": min_onset_recall,
        "maximum_validation_false_alerts_per_day": max_false_alerts_per_day,
        "winner": comparison.iloc[0].configuration if bool(comparison.iloc[0].validation_gate_met) else None,
        "winner_requires_validation_gate": True,
        "candidate_policies": candidates,
    }
    return comparison, summary


def write_aggregation_ablation(
    residual_path: Path,
    labels_path: Path,
    output_dir: Path,
    baseline_rows: int,
    warmup_rows: int,
    holdout_fraction: float = .30,
    min_onset_recall: float = .50,
    max_false_alerts_per_day: float = 1.0,
    session_aware: bool = False,
) -> dict:
    """Write two score streams and their one-time chronological replay results."""
    residual_frame = pd.read_parquet(residual_path).sort_values("Timestamp").reset_index(drop=True)
    sensors = [column for column in residual_frame if column != "Timestamp"]
    if not sensors:
        raise ValueError("residual artifact contains no sensor residual columns")
    if residual_frame.Timestamp.duplicated().any():
        raise ValueError("residual artifact has duplicate timestamps")
    if len(residual_frame) <= warmup_rows:
        raise ValueError("residual artifact is shorter than the shared warm-up")
    independent, mahalanobis = aggregate_residuals(residual_frame[sensors].to_numpy(), baseline_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = residual_frame.Timestamp.iloc[warmup_rows:].to_numpy()
    score_paths = {
        "independent_standardized_rms": output_dir / "independent_standardized_rms.csv",
        "mahalanobis": output_dir / "mahalanobis.csv",
    }
    for name, score in (("independent_standardized_rms", independent), ("mahalanobis", mahalanobis)):
        destination = score_paths[name]
        temporary = destination.with_suffix(".csv.tmp")
        pd.DataFrame({"Timestamp": timestamp, "Overall_AnomalyScore": score[warmup_rows:]}).to_csv(
            temporary, index=False
        )
        temporary.replace(destination)

    replay_frame = align_score_streams(labels_path, score_paths)
    if session_aware:
        comparison, summary = run_session_aware_replay(
            replay_frame,
            baseline_rows=baseline_rows - warmup_rows,
            holdout_fraction=holdout_fraction,
            min_onset_recall=min_onset_recall,
            max_false_alerts_per_day=max_false_alerts_per_day,
        )
    else:
        comparison, summary = run_operational_replay(
            replay_frame,
            holdout_fraction=holdout_fraction,
            min_onset_recall=min_onset_recall,
            max_false_alerts_per_day=max_false_alerts_per_day,
        )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    manifest = {
        "residual_path": str(residual_path),
        "residual_rows": len(residual_frame),
        "sensor_count": len(sensors),
        "baseline_rows": baseline_rows,
        "warmup_rows": warmup_rows,
        "session_aware": session_aware,
        "independent_aggregation": "RMS of residuals standardized by initial healthy baseline mean and standard deviation (diagonal covariance)",
        "mahalanobis_aggregation": "Full-covariance Mahalanobis distance fit only on initial healthy baseline residuals",
    }
    write_json(output_dir / "ablation_manifest.json", manifest)
    write_json(output_dir / "metrics.json", summary | {"configurations": comparison.to_dict(orient="records"), **manifest})
    return {"comparison": comparison, "summary": summary, "manifest": manifest}
