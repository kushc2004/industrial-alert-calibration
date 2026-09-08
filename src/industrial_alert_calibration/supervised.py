"""Chronological supervised incident-detection evaluation.

This is deliberately separate from the label-free anomaly pipeline.  It models
known historical attack/fault signatures and evaluates only later, whole
incidents.  Test labels are never used for fitting or threshold selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .events import (evaluate_events, group_positive_runs, incidents_to_frame,
                     persistent_alerts)
from .reevaluate import observed_days


@dataclass(frozen=True)
class ChronologicalSplit:
    fit_end: int
    test_start: int
    test_events: int


def _policy_metrics(
    scores: pd.Series,
    label: pd.Series,
    timestamp: pd.Series,
    threshold: float,
    persistence_points: int,
) -> tuple[dict, pd.Series]:
    """Evaluate one causal threshold/persistence alert policy."""
    alerts = persistent_alerts(scores.ge(threshold), timestamp, persistence_points)
    actual = group_positive_runs(label.astype(bool), 0, 1, timestamp)
    predicted = group_positive_runs(alerts, 0, 1, timestamp)
    metrics = evaluate_events(predicted, actual, timestamp)
    metrics |= {
        "threshold": float(threshold),
        "persistence_points": int(persistence_points),
        "normal_alert_fraction": float(alerts[label.eq(0)].mean()),
        "false_events_per_observed_day": metrics["false_alert_events"] / observed_days(
            pd.DataFrame({"timestamp": timestamp})
        ),
    }
    return metrics, alerts


def select_event_aware_policy(
    calibration_scores: pd.Series,
    calibration: pd.DataFrame,
    min_onset_recall: float = .50,
    max_false_events_per_day: float = 1.0,
) -> tuple[dict, list[dict]]:
    """Choose a policy using earlier labelled events, never test labels.

    The selected policy is deliberately constrained by an operational false
    alert budget and optimised for *new-onset* incident recall.  This is a
    supervised alert-calibration step, not an unsupervised detector threshold.
    """
    normal_scores = calibration_scores.loc[calibration.label.eq(0)]
    if len(normal_scores) < 100:
        raise ValueError("need at least 100 normal rows for policy selection")
    thresholds = sorted({float(normal_scores.quantile(q)) for q in (.90, .95, .975, .99, .995, .999)})
    candidates: list[dict] = []
    for threshold in thresholds:
        for points in (1, 3, 10, 30, 60):
            metrics, _ = _policy_metrics(
                calibration_scores, calibration.label, calibration.timestamp, threshold, points
            )
            candidates.append(metrics)

    eligible = [
        candidate for candidate in candidates
        if candidate["onset_event_recall"] >= min_onset_recall
        and candidate["false_events_per_observed_day"] <= max_false_events_per_day
    ]
    if eligible:
        selected = min(
            eligible,
            key=lambda x: (x["false_events_per_observed_day"], -x["onset_event_recall"],
                           x["median_detection_delay_seconds"] or float("inf")),
        )
        reason = "met_recall_and_false_alert_budget"
    else:
        # A failed budget must remain explicit rather than silently reporting a
        # cherry-picked recall/alert-rate trade-off as a successful policy.
        selected = min(
            candidates,
            key=lambda x: (-x["onset_event_recall"], x["false_events_per_observed_day"],
                           x["median_detection_delay_seconds"] or float("inf")),
        )
        reason = "no_policy_met_validation_gate"
    return selected | {
        "selection_reason": reason,
        "validation_gate_met": bool(eligible),
        "minimum_validation_onset_recall": min_onset_recall,
        "maximum_validation_false_events_per_day": max_false_events_per_day,
    }, candidates


def chronological_incident_split(frame: pd.DataFrame, holdout_fraction: float = .30) -> ChronologicalSplit:
    """Reserve the final whole labelled incidents and all subsequent rows for test."""
    events = group_positive_runs(frame["label"].astype(bool), 0, 1, frame["timestamp"])
    if len(events) < 4:
        raise ValueError("need at least four labelled incidents for a chronological holdout")
    holdout_count = max(1, int(np.ceil(len(events) * holdout_fraction)))
    test_start = events[-holdout_count].start
    # Reserve the final 20% of pre-test chronology for normal-only calibration.
    fit_end = int(test_start * .80)
    if fit_end < 100 or frame.label.iloc[:fit_end].nunique() < 2:
        raise ValueError("earlier chronology must contain both normal and labelled incidents")
    return ChronologicalSplit(fit_end=fit_end, test_start=test_start, test_events=holdout_count)


def _features(frame: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "label"}
    columns = [c for c in frame.select_dtypes(include="number").columns if c not in excluded]
    if not columns:
        raise ValueError("no numeric telemetry features")
    return columns


def run_supervised_incident_benchmark(frame: pd.DataFrame, holdout_fraction: float = .30) -> tuple[dict, pd.DataFrame, object]:
    """Fit early history, tune event-aware policy on middle history, test once.

    The classifier is intentionally a tabular baseline.  It answers whether
    known fault signatures transfer to later incidents, rather than claiming
    detection of novel, unseen anomaly types.
    """
    split = chronological_incident_split(frame, holdout_fraction)
    features = _features(frame)
    train = frame.iloc[:split.fit_end].copy()
    calibration = frame.iloc[split.fit_end:split.test_start].copy()
    test = frame.iloc[split.test_start:].reset_index(drop=True).copy()
    normal_calibration = calibration.loc[calibration.label.eq(0)]
    if len(normal_calibration) < 100:
        raise ValueError("need at least 100 normal calibration samples")

    # Equal class weights prevent the vast normal class from trivialising the
    # historical-incident classifier.  Sampling is deterministic for reruns.
    positive = train.loc[train.label.eq(1)]
    negative = train.loc[train.label.eq(0)]
    sample_size = min(len(negative), max(len(positive) * 4, 5_000))
    sampled_train = pd.concat([positive, negative.sample(sample_size, random_state=42)]).sample(
        frac=1, random_state=42
    )
    model = HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1.0,
        random_state=42,
    )
    model.fit(sampled_train[features], sampled_train.label)

    calibration_scores = pd.Series(model.predict_proba(calibration[features])[:, 1]).reset_index(drop=True)
    selected, candidates = select_event_aware_policy(calibration_scores, calibration.reset_index(drop=True))

    scores = pd.Series(model.predict_proba(test[features])[:, 1])
    metrics, alerts = _policy_metrics(
        scores, test.label, test.timestamp, selected["threshold"], selected["persistence_points"]
    )
    # The baseline changes only persistence; its threshold is identical to the
    # selected calibrated policy, making any false-alert reduction attributable
    # to the causal event-aware confirmation rule rather than a hidden cutoff.
    baseline_metrics, _ = _policy_metrics(scores, test.label, test.timestamp, selected["threshold"], 1)
    metrics |= {
        "protocol": "supervised_event_aware_alert_calibration",
        "fit_rows": len(train), "normal_calibration_rows": len(normal_calibration),
        "test_rows": len(test), "test_incidents": metrics["actual_events"],
        "event_aware_policy": selected,
        "unfiltered_same_threshold": baseline_metrics,
        "false_alert_event_reduction_vs_unfiltered": (
            1 - metrics["false_events_per_observed_day"] / baseline_metrics["false_events_per_observed_day"]
            if baseline_metrics["false_events_per_observed_day"] else None
        ),
        "attack_point_recall": float(alerts[test.label.eq(1)].mean()),
        "point_average_precision": float(average_precision_score(test.label, scores)),
        "point_roc_auc": float(roc_auc_score(test.label, scores)),
        "false_events_per_observed_day": metrics["false_alert_events"] / observed_days(test),
        "test_starts_at": str(test.timestamp.iloc[0]),
    }
    output = test[["timestamp", "label"]].copy()
    output["score"] = scores
    output["point_alert"] = alerts
    output["split"] = "held_out_later_incidents"
    return metrics | {"calibration_candidates": candidates}, output, model
