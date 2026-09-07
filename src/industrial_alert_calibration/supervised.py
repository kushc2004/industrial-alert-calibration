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
    """Fit on early labelled history, calibrate on later normal history, test once.

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

    calibration_score = pd.Series(model.predict_proba(normal_calibration[features])[:, 1])
    # Threshold is selected from normal data only.  Persistence selection also
    # sees only normal calibration rows, never held-out attack labels.
    threshold = float(calibration_score.quantile(.999))
    raw_calibration = calibration_score.ge(threshold)
    candidates = []
    for points in [1, 3, 10, 30, 60]:
        alerts = persistent_alerts(raw_calibration, normal_calibration.timestamp.reset_index(drop=True), points)
        events = group_positive_runs(alerts, 0, 1, normal_calibration.timestamp.reset_index(drop=True))
        candidates.append({"points": points, "normal_alert_fraction": float(alerts.mean()),
                           "false_events_per_day": len(events) / observed_days(normal_calibration)})
    feasible = [x for x in candidates if x["normal_alert_fraction"] <= .01 and x["false_events_per_day"] <= 1]
    selected = feasible[0] if feasible else candidates[-1]

    scores = pd.Series(model.predict_proba(test[features])[:, 1])
    alerts = persistent_alerts(scores.ge(threshold), test.timestamp, selected["points"])
    actual = group_positive_runs(test.label.astype(bool), 0, 1, test.timestamp)
    predicted = group_positive_runs(alerts, 0, 1, test.timestamp)
    metrics = evaluate_events(predicted, actual, test.timestamp)
    metrics |= {
        "protocol": "supervised_historical_incident_detection",
        "fit_rows": len(train), "normal_calibration_rows": len(normal_calibration),
        "test_rows": len(test), "test_incidents": len(actual), "threshold": threshold,
        "persistence_points": selected["points"], "calibration_budget_met": bool(feasible),
        "normal_alert_fraction": float(alerts[test.label.eq(0)].mean()),
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
