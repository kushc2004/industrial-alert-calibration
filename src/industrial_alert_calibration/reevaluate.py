"""Re-evaluate saved scores without fitting models or selecting on test labels."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import file_sha256, write_json
from .calibration import conformal_p_values
from .events import (ground_truth_events, group_positive_runs, evaluate_events,
                     incidents_to_frame, persistent_alerts)


def observed_days(frame):
    delta = frame.timestamp.diff().dt.total_seconds()
    cadence = delta[delta > 0].median()
    if pd.isna(cadence):
        raise ValueError("need positive timestamp intervals")
    return float(delta.clip(lower=0, upper=cadence).sum() + cadence) / 86400


def reevaluate(source: Path, destination: Path):
    frame = pd.read_parquet(source / "scores.parquet")
    dataset = next((name for name in ["metropt_raw", "swat"] if source.name.startswith(name + "-")), None)
    if dataset is None:
        raise ValueError(f"unknown dataset for saved run: {source.name}")
    cal = frame.loc[frame.split.eq("calibration")].reset_index(drop=True)
    if not cal.label.eq(0).all():
        raise ValueError("alarm calibration must be normal-only")
    # Freeze the threshold on first half, select persistence on second half.
    halfway = len(cal) // 2
    threshold_scores = cal.score.iloc[:halfway]
    validation = cal.iloc[halfway:].reset_index(drop=True)
    validation_raw = pd.Series(conformal_p_values(validation.score, threshold_scores) <= .01)
    candidates = []
    for points in [1, 3, 10, 30, 60]:
        alerts = persistent_alerts(validation_raw, validation.timestamp, points)
        events = group_positive_runs(alerts, 0, 1, validation.timestamp)
        candidates.append({"points": points, "false_events_per_observed_day": len(events) / observed_days(validation),
                           "normal_alarm_fraction": float(alerts.mean())})
    feasible = [x for x in candidates if x["false_events_per_observed_day"] <= 1 and x["normal_alarm_fraction"] <= .01]
    selected = feasible[0] if feasible else candidates[-1]
    evaluation = frame.loc[frame.split.eq("evaluation")].reset_index(drop=True)
    raw = pd.Series(conformal_p_values(evaluation.score, threshold_scores) <= .01)
    actual = ground_truth_events(evaluation, dataset)
    destination.mkdir(parents=True, exist_ok=True)
    incidents_to_frame(actual, evaluation.timestamp).to_csv(destination / "ground_truth.csv", index=False)
    rows = []
    for policy, points in [("unfiltered", 1), ("normal_calibrated_persistence", selected["points"])]:
        alerts = persistent_alerts(raw, evaluation.timestamp, points)
        predicted = group_positive_runs(alerts, 0, 1, evaluation.timestamp)
        metrics = evaluate_events(predicted, actual, evaluation.timestamp)
        labels = evaluation.label
        metrics.update(policy=policy, persistence_points=points,
                       validation_budget_met=bool(feasible) if policy != "unfiltered" else None,
                       point_average_precision=float(average_precision_score(labels, evaluation.score)),
                       point_roc_auc=float(roc_auc_score(labels, evaluation.score)),
                       normal_false_positive_rate=float(alerts[labels.eq(0)].mean()),
                       attack_point_recall=float(alerts[labels.eq(1)].mean()),
                       observed_days=observed_days(evaluation),
                       false_events_per_observed_day=metrics["false_alert_events"] / observed_days(evaluation))
        rows.append(metrics)
        incidents_to_frame(predicted, evaluation.timestamp).to_csv(destination / f"{policy}-incidents.csv", index=False)
        evaluation[f"{policy}_alert"] = alerts
    evaluation.to_parquet(destination / "evaluated_scores.parquet", index=False)
    write_json(destination / "calibration.json", {"candidates": candidates, "selected": selected,
               "validation_budget_met": bool(feasible), "threshold_rows": halfway,
               "validation_rows": len(validation), "alpha": .01})
    write_json(destination / "metrics.json", rows)
    write_json(destination / "provenance.json", {"source_scores_sha256": file_sha256(source / "scores.parquet"),
               "source_manifest_sha256": file_sha256(source / "manifest.json"),
               "source_run": source.name, "dataset": dataset,
               "delay_reference": "first observed sample in labelled window; not maintenance-report onset",
               "evaluation_source_sha256": file_sha256(Path(__file__)),
               "events_source_sha256": file_sha256(Path(__file__).with_name("events.py"))})
    return [{"run": source.name, **x} for x in rows]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.source.rglob("scores.parquet")):
        print(f"Re-evaluating {path.parent.name}", flush=True)
        rows.extend(reevaluate(path.parent, args.destination / path.parent.name))
    if not rows:
        raise ValueError("no saved scores found")
    pd.DataFrame(rows).to_csv(args.destination / "comparison.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
