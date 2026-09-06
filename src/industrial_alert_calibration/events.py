from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Incident:
    start: int
    end: int


def group_positive_runs(values: pd.Series, max_gap_steps: int, min_points: int) -> list[Incident]:
    positive = np.flatnonzero(values.to_numpy(dtype=bool))
    if len(positive) == 0:
        return []
    incidents: list[Incident] = []
    start = previous = int(positive[0])
    count = 1
    for current_value in positive[1:]:
        current = int(current_value)
        if current - previous <= max_gap_steps + 1:
            previous = current
            count += 1
            continue
        if count >= min_points:
            incidents.append(Incident(start, previous))
        start = previous = current
        count = 1
    if count >= min_points:
        incidents.append(Incident(start, previous))
    return incidents


def incidents_to_frame(incidents: list[Incident], timestamps: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"incident_id": idx + 1, "start_index": event.start, "end_index": event.end,
             "start_time": timestamps.iloc[event.start], "end_time": timestamps.iloc[event.end],
             "duration_points": event.end - event.start + 1}
            for idx, event in enumerate(incidents)
        ]
    )


def evaluate_events(predicted: list[Incident], actual: list[Incident]) -> dict[str, float | int | None]:
    matched_actual: set[int] = set()
    delays: list[int] = []
    matched_predicted = 0
    for alert in predicted:
        overlaps = [idx for idx, truth in enumerate(actual) if alert.start <= truth.end and alert.end >= truth.start]
        if overlaps:
            matched_predicted += 1
            idx = overlaps[0]
            matched_actual.add(idx)
            delays.append(max(0, alert.start - actual[idx].start))
    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = len(matched_actual) / len(actual) if actual else 0.0
    return {
        "predicted_events": len(predicted), "actual_events": len(actual), "matched_events": len(matched_actual),
        "event_precision": precision, "event_recall": recall,
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else None,
        "false_alert_events": len(predicted) - matched_predicted,
    }
