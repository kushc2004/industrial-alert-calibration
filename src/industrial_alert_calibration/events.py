from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Incident:
    start: int
    end: int


def group_positive_runs(
    values: pd.Series, max_gap_steps: int, min_points: int, timestamps: pd.Series | None = None
) -> list[Incident]:
    """Group positive samples, without bridging missing stretches of time.

    Some public incident files retain attack rows but omit the normal rows
    between attacks.  Positional adjacency alone would merge those distinct
    incidents, so an observed timestamp gap larger than the inferred sampling
    cadence also closes an incident.
    """
    positive = np.flatnonzero(values.to_numpy(dtype=bool))
    if len(positive) == 0:
        return []
    max_time_gap: pd.Timedelta | None = None
    if timestamps is not None:
        parsed = pd.to_datetime(timestamps, utc=True)
        cadence = parsed.diff().dropna()
        cadence = cadence[cadence > pd.Timedelta(0)]
        if not cadence.empty:
            # A low quantile recovers the nominal cadence even when this file
            # already omits long normal stretches between labelled incidents.
            max_time_gap = cadence.quantile(0.10) * (max_gap_steps + 1)
    incidents: list[Incident] = []
    start = previous = int(positive[0])
    count = 1
    for current_value in positive[1:]:
        current = int(current_value)
        time_contiguous = max_time_gap is None or timestamps.iloc[current] - timestamps.iloc[previous] <= max_time_gap
        if current - previous <= max_gap_steps + 1 and time_contiguous:
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


def evaluate_events(
    predicted: list[Incident], actual: list[Incident], timestamps: pd.Series | None = None
) -> dict[str, float | int | None]:
    matched_actual: set[int] = set()
    delays: list[int] = []
    delay_seconds: list[float] = []
    matched_predicted = 0
    for alert in predicted:
        overlaps = [idx for idx, truth in enumerate(actual) if alert.start <= truth.end and alert.end >= truth.start]
        if overlaps:
            matched_predicted += 1
            matched_actual.update(overlaps)
    # Delay is measured once per truth, only for a NEW alarm during the event.
    # An already-active alarm contributes overlap, not a zero-delay detection.
    onset_matches = 0
    for truth in actual:
        starts = [alert.start for alert in predicted if truth.start <= alert.start <= truth.end]
        if starts:
            first = min(starts)
            onset_matches += 1
            delays.append(first - truth.start)
            if timestamps is not None:
                delay_seconds.append((timestamps.iloc[first] - timestamps.iloc[truth.start]).total_seconds())
    precision = matched_predicted / len(predicted) if predicted else 0.0
    recall = len(matched_actual) / len(actual) if actual else 0.0
    return {
        "predicted_events": len(predicted), "actual_events": len(actual), "matched_events": len(matched_actual),
        "event_precision": precision, "event_recall": recall,
        "onset_event_recall": onset_matches / len(actual) if actual else 0.0,
        "onset_matched_events": onset_matches,
        "mean_detection_delay_steps": float(np.mean(delays)) if delays else None,
        "median_detection_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
        "false_alert_events": len(predicted) - matched_predicted,
    }
