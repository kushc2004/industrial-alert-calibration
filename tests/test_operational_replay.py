import numpy as np
import pandas as pd

from industrial_alert_calibration.operational_replay import run_operational_replay, select_alert_policy


def _frame(test_labels: list[int] | None = None) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=600, freq="min", tz="UTC")
    labels = np.zeros(600, dtype=int)
    for start in (80, 180, 280, 380, 480, 560):
        labels[start : start + 8] = 1
    if test_labels is not None:
        labels[480:] = test_labels
    score = labels.astype(float) + np.linspace(0, .01, len(labels))
    return pd.DataFrame({"timestamp": timestamps, "label": labels, "forecast": score, "reconstruction": score * .8})


def test_policy_selection_uses_only_the_earlier_calibration_slice():
    frame = _frame()
    calibration = frame.iloc[:480].copy()
    selected, _ = select_alert_policy(frame.forecast.iloc[:480], calibration, max_false_alerts_per_day=10)
    altered_later_labels = frame.copy()
    altered_later_labels.loc[altered_later_labels.index >= 480, "label"] = 0
    repeated, _ = select_alert_policy(
        altered_later_labels.forecast.iloc[:480], altered_later_labels.iloc[:480], max_false_alerts_per_day=10
    )
    assert selected["threshold"] == repeated["threshold"]
    assert selected["persistence_points"] == repeated["persistence_points"]


def test_replay_reports_a_configuration_count_and_gate():
    comparison, summary = run_operational_replay(_frame(), max_false_alerts_per_day=10)
    assert summary["configuration_count"] == 2
    assert len(comparison) == 2
    assert {"forecast", "reconstruction"} == set(comparison.configuration)
    assert comparison.validation_gate_met.all()
