import numpy as np
import pandas as pd

from industrial_alert_calibration.events import Incident, evaluate_events
from industrial_alert_calibration.scoring import robust_multivariate_score, isolation_forest_score, temporal_residual_score


def test_persistent_alarm_is_not_new_detection():
    metrics = evaluate_events([Incident(0, 99)], [Incident(10, 20), Incident(40, 50)])
    assert metrics["event_recall"] == 1
    assert metrics["onset_event_recall"] == 0
    assert metrics["mean_detection_delay_steps"] is None


def test_delay_once_per_truth():
    metrics = evaluate_events([Incident(12, 14), Incident(18, 20)], [Incident(10, 20)])
    assert metrics["mean_detection_delay_steps"] == 2


def test_missing_values_do_not_use_future():
    frame = pd.DataFrame({"x": np.arange(30, dtype=float)})
    frame.loc[15, "x"] = np.nan
    original = robust_multivariate_score(frame, ["x"], 10)
    frame.loc[16:, "x"] = 9999
    changed = robust_multivariate_score(frame, ["x"], 10)
    np.testing.assert_array_equal(original.iloc[:16], changed.iloc[:16])


def test_forest_saved_and_future_independent(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(100, 3)))
    first = isolation_forest_score(frame, list(frame.columns), 50, tmp_path / "first.joblib")
    frame.iloc[80:] = 1000
    second = isolation_forest_score(frame, list(frame.columns), 50, tmp_path / "second.joblib")
    np.testing.assert_array_equal(first.iloc[:80], second.iloc[:80])
    assert (tmp_path / "first.joblib").exists()


def test_temporal_residual_saved_and_future_independent(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(7).normal(size=(100, 3)))
    first = temporal_residual_score(frame, list(frame.columns), 50, tmp_path / "first.joblib")
    frame.iloc[80:] = 1000
    second = temporal_residual_score(frame, list(frame.columns), 50, tmp_path / "second.joblib")
    np.testing.assert_array_equal(first.iloc[:80], second.iloc[:80])
    assert (tmp_path / "first.joblib").exists()
