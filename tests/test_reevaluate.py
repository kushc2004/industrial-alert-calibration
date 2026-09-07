import numpy as np
import pandas as pd

from industrial_alert_calibration.events import ground_truth_events, persistent_alerts
from industrial_alert_calibration.reevaluate import reevaluate


def test_metropt_window_survives_missing_samples():
    frame = pd.DataFrame({"timestamp": pd.to_datetime([
        "2020-04-18 00:00:00Z", "2020-04-18 00:00:11Z", "2020-04-18 15:00:00Z",
        "2020-04-19 00:00:00Z", "2020-05-30 00:00:00Z"]), "label": [1, 1, 1, 0, 1]})
    assert [(x.start, x.end) for x in ground_truth_events(frame, "metropt_raw")] == [(0, 2), (4, 4)]


def test_persistence_never_backdates_or_bridges_gaps():
    timestamps = pd.Series(pd.to_datetime([
        "2020-01-01 00:00:00Z", "2020-01-01 00:00:01Z", "2020-01-01 00:00:02Z",
        "2020-01-01 01:00:00Z", "2020-01-01 01:00:01Z", "2020-01-01 01:00:02Z"]))
    assert persistent_alerts(pd.Series([True] * 6), timestamps, 3).tolist() == [False, False, True, False, False, True]


def test_evaluation_labels_do_not_select_policy(tmp_path):
    source = tmp_path / "swat-test"
    source.mkdir()
    frame = pd.DataFrame({"timestamp": pd.date_range("2020", periods=300, freq="s", tz="UTC"),
                          "score": np.random.default_rng(2).normal(size=300),
                          "split": ["calibration"] * 200 + ["evaluation"] * 100,
                          "label": [0] * 250 + [1] * 50})
    (source / "manifest.json").write_text('{}')
    frame.to_parquet(source / "scores.parquet")
    reevaluate(source, tmp_path / "first")
    frame.loc[200:, "label"] = [1] * 50 + [0] * 50
    frame.to_parquet(source / "scores.parquet")
    reevaluate(source, tmp_path / "second")
    assert (tmp_path / "first/calibration.json").read_text() == (tmp_path / "second/calibration.json").read_text()
