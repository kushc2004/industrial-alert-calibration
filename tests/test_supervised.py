import numpy as np
import pandas as pd

from industrial_alert_calibration.supervised import chronological_incident_split, run_supervised_incident_benchmark


def test_supervised_benchmark_holds_out_final_whole_incidents():
    n = 1_000
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
        "sensor": np.arange(n) % 7,
        "label": np.zeros(n, dtype=int),
    })
    for start in [100, 250, 400, 550, 700, 850]:
        frame.loc[start:start + 10, "label"] = 1
        frame.loc[start:start + 10, "sensor"] = 99
    split = chronological_incident_split(frame)
    assert split.test_start == 700
    metrics, scores, _ = run_supervised_incident_benchmark(frame)
    assert scores.timestamp.min() == frame.timestamp.iloc[700]
    assert metrics["test_incidents"] == 2
    assert metrics["fit_rows"] < 700
