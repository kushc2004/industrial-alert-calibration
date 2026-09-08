import numpy as np
import pandas as pd

from industrial_alert_calibration.classical_baselines import score_classical_baselines


def _telemetry() -> pd.DataFrame:
    rows = 500
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC"),
        "label": np.zeros(rows, dtype=int),
        "flow": rng.normal(size=rows),
        "pressure": rng.normal(size=rows),
    })
    frame.loc[420:, "label"] = 1
    return frame


def test_classical_scores_do_not_change_when_later_data_or_labels_change():
    original = _telemetry()
    changed = original.copy()
    changed.loc[changed.index >= 400, ["flow", "pressure"]] += 1_000
    changed.loc[changed.index >= 400, "label"] = 0
    boundary = original.timestamp.iloc[400]
    first, first_metadata = score_classical_baselines(original, boundary, rolling_window=30)
    second, second_metadata = score_classical_baselines(changed, boundary, rolling_window=30)
    pd.testing.assert_frame_equal(first.iloc[:400], second.iloc[:400])
    assert first_metadata == second_metadata
    assert first_metadata["rolling_score_is_causal"]
