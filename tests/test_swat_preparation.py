import pandas as pd
import pytest

from kaggle.long_swat_tsfm_replay import _preflight_baseline
from industrial_alert_calibration.swat_preparation import prepare_minute_swat, prepare_minute_swat_sessions


def test_swat_minute_preparation_preserves_an_attack_in_a_mixed_minute(tmp_path):
    source = tmp_path / "swat.csv"
    pd.DataFrame(
        {
            "Timestamp": ["28/12/2015 10:00:01", "28/12/2015 10:00:30", "28/12/2015 10:01:01"],
            "Normal/Attack": ["Normal", "Attack", "Normal"],
            "LIT101": [1.0, 3.0, 5.0],
            "MV101": [0, 1, 1],
        }
    ).to_csv(source, index=False)
    prepared, metadata = prepare_minute_swat(source)
    assert len(prepared) == 2
    assert prepared.label.tolist() == [1, 0]
    assert prepared.LIT101.tolist() == [2.0, 5.0]
    assert metadata["sensor_count"] == 2


def test_swat_preparation_deduplicates_timestamp_without_losing_attack_label(tmp_path):
    source = tmp_path / "swat.csv"
    pd.DataFrame(
        {
            "Timestamp": ["28/12/2015 10:00:01", "28/12/2015 10:00:01"],
            "Normal/Attack": ["Normal", "Attack"],
            "LIT101": [1.0, 3.0],
        }
    ).to_csv(source, index=False)
    prepared, _ = prepare_minute_swat(source)
    assert prepared.label.tolist() == [1]


def test_baseline_preflight_requires_healthy_rows_after_moment_context(tmp_path):
    prepared = tmp_path / "prepared.parquet"
    pd.DataFrame({"label": [0] * 600 + [1] * 10}).to_parquet(prepared)
    report = _preflight_baseline(prepared, 0.9, 512)
    assert report["baseline_rows"] == 549
    assert report["healthy_prefix_rows"] == 600


def test_baseline_preflight_rejects_contaminated_or_short_prefix(tmp_path):
    prepared = tmp_path / "prepared.parquet"
    pd.DataFrame({"label": [0] * 600 + [1] * 400}).to_parquet(prepared)
    with pytest.raises(ValueError, match="includes labelled attacks"):
        _preflight_baseline(prepared, 0.8, 512)
    with pytest.raises(ValueError, match="MOMENT needs more than 512"):
        _preflight_baseline(prepared, 0.2, 512)


def test_separate_sessions_preserve_normal_then_attack_order(tmp_path):
    normal = tmp_path / "normal.csv"
    attack = tmp_path / "attack.csv"
    pd.DataFrame({
        "Timestamp": ["28/12/2015 10:00:00", "28/12/2015 10:01:00"],
        "Normal/Attack": ["Normal", "Normal"], "LIT101": [1.0, 2.0],
    }).to_csv(normal, index=False)
    pd.DataFrame({
        "Timestamp": ["29/12/2015 10:00:00", "29/12/2015 10:01:00"],
        "Normal/Attack": ["Normal", "Attack"], "LIT101": [3.0, 4.0],
    }).to_csv(attack, index=False)
    prepared, metadata = prepare_minute_swat_sessions(normal, attack)
    assert prepared.label.tolist() == [0, 0, 0, 1]
    assert metadata["session_mode"] == "normal_then_attack"
