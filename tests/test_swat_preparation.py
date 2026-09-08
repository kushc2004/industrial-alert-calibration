import pandas as pd

from industrial_alert_calibration.swat_preparation import prepare_minute_swat


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
