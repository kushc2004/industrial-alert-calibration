import json

import numpy as np
import pandas as pd

from industrial_alert_calibration.datasets import load_dataset
from industrial_alert_calibration.events import group_positive_runs
from industrial_alert_calibration.pipeline import PipelineConfig, run_pipeline


def test_pipeline_creates_resumable_artifacts(tmp_path):
    n = 120
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC"),
        "temperature": rng.normal(0, 1, n),
        "pressure": rng.normal(0, 1, n),
        "label": np.zeros(n, dtype=int),
    })
    frame.loc[95:100, ["temperature", "pressure"]] += 12
    frame.loc[95:100, "label"] = 1
    input_path = tmp_path / "telemetry.csv"
    frame.to_csv(input_path, index=False)
    config = PipelineConfig(input_path=str(input_path), artifacts_dir=str(tmp_path / "artifacts"), run_id="synthetic", alpha=.10)
    metrics = run_pipeline(config)
    assert metrics["event_recall"] == 1.0
    assert metrics["evaluation_rows"] > 0
    run_dir = tmp_path / "artifacts" / "synthetic"
    assert {"scores.parquet", "incidents.csv", "metrics.json", "manifest.json"} <= {p.name for p in run_dir.iterdir()}
    resumed = run_pipeline(PipelineConfig(**(config.__dict__ | {"resume": True})))
    assert resumed == json.loads((run_dir / "metrics.json").read_text())
    assert str(tmp_path) not in (run_dir / "manifest.json").read_text()


def test_swat_reference_is_not_used_as_evaluation(tmp_path):
    reference = pd.DataFrame({
        " Timestamp": pd.date_range("2015-12-28 10:00", periods=100, freq="s").strftime("%d/%m/%Y %I:%M:%S %p"),
        " FIT101": np.linspace(1.0, 1.2, 100),
        "Normal/Attack": "Normal",
    })
    attack = pd.DataFrame({
        " Timestamp": pd.date_range("2015-12-30 10:00", periods=20, freq="s").strftime("%d/%m/%Y %I:%M:%S %p"),
        " FIT101": np.r_[np.ones(10), np.full(10, 25.0)],
        "Normal/Attack": ["Normal"] * 10 + ["Attack"] * 10,
    })
    reference_path, attack_path = tmp_path / "normal.csv", tmp_path / "attack.csv"
    reference.to_csv(reference_path, index=False)
    attack.to_csv(attack_path, index=False)
    metrics = run_pipeline(PipelineConfig(
        input_path=str(attack_path), reference_path=str(reference_path), dataset="swat",
        artifacts_dir=str(tmp_path / "artifacts"), run_id="swat", alpha=.10,
    ))
    assert metrics["reference_rows"] == len(reference)
    assert metrics["evaluation_rows"] == len(attack)
    scores = pd.read_parquet(tmp_path / "artifacts" / "swat" / "scores.parquet")
    assert set(scores.iloc[:20]["split"]) == {"baseline"}
    assert set(scores.iloc[20:30]["split"]) == {"calibration"}
    assert set(scores.iloc[30:100]["split"]) == {"reference_unused"}
    assert set(scores.iloc[100:]["split"]) == {"evaluation"}


def test_metropt_raw_derives_labels_and_drops_saved_index(tmp_path):
    path = tmp_path / "metro.csv"
    pd.DataFrame({
        "Unnamed: 0": [0, 1],
        "timestamp": ["2020-04-17 23:59:50", "2020-04-18 00:00:00"],
        "TP2": [1.0, 2.0],
    }).to_csv(path, index=False)
    frame = load_dataset(path, "metropt_raw")
    assert "Unnamed: 0" not in frame
    assert frame["label"].tolist() == [0, 1]


def test_event_grouping_does_not_bridge_timestamp_gaps():
    timestamps = pd.Series(pd.to_datetime([
        "2015-12-28 10:00:00Z", "2015-12-28 10:00:01Z", "2015-12-28 10:10:00Z"
    ]))
    events = group_positive_runs(pd.Series([True, True, True]), max_gap_steps=3, min_points=1, timestamps=timestamps)
    assert [(event.start, event.end) for event in events] == [(0, 1), (2, 2)]
