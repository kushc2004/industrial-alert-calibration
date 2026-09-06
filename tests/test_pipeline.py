import json

import numpy as np
import pandas as pd

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
