from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import file_sha256, read_json, write_json
from .calibration import conformal_p_values
from .datasets import DatasetPreset, load_dataset
from .events import evaluate_events, group_positive_runs, incidents_to_frame
from .scoring import robust_multivariate_score


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    dataset: DatasetPreset = "generic"
    timestamp_column: str = "timestamp"
    label_column: str | None = "label"
    score_column: str | None = None
    baseline_fraction: float = 0.20
    calibration_fraction: float = 0.30
    alpha: float = 0.01
    max_gap_steps: int = 3
    min_event_points: int = 2
    run_id: str = "default"
    artifacts_dir: str = "artifacts"
    resume: bool = False


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    source = Path(config.input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    run_dir = Path(config.artifacts_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    fingerprint = file_sha256(source)
    config_dict = asdict(config) | {
        "input_path": source.name,
        "artifacts_dir": Path(config.artifacts_dir).name,
        "resume": False,
    }
    if config.resume and manifest_path.exists():
        previous = read_json(manifest_path)
        if previous.get("input_sha256") == fingerprint and previous.get("config") == config_dict and previous.get("status") == "complete":
            return read_json(run_dir / "metrics.json")
        if previous.get("input_sha256") != fingerprint or previous.get("config") != config_dict:
            raise ValueError("cannot resume: input or configuration changed; use a new run-id")

    write_json(manifest_path, {"status": "running", "input_sha256": fingerprint, "config": config_dict})
    frame = load_dataset(source, config.dataset).reset_index(drop=True)
    if config.timestamp_column not in frame:
        raise ValueError(f"missing timestamp column: {config.timestamp_column}")
    frame[config.timestamp_column] = pd.to_datetime(frame[config.timestamp_column], utc=True)
    if not frame[config.timestamp_column].is_monotonic_increasing:
        raise ValueError("timestamps must be in chronological ascending order")
    baseline_end = int(len(frame) * config.baseline_fraction)
    calibration_end = int(len(frame) * config.calibration_fraction)
    if not 0 < config.baseline_fraction < config.calibration_fraction < 1:
        raise ValueError("require 0 < baseline_fraction < calibration_fraction < 1")
    if baseline_end < 10 or calibration_end - baseline_end < 10 or calibration_end >= len(frame):
        raise ValueError("fractions must leave 10 baseline rows, 10 calibration rows, and one evaluation row")
    excluded = {config.timestamp_column, config.label_column, config.score_column}
    features = [column for column in frame.select_dtypes(include="number").columns if column not in excluded]
    scores = frame[config.score_column].astype(float) if config.score_column else robust_multivariate_score(frame, features, baseline_end)
    p_values = conformal_p_values(scores, scores.iloc[baseline_end:calibration_end])
    output = pd.DataFrame({"timestamp": frame[config.timestamp_column], "score": scores, "p_value": p_values})
    output["split"] = "evaluation"
    output.loc[: baseline_end - 1, "split"] = "baseline"
    output.loc[baseline_end: calibration_end - 1, "split"] = "calibration"
    output["point_alert"] = output["p_value"] <= config.alpha
    if config.label_column and config.label_column in frame:
        output["label"] = frame[config.label_column].astype(int)
    output.to_parquet(run_dir / "scores.parquet", index=False)
    calibration_scores = scores.iloc[baseline_end:calibration_end]
    write_json(run_dir / "calibration.json", {"baseline_rows": baseline_end, "calibration_rows": calibration_end - baseline_end,
                                                "alpha": config.alpha,
                                                "score_quantiles": calibration_scores.quantile([.5, .9, .95, .99]).to_dict()})
    evaluation = output.iloc[calibration_end:].reset_index(names="source_row")
    predicted = group_positive_runs(evaluation["point_alert"], config.max_gap_steps, config.min_event_points)
    incidents_to_frame(predicted, evaluation["timestamp"]).to_csv(run_dir / "incidents.csv", index=False)
    metrics: dict[str, Any] = {"rows": len(frame), "baseline_rows": baseline_end,
                               "calibration_rows": calibration_end - baseline_end, "evaluation_rows": len(evaluation),
                               "evaluation_point_alerts": int(evaluation["point_alert"].sum()),
                               "evaluation_alert_rate": float(evaluation["point_alert"].mean())}
    if "label" in output:
        actual = group_positive_runs(evaluation["label"].astype(bool), config.max_gap_steps, 1)
        metrics |= evaluate_events(predicted, actual)
    write_json(run_dir / "metrics.json", metrics)
    write_json(manifest_path, {"status": "complete", "input_sha256": fingerprint, "config": config_dict})
    return metrics
