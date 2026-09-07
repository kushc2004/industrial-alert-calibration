from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import file_sha256, read_json, write_json
from .calibration import conformal_p_values
from .datasets import DatasetPreset, load_dataset
from .events import evaluate_events, group_positive_runs, incidents_to_frame
from .scoring import robust_multivariate_score, isolation_forest_score


@dataclass(frozen=True)
class PipelineConfig:
    input_path: str
    reference_path: str | None = None
    dataset: DatasetPreset = "generic"
    timestamp_column: str = "timestamp"
    label_column: str | None = "label"
    score_column: str | None = None
    detector: str = "robust"
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
    reference_source = Path(config.reference_path) if config.reference_path else None
    if reference_source and not reference_source.exists():
        raise FileNotFoundError(reference_source)
    fingerprint = file_sha256(source)
    reference_fingerprint = file_sha256(reference_source) if reference_source else None
    config_dict = asdict(config) | {
        "implementation_sha256": hashlib.sha256(b"".join(
            p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest(),
        "input_path": source.name,
        "reference_path": reference_source.name if reference_source else None,
        "artifacts_dir": Path(config.artifacts_dir).name,
        "resume": False,
    }
    if config.resume and manifest_path.exists():
        previous = read_json(manifest_path)
        if (previous.get("input_sha256") == fingerprint and previous.get("reference_sha256") == reference_fingerprint
                and previous.get("config") == config_dict and previous.get("status") == "complete"):
            return read_json(run_dir / "metrics.json")
        if (previous.get("input_sha256") != fingerprint or previous.get("reference_sha256") != reference_fingerprint
                or previous.get("config") != config_dict):
            raise ValueError("cannot resume: input or configuration changed; use a new run-id")

    write_json(manifest_path, {"status": "running", "input_sha256": fingerprint,
                               "reference_sha256": reference_fingerprint, "config": config_dict})
    write_json(run_dir / "environment.json", {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "joblib", "pyarrow"]
    })
    evaluation_frame = load_dataset(source, config.dataset).reset_index(drop=True)
    reference_frame = (load_dataset(reference_source, config.dataset).reset_index(drop=True)
                       if reference_source else evaluation_frame)
    if config.timestamp_column not in evaluation_frame or config.timestamp_column not in reference_frame:
        raise ValueError(f"missing timestamp column: {config.timestamp_column}")
    evaluation_frame[config.timestamp_column] = pd.to_datetime(evaluation_frame[config.timestamp_column], utc=True)
    reference_frame[config.timestamp_column] = pd.to_datetime(reference_frame[config.timestamp_column], utc=True)
    if not evaluation_frame[config.timestamp_column].is_monotonic_increasing or not reference_frame[config.timestamp_column].is_monotonic_increasing:
        raise ValueError("timestamps must be in chronological ascending order")
    baseline_end = int(len(reference_frame) * config.baseline_fraction)
    calibration_end = int(len(reference_frame) * config.calibration_fraction)
    if not 0 < config.baseline_fraction < config.calibration_fraction < 1:
        raise ValueError("require 0 < baseline_fraction < calibration_fraction < 1")
    if baseline_end < 10 or calibration_end - baseline_end < 10 or (not reference_source and calibration_end >= len(reference_frame)):
        raise ValueError("fractions must leave 10 baseline rows, 10 calibration rows, and one evaluation row")
    # A combined, labelled data set is allowed only when its chronological
    # reference segment is demonstrably normal.  Labels are never used for
    # scoring or threshold selection; this is a split-validity assertion.
    if config.label_column and config.label_column in reference_frame:
        reference_labels = reference_frame[config.label_column].iloc[:calibration_end]
        if reference_labels.astype(int).ne(0).any():
            raise ValueError(
                "baseline and calibration rows must be normal-labelled; "
                "use a chronological normal segment or a separate reference file"
            )
    excluded = {config.timestamp_column, config.label_column, config.score_column}
    features = [column for column in reference_frame.select_dtypes(include="number").columns
                if column not in excluded and column in evaluation_frame]
    if not features:
        raise ValueError("no shared numeric features available for scoring")
    scoring_frame = pd.concat([reference_frame, evaluation_frame], ignore_index=True) if reference_source else reference_frame
    if config.score_column:
        scores = scoring_frame[config.score_column].astype(float)
    elif config.detector == "robust":
        scores = robust_multivariate_score(scoring_frame, features, baseline_end)
    elif config.detector == "isolation_forest":
        scores = isolation_forest_score(scoring_frame, features, baseline_end, run_dir / "model.joblib")
    else:
        raise ValueError(f"unknown detector: {config.detector}")
    p_values = conformal_p_values(scores, scores.iloc[baseline_end:calibration_end])
    output = pd.DataFrame({"timestamp": scoring_frame[config.timestamp_column], "score": scores, "p_value": p_values})
    output["split"] = "evaluation"
    output.loc[: baseline_end - 1, "split"] = "baseline"
    output.loc[baseline_end:calibration_end - 1, "split"] = "calibration"
    if reference_source and calibration_end < len(reference_frame):
        output.loc[calibration_end:len(reference_frame) - 1, "split"] = "reference_unused"
    output["point_alert"] = output["p_value"] <= config.alpha
    if config.label_column and config.label_column in scoring_frame:
        output["label"] = scoring_frame[config.label_column].astype(int)
    output.to_parquet(run_dir / "scores.parquet", index=False)
    calibration_scores = scores.iloc[baseline_end:calibration_end]
    write_json(run_dir / "calibration.json", {"baseline_rows": baseline_end, "calibration_rows": calibration_end - baseline_end,
                                                "alpha": config.alpha,
                                                "score_quantiles": calibration_scores.quantile([.5, .9, .95, .99]).to_dict()})
    evaluation_start = len(reference_frame) if reference_source else calibration_end
    evaluation = output.iloc[evaluation_start:].reset_index(names="source_row")
    predicted = group_positive_runs(
        evaluation["point_alert"], config.max_gap_steps, config.min_event_points, evaluation["timestamp"]
    )
    incidents_to_frame(predicted, evaluation["timestamp"]).to_csv(run_dir / "incidents.csv", index=False)
    metrics: dict[str, Any] = {"rows": len(scoring_frame), "reference_rows": len(reference_frame),
                               "baseline_rows": baseline_end, "calibration_rows": calibration_end - baseline_end,
                               "reference_anomaly_rows": 0,
                               "evaluation_rows": len(evaluation),
                               "evaluation_point_alerts": int(evaluation["point_alert"].sum()),
                               "evaluation_alert_rate": float(evaluation["point_alert"].mean())}
    if "label" in output:
        actual = group_positive_runs(
            evaluation["label"].astype(bool), 0, 1, evaluation["timestamp"]
        )
        metrics |= evaluate_events(predicted, actual, evaluation["timestamp"])
        from sklearn.metrics import average_precision_score, roc_auc_score
        labels = evaluation["label"]
        if labels.nunique() == 2:
            metrics["point_average_precision"] = float(average_precision_score(labels, evaluation["score"]))
            metrics["point_roc_auc"] = float(roc_auc_score(labels, evaluation["score"]))
        for value, name in [(0, "normal_false_positive_rate"), (1, "attack_point_recall")]:
            selected = evaluation.loc[labels.eq(value), "point_alert"]
            metrics[name] = float(selected.mean()) if len(selected) else None
        elapsed_seconds = (evaluation["timestamp"].iloc[-1] - evaluation["timestamp"].iloc[0]).total_seconds()
        if elapsed_seconds > 0:
            metrics["evaluation_duration_days"] = elapsed_seconds / 86_400
            metrics["false_alerts_per_day"] = metrics["false_alert_events"] / metrics["evaluation_duration_days"]
    write_json(run_dir / "metrics.json", metrics)
    write_json(manifest_path, {"status": "complete", "input_sha256": fingerprint,
                               "reference_sha256": reference_fingerprint, "config": config_dict})
    return metrics
