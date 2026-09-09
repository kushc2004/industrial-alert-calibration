"""Compare diagonal and full-covariance aggregation of saved residuals.

The two streams in this module are deliberately derived from the *same* model
residual matrix and the same initial healthy calibration prefix.  This makes
the experiment an aggregation ablation, rather than a comparison between
different models, windows, or label splits.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import write_json
from .operational_replay import align_score_streams, run_operational_replay


def aggregate_residuals(residuals: np.ndarray, baseline_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal-covariance and Mahalanobis scores from one residual matrix.

    ``independent_standardized_rms`` is the root-mean-square standardized
    residual, equivalent to a diagonal-covariance distance up to a constant
    factor.  Mahalanobis additionally accounts for correlations observed only
    in the initial healthy prefix.
    """
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("residuals must be a non-empty two-dimensional matrix")
    if not 1 < baseline_rows <= len(values):
        raise ValueError("baseline_rows must be between 2 and the residual row count")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = values[:baseline_rows]
    center = baseline.mean(axis=0)
    scale = baseline.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (values - center) / scale
    independent_standardized_rms = np.sqrt(np.mean(standardized**2, axis=1))

    covariance = np.cov(baseline, rowvar=False) + np.eye(baseline.shape[1]) * 1e-6
    precision = np.linalg.pinv(covariance)
    delta = values - center
    mahalanobis = np.sqrt(np.einsum("ij,jk,ik->i", delta, precision, delta))
    return independent_standardized_rms, mahalanobis


def write_aggregation_ablation(
    residual_path: Path,
    labels_path: Path,
    output_dir: Path,
    baseline_rows: int,
    warmup_rows: int,
    holdout_fraction: float = .30,
    min_onset_recall: float = .50,
    max_false_alerts_per_day: float = 1.0,
) -> dict:
    """Write two score streams and their one-time chronological replay results."""
    residual_frame = pd.read_parquet(residual_path).sort_values("Timestamp").reset_index(drop=True)
    sensors = [column for column in residual_frame if column != "Timestamp"]
    if not sensors:
        raise ValueError("residual artifact contains no sensor residual columns")
    if residual_frame.Timestamp.duplicated().any():
        raise ValueError("residual artifact has duplicate timestamps")
    if len(residual_frame) <= warmup_rows:
        raise ValueError("residual artifact is shorter than the shared warm-up")
    independent, mahalanobis = aggregate_residuals(residual_frame[sensors].to_numpy(), baseline_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = residual_frame.Timestamp.iloc[warmup_rows:].to_numpy()
    score_paths = {
        "independent_standardized_rms": output_dir / "independent_standardized_rms.csv",
        "mahalanobis": output_dir / "mahalanobis.csv",
    }
    for name, score in (("independent_standardized_rms", independent), ("mahalanobis", mahalanobis)):
        destination = score_paths[name]
        temporary = destination.with_suffix(".csv.tmp")
        pd.DataFrame({"Timestamp": timestamp, "Overall_AnomalyScore": score[warmup_rows:]}).to_csv(
            temporary, index=False
        )
        temporary.replace(destination)

    replay_frame = align_score_streams(labels_path, score_paths)
    comparison, summary = run_operational_replay(
        replay_frame,
        holdout_fraction=holdout_fraction,
        min_onset_recall=min_onset_recall,
        max_false_alerts_per_day=max_false_alerts_per_day,
    )
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    manifest = {
        "residual_path": str(residual_path),
        "residual_rows": len(residual_frame),
        "sensor_count": len(sensors),
        "baseline_rows": baseline_rows,
        "warmup_rows": warmup_rows,
        "independent_aggregation": "RMS of residuals standardized by initial healthy baseline mean and standard deviation (diagonal covariance)",
        "mahalanobis_aggregation": "Full-covariance Mahalanobis distance fit only on initial healthy baseline residuals",
    }
    write_json(output_dir / "ablation_manifest.json", manifest)
    write_json(output_dir / "metrics.json", summary | {"configurations": comparison.to_dict(orient="records"), **manifest})
    return {"comparison": comparison, "summary": summary, "manifest": manifest}
