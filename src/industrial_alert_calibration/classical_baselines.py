"""Leakage-safe classical score streams for operational anomaly comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def load_telemetry(path: Path, timestamp_column: str, label_column: str) -> pd.DataFrame:
    """Load timestamped numeric telemetry, retaining the label only for fitting eligibility."""
    frame = _read(path).copy()
    missing = {timestamp_column, label_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"telemetry is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={timestamp_column: "timestamp", label_column: "label"})
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    frame["label"] = pd.to_numeric(frame.label, errors="raise").astype(int).ne(0).astype(int)
    if frame.timestamp.duplicated().any():
        raise ValueError("telemetry has duplicate timestamps")
    numeric = frame.select_dtypes(include="number").columns.difference(["label"])
    if len(numeric) < 2:
        raise ValueError("need at least two numeric telemetry columns")
    frame.loc[:, numeric] = frame.loc[:, numeric].apply(pd.to_numeric, errors="raise")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _features(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.select_dtypes(include="number").columns if column != "label"]


def score_classical_baselines(frame: pd.DataFrame, fit_before: pd.Timestamp, rolling_window: int = 60) -> tuple[pd.DataFrame, dict]:
    """Fit PCA/Isolation Forest on earlier normal rows and score all rows.

    The robust rolling score is strictly causal: at timestamp t it derives its
    centre and scale exclusively from observations before t.  PCA and Isolation
    Forest see only normal-labelled rows before ``fit_before``.  No later rows
    or labels affect any detector score.
    """
    if rolling_window < 10:
        raise ValueError("rolling_window must be at least 10")
    fit_before = pd.Timestamp(fit_before)
    if fit_before.tzinfo is None:
        fit_before = fit_before.tz_localize("UTC")
    features = _features(frame)
    train = frame.loc[frame.timestamp.lt(fit_before) & frame.label.eq(0), features]
    if len(train) < 200:
        raise ValueError("need at least 200 earlier normal rows to fit classical baselines")
    medians = train.median()
    matrix = frame[features].fillna(medians)
    scaler = RobustScaler(quantile_range=(25, 75)).fit(train.fillna(medians))
    scaled_train = scaler.transform(train.fillna(medians))
    scaled = scaler.transform(matrix)

    pca = PCA(n_components=.95, svd_solver="full", random_state=42).fit(scaled_train)
    reconstructed = pca.inverse_transform(pca.transform(scaled))
    pca_score = np.mean(np.square(scaled - reconstructed), axis=1)

    isolation_forest = IsolationForest(
        n_estimators=200, max_samples=min(1024, len(scaled_train)), contamination="auto", random_state=42, n_jobs=-1
    ).fit(scaled_train)
    isolation_score = -isolation_forest.score_samples(scaled)

    # Shift before rolling so the current value never contributes to its own
    # anomaly score. A MAD floor avoids division by zero on actuator columns.
    history = matrix.shift(1)
    rolling_median = history.rolling(rolling_window, min_periods=rolling_window // 2).median()
    rolling_mad = (history - rolling_median).abs().rolling(rolling_window, min_periods=rolling_window // 2).median()
    rolling_scale = (rolling_mad * 1.4826).replace(0, np.nan).fillna(1.0)
    rolling_score = (((matrix - rolling_median) / rolling_scale).abs().clip(upper=25).pow(2).mean(axis=1)).fillna(0.0)

    scores = frame[["timestamp"]].copy()
    scores["pca_reconstruction"] = pca_score
    scores["isolation_forest"] = isolation_score
    scores["robust_rolling_zscore"] = rolling_score
    metadata = {
        "fit_before": str(fit_before),
        "fit_rows": int(len(train)),
        "feature_count": len(features),
        "pca_components": int(pca.n_components_),
        "isolation_forest_trees": 200,
        "rolling_window_points": rolling_window,
        "rolling_score_is_causal": True,
        "fit_population": "earlier normal-labelled telemetry only",
    }
    return scores, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create PCA, Isolation Forest, and causal rolling-z score streams.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--fit-before", required=True, help="UTC boundary; rows at/after it cannot affect fitted models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamp-column", default="Timestamp")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--rolling-window", type=int, default=60)
    args = parser.parse_args()
    frame = load_telemetry(args.input, args.timestamp_column, args.label_column)
    scores, metadata = score_classical_baselines(frame, pd.Timestamp(args.fit_before), args.rolling_window)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["pca_reconstruction", "isolation_forest", "robust_rolling_zscore"]:
        scores[["timestamp", name]].rename(columns={"timestamp": "Timestamp", name: "Overall_AnomalyScore"}).to_csv(
            args.output / f"{name}.csv", index=False
        )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
