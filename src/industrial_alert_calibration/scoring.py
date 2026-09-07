from __future__ import annotations

import numpy as np
import pandas as pd


def robust_multivariate_score(
    frame: pd.DataFrame, feature_columns: list[str], baseline_end: int
) -> pd.Series:
    """Return a label-free robust distance, fitted only on an initial baseline."""
    if baseline_end < 10:
        raise ValueError("baseline must contain at least 10 rows")
    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.iloc[:baseline_end].median().fillna(0))
    baseline = values.iloc[:baseline_end]
    median = baseline.median()
    mad = (baseline - median).abs().median().replace(0, np.nan)
    # 1.4826 makes MAD consistent with standard deviation under a Normal model.
    scale = (1.4826 * mad).fillna(baseline.std(ddof=0)).replace(0, 1.0).fillna(1.0)
    z = (values - median) / scale
    return np.sqrt((z**2).sum(axis=1)).rename("score")


def isolation_forest_score(frame, feature_columns, baseline_end, model_path):
    """Fit only normal baseline rows; persist preprocessing and fitted trees."""
    import joblib
    from sklearn.ensemble import IsolationForest

    values = frame[feature_columns].astype(float).replace([np.inf, -np.inf], np.nan)
    medians = values.iloc[:baseline_end].median().fillna(0)
    values = values.fillna(medians).to_numpy(dtype=np.float32)
    model = IsolationForest(n_estimators=200, max_samples=min(1024, baseline_end),
                            random_state=42, n_jobs=-1)
    model.fit(values[:baseline_end])
    joblib.dump({"model": model, "features": feature_columns, "medians": medians}, model_path)
    scores = np.concatenate([-model.score_samples(values[i:i + 50000])
                             for i in range(0, len(values), 50000)])
    return pd.Series(scores, index=frame.index, name="score")
