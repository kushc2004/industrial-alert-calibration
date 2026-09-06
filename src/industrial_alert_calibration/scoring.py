from __future__ import annotations

import numpy as np
import pandas as pd


def robust_multivariate_score(
    frame: pd.DataFrame, feature_columns: list[str], baseline_end: int
) -> pd.Series:
    """Return a label-free robust distance, fitted only on an initial baseline."""
    if baseline_end < 10:
        raise ValueError("baseline must contain at least 10 rows")
    values = frame[feature_columns].astype(float).interpolate(limit_direction="both")
    baseline = values.iloc[:baseline_end]
    median = baseline.median()
    mad = (baseline - median).abs().median().replace(0, np.nan)
    # 1.4826 makes MAD consistent with standard deviation under a Normal model.
    scale = (1.4826 * mad).fillna(baseline.std(ddof=0)).replace(0, 1.0).fillna(1.0)
    z = (values - median) / scale
    return np.sqrt((z**2).sum(axis=1)).rename("score")
