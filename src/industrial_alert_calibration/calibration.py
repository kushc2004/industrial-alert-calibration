from __future__ import annotations

import numpy as np
import pandas as pd


def conformal_p_values(scores: pd.Series, calibration_scores: pd.Series) -> pd.Series:
    """Finite-sample upper-tail conformal p-values (small means anomalous)."""
    reference = np.sort(calibration_scores.to_numpy(dtype=float))
    if len(reference) == 0:
        raise ValueError("calibration scores cannot be empty")
    rank = np.searchsorted(reference, scores.to_numpy(dtype=float), side="left")
    return pd.Series((len(reference) - rank + 1) / (len(reference) + 1), index=scores.index, name="p_value")
