import numpy as np

from industrial_alert_calibration.aggregation_ablation import aggregate_residuals


def test_aggregation_uses_only_the_baseline_for_its_moments():
    residuals = np.array([
        [1.0, 1.0], [2.0, 2.0], [1.5, 1.5], [50.0, -50.0],
    ])
    independent, mahalanobis = aggregate_residuals(residuals, baseline_rows=3)
    changed_later = residuals.copy()
    changed_later[-1] = [5000.0, -5000.0]
    repeated_independent, repeated_mahalanobis = aggregate_residuals(changed_later, baseline_rows=3)

    np.testing.assert_allclose(independent[:3], repeated_independent[:3])
    np.testing.assert_allclose(mahalanobis[:3], repeated_mahalanobis[:3])
    assert repeated_independent[-1] > independent[-1]
    assert repeated_mahalanobis[-1] > mahalanobis[-1]
