import numpy as np
import pandas as pd

from industrial_alert_calibration.aggregation_ablation import aggregate_residuals, run_session_aware_replay


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


def test_session_aware_replay_keeps_normal_false_alerts_separate_from_attack_recall():
    normal_rows = 800
    normal_time = pd.date_range("2024-01-01", periods=normal_rows, freq="10s")
    # The attack mirror contains only labelled attack rows. Timestamp gaps
    # delineate complete incidents without inventing normal observations.
    attack_time = []
    start = normal_time[-1] + pd.Timedelta(seconds=10)
    for incident in range(6):
        attack_time.extend(pd.date_range(start + pd.Timedelta(minutes=5 * incident), periods=5, freq="10s"))
    timestamp = normal_time.append(pd.DatetimeIndex(attack_time))
    labels = np.r_[np.zeros(normal_rows, dtype=int), np.ones(len(attack_time), dtype=int)]
    score = np.r_[np.linspace(0.0, 0.01, normal_rows), np.full(len(attack_time), 10.0)]
    frame = pd.DataFrame({"timestamp": timestamp, "label": labels, "diagonal": score, "mahalanobis": score})

    comparison, summary = run_session_aware_replay(
        frame, baseline_rows=300, holdout_fraction=.50, max_false_alerts_per_day=10.0
    )

    assert summary["protocol"] == "session_aware_alert_budget_matched_replay"
    assert summary["held_out_normal_rows"] > 0
    assert summary["held_out_incidents"] == 3
    assert comparison["held_out_attack_onset_event_recall"].eq(1.0).all()
    assert comparison["held_out_normal_observed_days"].gt(0).all()
