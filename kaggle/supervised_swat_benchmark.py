"""Held-out historical-incident benchmark for the public SWaT mirror."""
import json
from pathlib import Path

import joblib

from industrial_alert_calibration.artifacts import write_json
from industrial_alert_calibration.datasets import load_dataset
from industrial_alert_calibration.events import incidents_to_frame, group_positive_runs
from industrial_alert_calibration.supervised import run_supervised_incident_benchmark

ROOT = Path('/kaggle/working/artifacts/supervised-swat-v2-event-aware')
frame = load_dataset(Path('/kaggle/input/swat-dataset-secure-water-treatment-system/merged.csv'), 'swat')
metrics, scores, model = run_supervised_incident_benchmark(frame)
ROOT.mkdir(parents=True, exist_ok=True)
scores.to_parquet(ROOT / 'held_out_scores.parquet', index=False)
incidents_to_frame(group_positive_runs(scores.label.astype(bool), 0, 1, scores.timestamp), scores.timestamp).to_csv(
    ROOT / 'held_out_ground_truth.csv', index=False
)
joblib.dump(model, ROOT / 'model.joblib')
write_json(ROOT / 'metrics.json', metrics)
print(json.dumps(metrics, indent=2))
