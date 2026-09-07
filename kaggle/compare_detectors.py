"""Fixed, label-free model comparison; each completed run can be resumed."""
import json
from pathlib import Path

import pandas as pd
from industrial_alert_calibration.pipeline import PipelineConfig, run_pipeline
from industrial_alert_calibration.reevaluate import reevaluate

ROOT = Path('/kaggle/working/artifacts')
DATASETS = {
    'metropt_raw': '/kaggle/input/metropt-3-dataset/MetroPT3(AirCompressor).csv',
    'swat': '/kaggle/input/swat-dataset-secure-water-treatment-system/merged.csv',
}
rows = []
for dataset, path in DATASETS.items():
    for detector in ['robust', 'isolation_forest']:
        name = f'{dataset}-{detector}-v4'
        print(f'Starting {name}', flush=True)
        metrics = run_pipeline(PipelineConfig(
            input_path=path, dataset=dataset, detector=detector,
            run_id=name, artifacts_dir=str(ROOT), resume=True,
        ))
        rows.extend(reevaluate(ROOT / name, ROOT / 'corrected' / name))
        pd.DataFrame(rows).to_csv(ROOT / 'comparison.csv', index=False)
        print(json.dumps(rows[-1], indent=2), flush=True)
