"""Kaggle entrypoint: writes resumable run artifacts to /kaggle/working by default."""
from __future__ import annotations

import argparse
import json

from industrial_alert_calibration.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", help="normal-only reference telemetry for calibration")
    parser.add_argument("--dataset", choices=["generic", "metropt", "metropt_raw", "chiller", "swat"], default="generic")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-dir", default="/kaggle/working/artifacts")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--alpha", type=float, default=.01)
    parser.add_argument("--baseline-fraction", type=float, default=.20)
    parser.add_argument("--calibration-fraction", type=float, default=.30)
    parser.add_argument("--max-gap-steps", type=int, default=3)
    parser.add_argument("--min-event-points", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = PipelineConfig(input_path=args.input, reference_path=args.reference, dataset=args.dataset, run_id=args.run_id, artifacts_dir=args.artifacts_dir,
                            timestamp_column=args.timestamp_column, label_column=args.label_column,
                            alpha=args.alpha, baseline_fraction=args.baseline_fraction,
                            calibration_fraction=args.calibration_fraction, max_gap_steps=args.max_gap_steps,
                            min_event_points=args.min_event_points, resume=args.resume)
    print(json.dumps(run_pipeline(config), indent=2))


if __name__ == "__main__":
    main()
