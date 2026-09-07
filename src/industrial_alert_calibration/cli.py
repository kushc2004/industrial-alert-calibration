from __future__ import annotations

import argparse
import json

from .pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate multivariate industrial anomaly alerts into incidents.")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("input_path")
    parser.add_argument("--reference")
    parser.add_argument("--dataset", choices=["generic", "metropt", "metropt_raw", "chiller", "swat"], default="generic")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column")
    parser.add_argument("--detector", choices=["robust", "isolation_forest", "temporal_ridge"], default="robust")
    parser.add_argument("--baseline-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-gap-steps", type=int, default=3)
    parser.add_argument("--min-event-points", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    values = vars(args)
    values.pop("command")
    values["reference_path"] = values.pop("reference")
    config = PipelineConfig(**values)
    print(json.dumps(run_pipeline(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
