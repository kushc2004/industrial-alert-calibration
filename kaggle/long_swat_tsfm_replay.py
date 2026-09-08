"""Resumable long-horizon SWaT replay orchestration for Kaggle.

The public part of this workflow prepares public telemetry and evaluates score
files.  Model runtime code and checkpoints are mounted separately as a private
Kaggle input, so neither is copied into this repository or output artifacts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from industrial_alert_calibration.operational_replay import main as replay_main
from industrial_alert_calibration.swat_preparation import write_prepared_swat


CONFIGURATIONS = (
    "moment_forecast",
    "moment_reconstruction",
    "gtt_forecast",
    "gtt_reconstruction",
    "gtt_fewshot",
)


def _valid_score(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 100


def _restore_artifacts(cache_dir: Path, run_dir: Path) -> None:
    """Seed a new Kaggle session from a read-only saved artifact dataset."""
    for relative in ("prepared_swat.parquet", "labels.csv", "preparation.json", "scores/score_manifest.json"):
        source = cache_dir / relative
        destination = run_dir / relative
        if source.exists() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    cached_scores = cache_dir / "scores"
    if cached_scores.exists():
        destination_scores = run_dir / "scores"
        destination_scores.mkdir(parents=True, exist_ok=True)
        for source in cached_scores.glob("*.csv"):
            destination = destination_scores / source.name
            if not _valid_score(destination) and _valid_score(source):
                shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chronological SWaT TSFM operational replay.")
    parser.add_argument("--swat-input", required=True, help="Public SWaT normal+attack or merged CSV/XLSX")
    parser.add_argument("--private-runner", required=True, help="Mounted private model-runtime score runner")
    parser.add_argument("--run-dir", default="/kaggle/working/long-swat-replay")
    # Five-minute aggregation keeps the full normal+attack release tractable
    # for per-window forecasting while materially extending the observed span.
    parser.add_argument("--cadence", default="5min")
    parser.add_argument("--holdout-fraction", type=float, default=.30)
    parser.add_argument("--max-false-alerts-per-day", type=float, default=1.0)
    parser.add_argument("--min-onset-recall", type=float, default=.50)
    parser.add_argument("--baseline-fraction", type=float, default=.20,
                        help="Initial known-healthy fraction used only to fit score scaling.")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS),
                        help="Configurations to score in this invocation; enables checkpointed Kaggle runs.")
    parser.add_argument("--artifact-cache", help="Read-only prior long-swat-replay artifact directory to restore.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    prepared_path = run_dir / "prepared_swat.parquet"
    labels_path = run_dir / "labels.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_cache:
        _restore_artifacts(Path(args.artifact_cache), run_dir)

    if not (args.resume and prepared_path.exists() and labels_path.exists()):
        metadata = write_prepared_swat(Path(args.swat_input), prepared_path, args.cadence)
        import pandas as pd
        pd.read_parquet(prepared_path)[["Timestamp", "label"]].to_csv(labels_path, index=False)
        (run_dir / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    missing = [name for name in args.configs if not _valid_score(scores_dir / f"{name}.csv")]
    if missing:
        command = [
            sys.executable, args.private_runner, "--prepared", str(prepared_path),
            "--output-dir", str(scores_dir), "--baseline-fraction", str(args.baseline_fraction),
            "--configs", *missing,
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True)

    incomplete = [name for name in CONFIGURATIONS if not _valid_score(scores_dir / f"{name}.csv")]
    if incomplete:
        (run_dir / "run_status.json").write_text(json.dumps({
            "status": "checkpointed", "completed_configs": [name for name in CONFIGURATIONS if name not in incomplete],
            "remaining_configs": incomplete,
        }, indent=2) + "\n", encoding="utf-8")
        print("Checkpoint saved. Remaining configurations: " + ", ".join(incomplete))
        return

    score_args = [item for name in CONFIGURATIONS for item in ("--score", f"{name}={scores_dir / (name + '.csv')}")]
    sys.argv = [
        "industrial-alert-replay", "--labels", str(labels_path), *score_args,
        "--output", str(run_dir / "replay"), "--timestamp-column", "Timestamp",
        "--label-column", "label", "--score-column", "Overall_AnomalyScore",
        "--holdout-fraction", str(args.holdout_fraction),
        "--max-false-alerts-per-day", str(args.max_false_alerts_per_day),
        "--min-onset-recall", str(args.min_onset_recall),
    ]
    replay_main()


if __name__ == "__main__":
    main()
