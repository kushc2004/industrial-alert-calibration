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

import pandas as pd

from industrial_alert_calibration.operational_replay import main as replay_main
from industrial_alert_calibration.swat_preparation import write_prepared_swat, write_prepared_swat_sessions


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


def _preflight_baseline(prepared_path: Path, baseline_fraction: float, minimum_rows: int) -> dict[str, int | float | bool]:
    """Validate that the common healthy calibration prefix supports all methods.

    MOMENT's forecast scores require a 512-step context.  Calibration must use
    only scores after that context, hence the strict ``> 512`` requirement.
    Keeping this check here avoids starting any private model process when a
    public-data/cadence choice cannot support the requested protocol.
    """
    frame = pd.read_parquet(prepared_path, columns=["label"])
    if not 0.0 < baseline_fraction < 1.0:
        raise ValueError("--baseline-fraction must be strictly between 0 and 1")
    baseline_rows = int(len(frame) * baseline_fraction)
    attacked = frame["label"].to_numpy().nonzero()[0]
    healthy_prefix_rows = int(attacked[0]) if len(attacked) else len(frame)
    report: dict[str, int | float | bool] = {
        "prepared_rows": len(frame),
        "baseline_fraction": baseline_fraction,
        "baseline_rows": baseline_rows,
        "healthy_prefix_rows": healthy_prefix_rows,
        "minimum_baseline_rows": minimum_rows,
        "baseline_is_healthy": baseline_rows <= healthy_prefix_rows,
    }
    if baseline_rows <= minimum_rows:
        raise ValueError(
            "Baseline preflight failed: "
            f"{baseline_rows} calibration rows at this cadence, but MOMENT needs more than "
            f"{minimum_rows}. Use a finer --cadence only if the raw release has higher "
            "temporal resolution, or use a longer known-healthy telemetry period."
        )
    if baseline_rows > healthy_prefix_rows:
        raise ValueError(
            "Baseline preflight failed: the requested calibration prefix includes labelled attacks "
            f"(baseline rows={baseline_rows}, known healthy prefix={healthy_prefix_rows})."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a chronological SWaT TSFM operational replay.")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--swat-input", help="Single public SWaT CSV/XLSX; use only when its order is verified.")
    sources.add_argument("--normal-input", help="Public SWaT normal-session CSV/XLSX; requires --attack-input.")
    parser.add_argument("--attack-input", help="Public SWaT attack-session CSV/XLSX; requires --normal-input.")
    parser.add_argument("--private-runner", required=True, help="Mounted private model-runtime score runner")
    parser.add_argument("--private-source-root", help="Private input directory containing the model runtime source.")
    parser.add_argument("--moment-checkpoint", help="Private mounted MOMENT checkpoint directory.")
    parser.add_argument("--gtt-checkpoint", help="Private mounted GTT checkpoint file.")
    parser.add_argument("--run-dir", default="/kaggle/working/long-swat-replay")
    # Five-minute aggregation keeps the full normal+attack release tractable
    # for per-window forecasting while materially extending the observed span.
    parser.add_argument("--cadence", default="5min")
    parser.add_argument("--holdout-fraction", type=float, default=.30)
    parser.add_argument("--max-false-alerts-per-day", type=float, default=1.0)
    parser.add_argument("--min-onset-recall", type=float, default=.50)
    parser.add_argument("--baseline-fraction", type=float, default=.20,
                        help="Initial known-healthy fraction used only to fit score scaling.")
    parser.add_argument("--minimum-baseline-rows", type=int, default=512,
                        help="Shared score warm-up requirement; retain 512 for MOMENT comparison.")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS),
                        help="Configurations to score in this invocation; enables checkpointed Kaggle runs.")
    parser.add_argument("--artifact-cache", help="Read-only prior long-swat-replay artifact directory to restore.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if bool(args.normal_input) != bool(args.attack_input):
        parser.error("--normal-input and --attack-input must be supplied together")

    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    prepared_path = run_dir / "prepared_swat.parquet"
    labels_path = run_dir / "labels.csv"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_cache:
        _restore_artifacts(Path(args.artifact_cache), run_dir)

    if not (args.resume and prepared_path.exists() and labels_path.exists()):
        if args.normal_input:
            metadata = write_prepared_swat_sessions(
                Path(args.normal_input), Path(args.attack_input), prepared_path, args.cadence
            )
        else:
            metadata = write_prepared_swat(Path(args.swat_input), prepared_path, args.cadence)
        pd.read_parquet(prepared_path)[["Timestamp", "label"]].to_csv(labels_path, index=False)
        (run_dir / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        preflight = _preflight_baseline(prepared_path, args.baseline_fraction, args.minimum_baseline_rows)
    except ValueError as error:
        (run_dir / "preflight.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        raise
    (run_dir / "preflight.json").write_text(json.dumps({"status": "passed", **preflight}, indent=2) + "\n", encoding="utf-8")

    missing = [name for name in args.configs if not _valid_score(scores_dir / f"{name}.csv")]
    if missing:
        command = [
            sys.executable, args.private_runner, "--prepared", str(prepared_path),
            "--output-dir", str(scores_dir), "--baseline-fraction", str(args.baseline_fraction),
            "--configs", *missing,
        ]
        if args.private_source_root:
            command.extend(["--source-root", args.private_source_root])
        if args.moment_checkpoint:
            command.extend(["--moment-checkpoint", args.moment_checkpoint])
        if args.gtt_checkpoint:
            command.extend(["--gtt-checkpoint", args.gtt_checkpoint])
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
