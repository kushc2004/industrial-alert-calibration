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

from industrial_alert_calibration.aggregation_ablation import write_aggregation_ablation
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


def _valid_residuals(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000


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
    cached_residuals = cache_dir / "residuals"
    if cached_residuals.exists():
        destination_residuals = run_dir / "residuals"
        destination_residuals.mkdir(parents=True, exist_ok=True)
        for source in cached_residuals.glob("*.parquet"):
            destination = destination_residuals / source.name
            if not _valid_residuals(destination) and _valid_residuals(source):
                shutil.copy2(source, destination)


def _preflight_baseline(prepared_path: Path, baseline_fraction: float, minimum_rows: int) -> dict[str, int | float | bool]:
    """Validate the healthy calibration prefix used for score aggregation.

    The model-score warm-up and the statistical calibration set have distinct
    purposes.  MOMENT can retain a 512-step scoring warm-up while the
    residual-aggregation fit uses an earlier healthy prefix.  The latter only
    needs enough observations to stably estimate the 51-sensor covariance.
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
    if baseline_rows < minimum_rows:
        raise ValueError(
            "Baseline preflight failed: "
            f"{baseline_rows} healthy calibration rows are available, but this aggregation "
            f"fit requires at least {minimum_rows}. Use a longer known-healthy telemetry "
            "period or a finer cadence only when the raw release supports it."
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
    parser.add_argument("--minimum-baseline-rows", type=int, default=128,
                        help="Minimum healthy rows for score calibration; 128 exceeds twice the 51 sensors.")
    parser.add_argument("--warmup-rows", type=int, default=512,
                        help="Initial rows excluded from every score stream and aggregation comparison.")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS),
                        help="Configurations to score in this invocation; enables checkpointed Kaggle runs.")
    parser.add_argument("--aggregation-ablation-config", choices=CONFIGURATIONS,
                        help="Score one configuration, export its residual matrix, and compare diagonal against full-covariance aggregation.")
    parser.add_argument("--artifact-cache", help="Read-only prior long-swat-replay artifact directory to restore.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if bool(args.normal_input) != bool(args.attack_input):
        parser.error("--normal-input and --attack-input must be supplied together")

    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    residuals_dir = run_dir / "residuals"
    prepared_path = run_dir / "prepared_swat.parquet"
    labels_path = run_dir / "labels.csv"
    preparation_path = run_dir / "preparation.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_cache:
        _restore_artifacts(Path(args.artifact_cache), run_dir)

    expected_session_mode = "normal_then_attack_replay_clock" if args.normal_input else "single_file"
    existing_metadata: dict[str, object] = {}
    if preparation_path.exists():
        try:
            existing_metadata = json.loads(preparation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    preparation_matches = (
        existing_metadata.get("session_mode") == expected_session_mode
        and existing_metadata.get("cadence") == args.cadence
    )
    needs_preparation = not (
        args.resume and prepared_path.exists() and labels_path.exists() and preparation_matches
    )
    if needs_preparation:
        # Never reuse score streams produced against a different source order or
        # cadence.  Those files look valid structurally but would invalidate a
        # chronological replay.
        if scores_dir.exists():
            for score_path in scores_dir.glob("*.csv"):
                score_path.unlink()
            manifest = scores_dir / "score_manifest.json"
            if manifest.exists():
                manifest.unlink()
        if residuals_dir.exists():
            for residual_path in residuals_dir.glob("*.parquet"):
                residual_path.unlink()
        if args.normal_input:
            metadata = write_prepared_swat_sessions(
                Path(args.normal_input), Path(args.attack_input), prepared_path, args.cadence
            )
        else:
            metadata = write_prepared_swat(Path(args.swat_input), prepared_path, args.cadence)
        pd.read_parquet(prepared_path)[["Timestamp", "label"]].to_csv(labels_path, index=False)
        preparation_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    try:
        preflight = _preflight_baseline(
            prepared_path, args.baseline_fraction,
            args.minimum_baseline_rows,
        )
    except ValueError as error:
        (run_dir / "preflight.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n", encoding="utf-8")
        raise
    (run_dir / "preflight.json").write_text(json.dumps({"status": "passed", **preflight}, indent=2) + "\n", encoding="utf-8")

    requested_configs = [args.aggregation_ablation_config] if args.aggregation_ablation_config else args.configs
    if args.aggregation_ablation_config:
        target = args.aggregation_ablation_config
        missing = [target] if not (
            _valid_score(scores_dir / f"{target}.csv")
            and _valid_residuals(residuals_dir / f"{target}.parquet")
        ) else []
    else:
        missing = [name for name in requested_configs if not _valid_score(scores_dir / f"{name}.csv")]
    if missing:
        command = [
            sys.executable, args.private_runner, "--prepared", str(prepared_path),
            "--output-dir", str(scores_dir), "--baseline-fraction", str(args.baseline_fraction),
            "--warmup-rows", str(args.warmup_rows), "--configs", *missing,
            "--residual-dir", str(residuals_dir),
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

    if args.aggregation_ablation_config:
        target = args.aggregation_ablation_config
        residual_path = residuals_dir / f"{target}.parquet"
        if not _valid_residuals(residual_path):
            raise RuntimeError(f"Private runner did not write the required residual artifact: {residual_path}")
        baseline_rows = int(len(pd.read_parquet(prepared_path, columns=["label"])) * args.baseline_fraction)
        result = write_aggregation_ablation(
            residual_path, labels_path, run_dir / "aggregation_ablation" / target,
            baseline_rows=baseline_rows, warmup_rows=args.warmup_rows,
            holdout_fraction=args.holdout_fraction, min_onset_recall=args.min_onset_recall,
            max_false_alerts_per_day=args.max_false_alerts_per_day,
        )
        print(result["comparison"].to_string(index=False))
        print(f"Wrote aggregation ablation artifacts to {run_dir / 'aggregation_ablation' / target}")
        return

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
