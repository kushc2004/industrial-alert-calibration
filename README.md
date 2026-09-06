# Industrial Alert Calibration

A standalone, public implementation for turning multivariate industrial telemetry into calibrated, event-level anomaly alerts. It is intentionally independent of any proprietary code or data.

The pipeline uses a chronological baseline to fit robust per-sensor statistics, converts deviations to a multivariate anomaly score, applies split-conformal calibration, groups point alerts into incidents, and evaluates both point and event-level performance.

## Why this exists

Raw anomaly scores create alert floods. This project makes the operational layer explicit:

`telemetry -> robust anomaly score -> conformal p-value -> point alert -> incident -> operational metrics`

It reports alert volume, event precision/recall, detection delay, and a false-alert rate; it does not tune on the evaluation period.

## Input

Provide a CSV or Parquet file with timestamped numeric sensor columns. `label` is optional but required for evaluation. Labels use `0` for normal and `1` for anomaly. A precomputed `score` column can be supplied with `--score-column`; otherwise a robust multivariate score is built from the numeric sensor columns. The built-in `metropt` and `chiller` adapters standardize the included benchmark CSV schemas.

```text
timestamp,pressure,temperature,vibration,label
2026-01-01T00:00:00Z,10.2,78.1,0.03,0
```

## Local quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
industrial-alerts run data/telemetry.csv --timestamp-column timestamp --label-column label \
  --run-id swat-v1 --artifacts-dir artifacts --alpha 0.01 --max-gap-steps 3 --min-event-points 2

industrial-alerts run MetroPT3_downsampled_interpolate_new_labelled.csv --dataset metropt --run-id metropt-v1
industrial-alerts run April2021_labelled.csv --dataset chiller --run-id chiller-v1
```

The runner is resumable. Re-run the exact command with `--resume`; completed stages with a matching input fingerprint and configuration are reused. Each run writes a portable artifact directory:

```text
artifacts/swat-v1/
  manifest.json            # stage status, configuration and data checksum
  scores.parquet           # score, p-value and point alerts
  calibration.json         # frozen calibration distribution summary
  incidents.csv            # grouped alert incidents
  metrics.json             # event and point-level metrics
```

## Kaggle

Create a Kaggle Notebook with internet disabled, attach your telemetry dataset, and upload this repository as a code dataset or clone a tagged release. Then run:

```bash
pip install -e /kaggle/input/industrial-alert-calibration
python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input /kaggle/input/labelled-industrial-telemetry-benchmarks/metropt_labelled.csv \
  --dataset metropt --run-id metropt-conformal-v1

python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input /kaggle/input/labelled-industrial-telemetry-benchmarks/chiller_april_2021_labelled.csv \
  --dataset chiller --run-id chiller-conformal-v1
```

The script writes to `/kaggle/working/artifacts/<run-id>`. Publish that directory as the separate `industrial-alert-calibration-artifacts` Kaggle dataset after a successful run. Dataset versioning preserves partial/recovered outputs separately from source code.

## Method and validation boundary

The initial `baseline_fraction` fits robust feature centers and scales. The following chronological segment up to `calibration_fraction` forms the label-free conformal calibration distribution. The remainder is evaluation-only; labels, if present, are used only there. The conformal p-value is the finite-sample upper-tail rank of a score against the frozen calibration distribution. A p-value at or below `alpha` becomes a point alert.

This is a reproducible research pipeline, not a claim that it diagnoses root cause or is production-ready. Use known incident windows and domain review before operational deployment.
