# Industrial Alert Calibration

A standalone, public implementation for turning multivariate industrial telemetry into calibrated, event-level anomaly alerts. It is intentionally independent of any proprietary code or data.

The pipeline uses a chronological baseline to fit robust per-sensor statistics, converts deviations to a multivariate anomaly score, applies split-conformal calibration, groups point alerts into incidents, and evaluates both point and event-level performance.

## Why this exists

Raw anomaly scores create alert floods. This project makes the operational layer explicit:

`telemetry -> robust anomaly score -> conformal p-value -> point alert -> incident -> operational metrics`

It reports alert volume, event precision/recall, detection delay, and a false-alert rate; it does not tune on the evaluation period.

## Input

Provide a CSV or Parquet file with timestamped numeric sensor columns. `label` is optional but required for evaluation. Labels use `0` for normal and `1` for anomaly. A precomputed `score` column can be supplied with `--score-column`; otherwise a robust multivariate score is built from the numeric sensor columns. The built-in adapters include `metropt_raw` (the original MetroPT-3 CSV with published failure windows) and `swat` (the normal/attack schema in either CSV mirrors or the official SWaT Excel workbooks).

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

industrial-alerts run 'MetroPT3(AirCompressor).csv' --dataset metropt_raw --run-id metropt-v1
industrial-alerts run SWaT_Dataset_Attack_v0.xlsx \
  --reference SWaT_Dataset_Normal_v1.xlsx --dataset swat --run-id swat-v1
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

Attach MetroPT-3 plus a Kaggle dataset you create from the official SWaT A1/A2 file pair, clone this repository, and run:

```bash
pip install -e /kaggle/input/industrial-alert-calibration
python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input '/kaggle/input/metropt-3-dataset/MetroPT3(AirCompressor).csv' \
  --dataset metropt_raw --run-id metropt-conformal-v2

python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input /kaggle/input/swat-a1-a2-physical/SWaT_Dataset_Attack_v0.xlsx \
  --reference /kaggle/input/swat-a1-a2-physical/SWaT_Dataset_Normal_v1.xlsx \
  --dataset swat --run-id swat-conformal-v2
```

The script writes to `/kaggle/working/artifacts/<run-id>`. Publish that directory as the separate `industrial-alert-calibration-artifacts` Kaggle dataset after a successful run. Dataset versioning preserves partial/recovered outputs separately from source code.

## Method and validation boundary

The initial `baseline_fraction` fits robust feature centers and scales. The following chronological segment up to `calibration_fraction` forms the label-free conformal calibration distribution. With one input, the remaining chronology is evaluation-only. With `--reference`, the reference file supplies baseline and calibration only, and the input file is wholly evaluation-only; its labels never influence fitting, calibration, or threshold selection. The conformal p-value is the finite-sample upper-tail rank of a score against the frozen calibration distribution. A p-value at or below `alpha` becomes a point alert.

For the defensible SWaT experiment, use the iTrust A1/A2 December 2015 release: `Normal_v1` is strictly baseline/calibration data; `Attack_v0` is wholly evaluation data and retains both normal and attack-labelled periods. The adapter accepts the official `.xlsx` files, drops their unit-identification row, canonicalizes known whitespace variants such as `A ttack`, parses day-first timestamps, orders records by time, and removes duplicate timestamps. It reports event precision/recall, false-alert events per calendar day of evaluation, and median incident detection delay. Do not use an attack-only mirror to claim false-alert rates or operational alert reduction.

This is a reproducible research pipeline, not a claim that it diagnoses root cause or is production-ready. Use known incident windows and domain review before operational deployment.
