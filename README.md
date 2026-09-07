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
industrial-alerts run merged.csv --dataset swat --run-id swat-v1
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

Attach [MetroPT-3](https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset) and the public [SWaT mirror](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system), clone this repository, and run:

```bash
pip install -e /kaggle/input/industrial-alert-calibration
python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input '/kaggle/input/metropt-3-dataset/MetroPT3(AirCompressor).csv' \
  --dataset metropt_raw --run-id metropt-conformal-v2

python /kaggle/input/industrial-alert-calibration/kaggle/run_kaggle.py \
  --input /kaggle/input/swat-dataset-secure-water-treatment-system/merged.csv \
  --dataset swat --run-id swat-conformal-v2
```

The script writes to `/kaggle/working/artifacts/<run-id>`. Publish that directory as the separate `industrial-alert-calibration-artifacts` Kaggle dataset after a successful run. Dataset versioning preserves partial/recovered outputs separately from source code.

## Method and validation boundary

The initial `baseline_fraction` fits robust feature centers and scales. The following chronological segment up to `calibration_fraction` forms the label-free conformal calibration distribution. With one input, the remaining chronology is evaluation-only. With `--reference`, the reference file supplies baseline and calibration only, and the input file is wholly evaluation-only; its labels never influence fitting, calibration, or threshold selection. When labels are present, the runner asserts that the baseline and calibration rows are all normal before fitting. The conformal p-value is the finite-sample upper-tail rank of a score against the frozen calibration distribution. A p-value at or below `alpha` becomes a point alert.

For the public SWaT experiment, use this mirror's `merged.csv`, not `attack.csv` alone: it contains normal and attack-labelled telemetry, so the held-out portion supports false-alert events per calendar day as well as event precision/recall and median incident detection delay. The adapter canonicalizes whitespace variants such as `A ttack`, parses day-first timestamps, orders records by time, removes duplicate timestamps, and rejects a run if an attack-labelled row enters its baseline or calibration segment. `normal.csv` plus `attack.csv` remains useful for a separate-reference experiment, but `attack.csv` alone cannot support a false-alert-rate claim.

This is a reproducible research pipeline, not a claim that it diagnoses root cause or is production-ready. Use known incident windows and domain review before operational deployment.
# Detector comparison v3

The Kaggle benchmark compares a robust-distance baseline with a trained
Isolation Forest (200 trees, seed 42, 1,024 samples/tree). Both use identical
chronological baseline/calibration/evaluation splits and alpha=0.01. No attack
labels select model parameters or thresholds. Missing values use baseline-only
medians; fitted trees and preprocessing are saved in `model.joblib`.

Event overlap recall is NOT new-alarm recall. `onset_event_recall` requires a
new alarm starting during the labelled incident; delays are reported once per
detected incident and exclude alarms already active before it. Always report
normal-point false-positive rate alongside recall: a continuously active alarm
can obtain high overlap recall without being useful. Ground-truth events are
not merged using the predicted-alarm gap tolerance.

Version 2 results used an incorrect first-overlap-only event matcher. Do not
compare those event metrics directly with version 3. Prior test results have
already been inspected; this is an exploratory benchmark, not an untouched
confirmatory evaluation. Temporal dependence and distribution shifts mean
nominal conformal alpha does not guarantee the observed false-positive rate.
Resume checks include source-code fingerprints. Each completed detector run
is saved separately; cross-session recovery requires restoring its artifact
folder before running the notebook. A failed partial detector run is recomputed.

## Corrected evaluation v4

Re-evaluate saved model scores without retraining:

```bash
python -m industrial_alert_calibration.reevaluate artifacts/previous-runs artifacts/corrected-v4
```

MetroPT ground truth now uses one incident per published failure window, rather
than fragmenting windows at sampling gaps. **The v3 MetroPT event counts and
event recalls are superseded.** SWaT retains label runs separated by normal
samples or substantial timestamp gaps.

The normal calibration segment is divided chronologically: its first half
freezes the score threshold (alpha=0.01); its second half selects the shortest
consecutive-exceedance persistence from 1, 3, 10, 30, or 60 samples meeting both
one false event per observed day and 1% normal alarm occupancy. If none qualifies,
60 samples is a flagged fallback, not a successful calibration. Evaluation labels
do not select this policy. This normal-only selection does not guarantee useful
attack recall or a held-out false-alarm budget under distribution shift.

Unfiltered and persistence policies use the same frozen threshold. Persistence
starts at confirmation, never retroactively at the first exceedance, and resets
across missing stretches. Rates use estimated observed telemetry exposure
(timestamp increments capped at the median cadence), not missing calendar time.
Delay references the first observed sample in each labelled window and is
reported only for incidents with a new alarm; it is not advance warning of failure.
Outputs include calibration candidates, the budget-met flag, ground-truth and
predicted incidents, evaluation scores, and source checksums. The Kaggle comparison
runner writes v4 runs and the corrected policy comparison automatically.

Because earlier evaluation outcomes informed this protocol correction, results
remain exploratory. Do not present them as an untouched confirmatory test.

The next benchmark adds a temporal Ridge residual detector. It predicts the
full sensor vector from the immediately preceding vector and scores robustly
standardized prediction residuals. It is fit only on normal baseline data, so it
tests temporal multivariate behavior rather than re-running the pointwise model.
