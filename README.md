# Industrial Alert Calibration

A standalone, public implementation for turning multivariate industrial telemetry into calibrated, event-level anomaly alerts. It is intentionally independent of any proprietary code or data.

The pipeline uses a chronological baseline to fit robust per-sensor statistics, converts deviations to a multivariate anomaly score, applies split-conformal calibration, groups point alerts into incidents, and evaluates both point and event-level performance.

The optional `temporal_tcn` detector instead uses only past sensor windows to
forecast the next multivariate reading. It is fitted solely on normal baseline
telemetry and scores the resulting prediction residual; labels remain post-hoc
evaluation data.

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

## Five-configuration operational replay

When several anomaly configurations have already produced timestamped scores,
evaluate their deployment policies fairly with the replay command. It aligns
the label file and every score stream to their common timestamps, tunes each
configuration's score threshold and causal confirmation duration on earlier
labelled incidents only, then reserves the final 30% of complete incidents for
one untouched test. Each configuration must meet the same predeclared
validation gate: at least 50% new-onset incident recall and no more than one
false alert event per observed day.

```bash
industrial-alert-replay \
  --labels merged.csv --timestamp-column Timestamp --label-column label \
  --score model_a_forecast=outputs/model_a_forecast.csv \
  --score model_a_reconstruction=outputs/model_a_reconstruction.csv \
  --score model_b_forecast=outputs/model_b_forecast.csv \
  --score model_b_reconstruction=outputs/model_b_reconstruction.csv \
  --score model_b_fewshot=outputs/model_b_fewshot.csv \
  --output artifacts/five-configuration-replay
```

Every score CSV needs `Timestamp` and `Overall_AnomalyScore` columns. The
resulting `comparison.csv` reports held-out incident recall, detection delay,
false alerts per observed day, and whether the policy passed its *earlier*
validation gate. A configuration that fails that gate is shown for diagnosis
but cannot be declared the winner. Raw score files and result artifacts stay
outside this repository; attach them as a Kaggle input or publish them as a
separately versioned artifact dataset only when their licence permits it.

Generate three transparent classical score streams before replaying them. The
PCA reconstruction and Isolation Forest models fit only normal-labelled rows
before the frozen test boundary; the rolling z-score uses only prior telemetry
at every timestamp.

```bash
industrial-classical-scores --input merged.csv \
  --fit-before '2015-12-31T22:06:00Z' --output artifacts/classical-scores
```

Then append `--score pca_reconstruction=...`, `--score isolation_forest=...`,
and `--score robust_rolling_zscore=...` to the replay command. The precise
boundary must be recorded from the initial five-configuration replay rather
than selected after inspecting classical-baseline results.

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

## Separate supervised historical-incident benchmark

`kaggle/supervised_swat_benchmark.py` is intentionally **not** part of the
label-free comparison. It trains a HistGradientBoosting classifier using only
the early labelled SWaT history, reserves the last 30% of complete labelled
incidents for a single chronological test, and uses the intervening labelled
history to choose an event-aware threshold/persistence policy. Selection has a
predeclared gate: at least 50% new-onset incident recall and at most one false
alert event per observed day. The final test labels are never used for fitting,
policy selection, or threshold selection. The saved test scores, ground truth,
model, candidates, and metrics make the split auditable.

This experiment answers a narrower operational question: whether signatures of
known historical incidents transfer to later incidents and whether an
event-aware confirmation rule reduces alerts at the *same score cutoff*. It
cannot support a claim of novel-attack detection, root-cause diagnosis, or
general anomaly detection. Report new-onset incident recall together with
normal-time alert occupancy, false alert events per observed day, and the
unfiltered same-threshold baseline. If the selection gate fails, it is a
negative result, not a CV performance claim.

## Long-horizon TSFM replay on SWaT

`kaggle/long_swat_tsfm_replay.py` produces a longer, auditable replay from the
public normal-plus-attack SWaT release. It aggregates readings to five-minute
intervals, preserves an interval's attack label whenever any constituent row is
attacked, and fits score scaling only on the initial known-healthy 20% of the
chronology. It then writes five aligned score streams and evaluates all of them
on the same untouched final 30% of complete attack episodes.

The model runtime and checkpoints are intentionally separate from this public
repository. Mount them as a private Kaggle dataset with this layout:

```text
private-tsfm-runtime/
  kaggle_private/run_tsfm_scores.py
  src/...
  checkpoints/MOMENT-1-small/
  checkpoints/GTT-1.7/GTT-1.7.pt
```

With the public SWaT dataset and that private dataset attached to a GPU Kaggle
notebook, run:

```bash
pip install -e .
python kaggle/long_swat_tsfm_replay.py \
  --swat-input /kaggle/input/swat-dataset-secure-water-treatment-system/merged.csv \
  --private-runner /kaggle/input/private-tsfm-runtime/kaggle_private/run_tsfm_scores.py \
  --resume
```

The run folder is resumable: completed score files are reused. Preserve its
`preparation.json`, score manifest, and `replay/metrics.json`; only measured
held-out results from these artifacts should be used in a CV claim.

For failure resilience across Kaggle sessions, run configurations in small
groups (for example, the two MOMENT configurations), save the notebook output
as a private artifact dataset, then attach that artifact dataset to the next
session. Pass its `long-swat-replay` directory through `--artifact-cache`; the
runner restores valid completed score files and only computes the requested
missing configurations. A complete five-score cache triggers the replay
automatically.
