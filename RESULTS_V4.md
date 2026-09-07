# Corrected saved-score evaluation

Evaluated the four completed Kaggle v6 score artifacts without refitting models.
MetroPT truth comprises four published failure windows; SWaT comprises 35
labelled incidents. Old MetroPT fragmented-event metrics are superseded.

| Isolation Forest policy | MetroPT false events / observed day | MetroPT new-alarm incidents | SWaT false events / observed day | SWaT new-alarm incidents |
|---|---:|---:|---:|---:|
| Unfiltered | 165.503 | 4/4 | 666.346 | 20/35 |
| 60-sample persistence fallback | 0.139 | 3/4 | 52.280 | 15/35 |

Both Isolation Forest runs **failed the combined normal-validation budget**.
The 60-sample policy is the predeclared fallback, not a successfully calibrated
policy. On evaluation, normal alarm occupancy is 1.50% for MetroPT and 16.93%
for SWaT. Median delays among newly detected incidents are 8,843 seconds and
113 seconds respectively; missed incidents are excluded from these medians.
MetroPT overlap recall is 4/4, but only 3/4 receive a new alarm during the window.

The robust baseline selects 3 samples on MetroPT and 60 on SWaT. Although both
meet the validation budget, evaluation gives 7.081 false events/day and 1/4
new-alarm incidents on MetroPT, and zero alarms (0/35 detections) on SWaT.
This illustrates why false-alarm suppression alone is not success.

These are exploratory results, not a cross-dataset success claim or evidence of
early failure prediction. Thresholds use the first half of normal calibration;
persistence uses only its second half. Evaluation labels are used only to score
the frozen policies. See README for exposure, timestamp-gap, and delay semantics.

## Temporal residual benchmark (Kaggle version 7)

The temporal Ridge detector was then trained from normal baseline data to predict
each sensor vector from the preceding vector. Its fixed 30-sample persistence
policy met the normal-validation budget on SWaT. On the held-out SWaT segment it
produced 0.130 false alert events per observed day and a 0.014% normal-point
false-positive rate, but only 7/35 new-alarm incident detections (20.0% recall).
Its point AP (0.173) and ROC-AUC (0.736) were below Isolation Forest. This is a
calibration/precision result, not a replacement for the higher-recall model.

On MetroPT, its selected 3-sample policy met validation but produced 21.669
false events per observed day and detected 3/4 new-alarm windows. The model is
therefore not a cross-dataset solution. All six detector-dataset combinations
were executed in public Kaggle kernel version 7; the comparison artifact is
preserved with that run.

## Causal TCN benchmark (Kaggle version 9)

The normal-only causal TCN experiment does **not** pass the operational
criterion. Its normal-calibrated 30-sample policy appears to overlap all 35/35
SWaT incidents and reports only 0.261 false *events* per observed day, but
that event count masks a pathological long-running alarm: 65.00% of normal
evaluation timestamps are alerted and no incident receives a new alarm after
its labelled-window start (0/35 onset recall). The model is therefore already
alarming through normal portions of the evaluation window; it is not an
early-warning detector.

This result is retained to demonstrate why event counts must be read alongside
normal-point alert occupancy and onset recall. It is a negative benchmark
result, not evidence for a CV performance claim. The full v9 comparison CSV,
per-run score artifacts, calibration candidates, and provenance are preserved
in the public Kaggle output.

## Supervised historical-incident benchmark (Kaggle version 10)

The separate supervised experiment also does **not** earn a performance claim.
A HistGradientBoosting classifier was trained only on earlier labelled SWaT
history; the final 11 complete incidents were kept untouched for chronological
test. The threshold and a 30-sample persistence policy were selected using
normal rows before that test boundary. It produced zero false alert events per
observed day and 0.071% normal-time alert occupancy, but generated a new alarm
for only 2/11 later incidents (18.2% onset recall; 8.96% attack-point recall).

The low false-alert rate is therefore not sufficient: known early incident
signatures did not transfer reliably to the later incidents. This is retained
as a reproducible negative result, not a CV performance claim and not evidence
of novel-anomaly detection.
