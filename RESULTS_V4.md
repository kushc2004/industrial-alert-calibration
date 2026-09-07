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
