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
