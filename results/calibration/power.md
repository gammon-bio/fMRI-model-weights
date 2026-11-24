# Power calibration (validation data only; calibration seeds 100-103, disjoint from sweep seeds 0-4)

Headroom = reweighted minus vanilla fine-tune balanced accuracy on `val_balanced` (n = 210, 50/50 labels,
training sites). Target >= 0.25. No test subjects were generated during calibration.

| setting | seed | vanilla | reweight | headroom |
|---|---|---|---|---|
| conf1_effect -0.5 {'data.confound.strength': 1.0} | 100 | 0.500 | 0.548 | 0.048 |
| conf1_effect -0.5 {'data.confound.strength': 1.0} | 101 | 0.543 | 0.643 | 0.100 |
| base_effect -0.5  | 100 | 0.543 | 0.800 | 0.257 |
| base_effect -0.5  | 101 | 0.852 | 0.933 | 0.081 |
| base_effect -0.65  | 100 | 0.890 | 0.933 | 0.043 |
| base_effect -0.65  | 101 | 0.981 | 0.986 | 0.005 |
| base_effect -0.8  | 100 | 1.000 | 1.000 | 0.000 |
| base_effect -0.8  | 101 | 1.000 | 1.000 | 0.000 |
| n100_effect -0.4 {'data.subjects_per_site': 100} | 100 | 0.581 | 0.890 | 0.310 |
| n100_effect -0.4 {'data.subjects_per_site': 100} | 101 | 0.548 | 0.833 | 0.286 |
| n100_effect -0.4 {'data.subjects_per_site': 100} | 102 | 0.657 | 0.829 | 0.171 |
| n100_effect -0.4 {'data.subjects_per_site': 100} | 103 | 0.510 | 0.862 | 0.352 |
| prev95_effect -0.5 {'data.diagnosis.train_prevalence': [0.95, 0.05, 0.95]} | 100 | 0.500 | 0.505 | 0.005 |
| prev95_effect -0.5 {'data.diagnosis.train_prevalence': [0.95, 0.05, 0.95]} | 101 | 0.500 | 0.724 | 0.224 |

Chosen: subjects_per_site 100, effect -0.4. Headroom over 4 calibration seeds: mean 0.280, SD 0.077, min 0.171; 3/4 seeds >= 0.25.
Effect size alone fails: a stronger effect lets the vanilla fine-tune learn the real signal, removing the shortcut.
Hard condition: dropped, see hard.json (closest FC recovery 0.794, outside 0.75 +/- 0.03).
