# Explainability Report

This report explains which feature groups matter most for the strongest predictive tasks and how the sequence model distributes attention over early fixation positions.
Sequence profile: `full`.

## Group Ablation

| task              | feature_set      | ablation           |   balanced_accuracy |   score_drop |
|:------------------|:-----------------|:-------------------|--------------------:|-------------:|
| label_found_waldo | gaze_plus_target | none               |            0.68915  |    0         |
| label_found_waldo | gaze_plus_target | gaze_dynamics      |            0.61437  |    0.0747801 |
| label_found_waldo | gaze_plus_target | spatial_search     |            0.627566 |    0.0615836 |
| label_found_waldo | gaze_plus_target | entropy_structure  |            0.552786 |    0.136364  |
| label_found_waldo | gaze_plus_target | saliency_alignment |            0.76393  |   -0.0747801 |
| label_found_waldo | gaze_plus_target | target_contact     |            0.458944 |    0.230205  |
| label_long_search | gaze_only        | none               |            0.693182 |    0         |
| label_long_search | gaze_only        | gaze_dynamics      |            0.715909 |   -0.0227273 |
| label_long_search | gaze_only        | spatial_search     |            0.713636 |   -0.0204545 |
| label_long_search | gaze_only        | entropy_structure  |            0.572727 |    0.120455  |
| label_long_search | gaze_only        | saliency_alignment |            0.668182 |    0.025     |
| label_long_search | gaze_only        | target_contact     |            0.693182 |    0         |

## Linear Weight Ranking: label_found_waldo / gaze_plus_target

| feature                      |    weight |   abs_weight | direction   |
|:-----------------------------|----------:|-------------:|:------------|
| early_spatial_coverage_hull  | -1.55417  |     1.55417  | negative    |
| early_total_fix_duration_s   |  1.38789  |     1.38789  | positive    |
| early_gaze_entropy           |  1.33008  |     1.33008  | positive    |
| early_gaze_count             | -1.29722  |     1.29722  | negative    |
| early_saccade_length_avg     |  1.2915   |     1.2915   | positive    |
| early_scanpath_length_per_s  | -1.15749  |     1.15749  | negative    |
| early_direct_hit_count       |  1.15493  |     1.15493  | positive    |
| early_hit_any                |  1.15151  |     1.15151  | positive    |
| early_fix_count              |  1.08772  |     1.08772  | positive    |
| early_min_distance_to_target |  0.989931 |     0.989931 | positive    |
| early_waldo_hit_count        |  0.988251 |     0.988251 | positive    |
| early_peripheral_hit_count   |  0.857317 |     0.857317 | positive    |

## Linear Weight Ranking: label_long_search / gaze_only

| feature                     |    weight |   abs_weight | direction   |
|:----------------------------|----------:|-------------:|:------------|
| early_gaze_count            |  1.1963   |     1.1963   | positive    |
| early_mean_saliency         |  1.09626  |     1.09626  | positive    |
| early_scanpath_length_per_s | -1.0119   |     1.0119   | negative    |
| early_saccade_length_median | -0.96164  |     0.96164  | negative    |
| early_saccade_length_avg    |  0.920786 |     0.920786 | positive    |
| early_avg_fix_duration_s    | -0.835714 |     0.835714 | negative    |
| early_fixation_entropy_4x4  |  0.76509  |     0.76509  | positive    |
| early_fix_count             |  0.750455 |     0.750455 | positive    |
| early_spatial_coverage_hull | -0.744466 |     0.744466 | negative    |
| early_gaze_entropy          |  0.58926  |     0.58926  | positive    |
| early_fix_rate              | -0.435287 |     0.435287 | negative    |
| early_unique_grid_cells_4x4 | -0.41481  |     0.41481  | negative    |

## Sequence Attention Summary

|   position_index |   mean_attention |   median_attention |   n_sequences |
|-----------------:|-----------------:|-------------------:|--------------:|
|                0 |      0.197111    |        0.000782457 |            39 |
|                1 |      0.056104    |        0.000820762 |            39 |
|                2 |      0.0929994   |        0.000255297 |            39 |
|                3 |      0.0538733   |        8.07542e-05 |            39 |
|                4 |      0.110028    |        0.000103194 |            38 |
|                5 |      0.0411136   |        5.53333e-05 |            34 |
|                6 |      0.12655     |        6.12158e-05 |            33 |
|                7 |      0.0046994   |        8.92261e-05 |            29 |
|                8 |      0.0917456   |        0.000205082 |            28 |
|                9 |      0.0595713   |        3.45636e-05 |            25 |
|               10 |      0.0558675   |        0.000106174 |            23 |
|               11 |      0.155533    |        4.20621e-05 |            20 |
|               12 |      0.0639074   |        3.4228e-05  |            16 |
|               13 |      0.074851    |        2.79826e-05 |            14 |
|               14 |      0.00461228  |        2.61965e-05 |            11 |
|               15 |      0.214835    |        9.12134e-05 |             9 |
|               16 |      0.332254    |        0.000198141 |             3 |
|               17 |      1.57306e-06 |        1.57306e-06 |             1 |

- Large score drops in ablation indicate feature groups that carry real predictive signal.
- Large positive linear weights push the prediction toward the positive class; negative weights push against it.
- Sequence attention helps show whether later fixations inside the early window matter more than the very first ones.
