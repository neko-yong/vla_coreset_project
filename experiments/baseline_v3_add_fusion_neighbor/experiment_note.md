# Baseline V3: Add Fusion + Temporal Neighbor Coreset

## 1. Experiment Purpose

This version extends baseline_v2 with Fusion + Temporal Neighbor Coreset. The goal is to test whether local temporal context around high action-surprise key frames helps coreset selection.

## 2. Difference from Previous Versions

- Baseline V1 contains Random 10%, Action-Change 10%, and Fusion 10%.
- Baseline V2 adds Visual-Cluster Only 10%.
- Baseline V3 adds Fusion + Temporal Neighbor 10%.
- Full Data 100% is included as an upper-bound reference.

## 3. Dataset and Feature Setting

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Image field: `observation.images.top`
- Action label: `action[:7]`
- Feature extractor: frozen ImageNet-pretrained ResNet18
- Feature dimension: 512
- Train episodes: first 40 episodes
- Test episodes: last 10 episodes
- Train samples: 16000
- Test samples: 4000

## 4. Methods

- Random: random 10% baseline.
- Action-Change: action surprise from adjacent action changes only.
- Visual-Cluster: visual clustering coverage only, with random sampling inside each cluster.
- Fusion: visual clustering coverage plus action surprise.
- Fusion-Neighbor: adds t-1 / t+1 temporal neighbors from the same episode around high action-surprise Fusion anchors.
- Full: 100% training set upper-bound reference.

## 5. Main Results

| method | sample_ratio | num_train_samples | num_test_samples | test_mse | joint_1_mse | joint_2_mse | joint_3_mse | joint_4_mse | joint_5_mse | joint_6_mse | joint_7_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.1 | 1600 | 4000 | 0.004198 | 0.000085 | 0.002160 | 0.003028 | 0.000982 | 0.002050 | 0.001607 | 0.019476 |
| action_change | 0.1 | 1600 | 4000 | 0.007531 | 0.000249 | 0.006123 | 0.003494 | 0.002152 | 0.008334 | 0.003501 | 0.028867 |
| visual_cluster | 0.1 | 1600 | 4000 | 0.004116 | 0.000116 | 0.001892 | 0.002525 | 0.001039 | 0.001888 | 0.001927 | 0.019428 |
| fusion | 0.1 | 1600 | 4000 | 0.006524 | 0.000133 | 0.003041 | 0.004244 | 0.001372 | 0.003670 | 0.002133 | 0.031075 |
| fusion_neighbor | 0.1 | 1600 | 4000 | 0.006585 | 0.000123 | 0.004673 | 0.003747 | 0.001428 | 0.004075 | 0.002451 | 0.029597 |
| full | 1.0 | 16000 | 4000 | 0.003102 | 0.000124 | 0.000835 | 0.000604 | 0.000737 | 0.001656 | 0.001484 | 0.016276 |

## 6. Observation

- Lowest test MSE among all methods: `full` (0.003102).
- Lowest test MSE among 10% methods: `visual_cluster` (0.004116).
- `fusion_neighbor` is not better than `random` (0.006585 vs. 0.004198).
- `fusion_neighbor` is not better than `fusion` (0.006585 vs. 0.006524).
- `fusion_neighbor` is not better than `visual_cluster` (0.006585 vs. 0.004116).
- `visual_cluster` is better than `random` (0.004116 vs. 0.004198).
- Visual-Cluster Only is the best 10% method, suggesting that visual state coverage is more important than pure action-change emphasis in the current task.
- Fusion + Temporal Neighbor does not improve Fusion. Under a fixed 10% budget, temporal expansion preserves local context but consumes samples that could otherwise cover more anchors or visual states.
- Fusion + Temporal Neighbor does not improve Visual-Cluster. This suggests that, with single-frame ResNet18 features, preserving visual state coverage is more effective than expanding neighborhoods around action-change frames.
- Full Data 100% is better than all 10% methods (0.003102 vs. best 10% 0.004116).

## 7. Archived Files

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `fusion_neighbor_sample_table.csv`
- `fusion_neighbor_selection_info.json`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
