# Baseline V2: Add Visual-Cluster Only Coreset

## 1. Experiment Purpose

This version extends `baseline_v1_random_action_fusion` with a Visual-Cluster Only ablation. The goal is to verify whether visual state coverage alone helps coreset selection.

## 2. Difference from Baseline V1

- Baseline V1 contains Random 10%, Action-Change 10%, and Fusion 10%.
- Baseline V2 adds Visual-Cluster Only 10%.
- Full Data 100% is included as an upper-bound reference.

## 3. Dataset and Feature Setting

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Image field: `observation.images.top`
- Action label: `action[:7]`
- Feature extractor: frozen ImageNet-pretrained ResNet18
- Feature dimension: 512

## 4. Methods

- Random: random 10% baseline.
- Action-Change: action surprise from adjacent action changes only.
- Visual-Cluster: visual clustering coverage only, with random sampling inside each cluster.
- Fusion: visual clustering coverage plus action surprise.
- Full: 100% training set upper-bound reference.

## 5. Main Results

| method | sample_ratio | num_train_samples | num_test_samples | test_mse | joint_1_mse | joint_2_mse | joint_3_mse | joint_4_mse | joint_5_mse | joint_6_mse | joint_7_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.1 | 1600 | 4000 | 0.004198 | 0.000085 | 0.002160 | 0.003028 | 0.000982 | 0.002050 | 0.001607 | 0.019476 |
| action_change | 0.1 | 1600 | 4000 | 0.007531 | 0.000249 | 0.006123 | 0.003494 | 0.002152 | 0.008334 | 0.003501 | 0.028867 |
| visual_cluster | 0.1 | 1600 | 4000 | 0.004116 | 0.000116 | 0.001892 | 0.002525 | 0.001039 | 0.001888 | 0.001927 | 0.019428 |
| fusion | 0.1 | 1600 | 4000 | 0.006524 | 0.000133 | 0.003041 | 0.004244 | 0.001372 | 0.003670 | 0.002133 | 0.031075 |
| full | 1.0 | 16000 | 4000 | 0.003102 | 0.000124 | 0.000835 | 0.000604 | 0.000737 | 0.001656 | 0.001484 | 0.016276 |

## 6. Observation

- Among 10% methods, the lowest test MSE is `visual_cluster` (0.004116).
- `visual_cluster` is better than `random` (0.004116 vs. 0.004198).
- `visual_cluster` is better than `action_change` (0.004116 vs. 0.007531).
- `visual_cluster` is better than `fusion` (0.004116 vs. 0.006524).
- Visual-Cluster Only is better than Random 10%, suggesting that state coverage from ResNet18 feature clustering can improve coreset quality.
- Fusion does not exceed Visual-Cluster. In this task, action-change scores may overemphasize abrupt action frames and introduce distribution bias; preserving visual state coverage alone appears more stable.
- Full Data 100% is better than all 10% methods (0.003102 vs. best 10% 0.004116).

## 7. Archived Files

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `visual_cluster_sample_table.csv`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
