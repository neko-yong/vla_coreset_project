# Baseline V1: Random / Action-Change / Fusion

## 1. Experiment Purpose

This version is the first complete closed-loop baseline experiment. It is archived as a comparison reference for later optimization experiments.

## 2. Dataset

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Total episodes: 50
- Total frames: 20000
- Image field: `observation.images.top`
- Action label: `action[:7]`

## 3. Feature Extractor

- Frozen ImageNet-pretrained ResNet18
- Feature dimension: 512
- Input image tensor shape: `3 x 480 x 640`
- Saved feature file: `outputs/features/features.npy`

## 4. Train/Test Split

- Train episodes: first 40 episodes
- Test episodes: last 10 episodes
- Train samples: 16000
- Test samples: 4000
- Test set is fixed and never used for sampling or scaler fitting.

## 5. Sampling Methods

- Random 10%: random baseline sampled from the training frames.
- Action-Change Coreset 10%: uses adjacent action difference within the same episode as action surprise.
- Fusion Coreset 10%: uses visual KMeans clustering for state coverage, then selects high action-surprise samples inside each cluster.

## 6. MLP Setting

- Model: `512 -> 256 -> 128 -> 7`
- Loss: `MSELoss`
- Optimizer: `Adam`
- Epochs: 100
- Batch size: 128
- Learning rate: 0.001
- Seed: 42

## 7. Main Results

| method | sample_ratio | num_train_samples | num_test_samples | test_mse | joint_1_mse | joint_2_mse | joint_3_mse | joint_4_mse | joint_5_mse | joint_6_mse | joint_7_mse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random | 0.1 | 1600 | 4000 | 0.004198 | 0.000085 | 0.002160 | 0.003028 | 0.000982 | 0.002050 | 0.001607 | 0.019476 |
| action_change | 0.1 | 1600 | 4000 | 0.007531 | 0.000249 | 0.006123 | 0.003494 | 0.002152 | 0.008334 | 0.003501 | 0.028867 |
| fusion | 0.1 | 1600 | 4000 | 0.006524 | 0.000133 | 0.003041 | 0.004244 | 0.001372 | 0.003670 | 0.002133 | 0.031075 |
| full | 1.0 | 16000 | 4000 | 0.003102 | 0.000124 | 0.000835 | 0.000604 | 0.000737 | 0.001656 | 0.001484 | 0.016276 |

## 8. Current Observation

- Lowest test MSE: `full` (0.003102).
- Highest test MSE: `action_change` (0.007531).
- Fusion improves over Action-Change (0.006524 vs. 0.007531).
- Fusion does not improve over Random (0.006524 vs. 0.004198).
- Fusion not outperforming Random is not an experimental failure. It suggests that, in the current relatively regular dataset, random 10% sampling may already cover a broad state distribution, while frozen ResNet18 features and KMeans clusters may not fully encode the robot manipulation state relevant to action prediction.

## 9. Files Archived

- `results.csv`
- `eval_*.json`
- `train_log_*.csv`
- `selected_indices_*.npy`
- `mse_comparison.png`
- `action_change_selected.png`
- `pca_feature_distribution.png`
- `selected_frame_distribution.png`
- `joint_mse_comparison.png`
- `mlp_*.pt`
