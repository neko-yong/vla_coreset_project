# 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

本项目用于课程设计题目“基于脑启发核心集选择的轻量级 VLA 机械臂动作预测”。实验使用 ALOHA Sim Transfer Cube Human Demonstrations 数据集，构建从单视角图像特征到单臂 7 自由度动作的回归任务，并比较不同 10% 样本选择策略在固定测试集上的 MSE。

课程指定数据集：

```text
lerobot/aloha_sim_transfer_cube_human
```

## 已完成内容

- Stage 1：使用 `LeRobotDataset` 检查数据集读取和字段结构。
- Stage 2：使用冻结 ImageNet 预训练 ResNet18 提取 512 维图像特征。
- Stage 3：固定 episode 级 train/test 划分，并生成多种 10% 核心集选择结果。
- Stage 4：使用统一 MLP 训练，并在固定测试集上评估整体 MSE 和 7 个关节 MSE。
- Stage 5：生成实验结果图、报告表格和归档材料。

已归档实验版本：

```text
experiments/baseline_v1_random_action_fusion/
experiments/baseline_v2_add_visual_cluster/
experiments/baseline_v3_add_fusion_neighbor/
```

## 环境配置

推荐使用 Python 3.11 的 conda 环境：

```powershell
conda create -n vla311 python=3.11 -y
conda activate vla311
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib tqdm datasets pillow huggingface_hub opencv-python safetensors av
pip install lerobot==0.4.4
```

也可以使用依赖文件：

```powershell
pip install -r requirements.txt
```

## 数据与任务约定

普通 `datasets.load_dataset` 当前只能读取表格字段，无法直接获得图像。因此项目主读取方式使用：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

关键字段：

- 图像字段：`observation.images.top`
- 图像 shape：`(3, 480, 640)`
- 图像范围：`[0, 1]`
- 原始动作字段：`action`
- 原始动作 shape：`(14,)`
- 本实验动作标签：`action[:7]`

固定划分规则：

- 按 `episode_index` 排序。
- 前 40 个 episode 作为训练集，共 16000 帧。
- 后 10 个 episode 作为测试集，共 4000 帧。
- 测试集不能参与采样、聚类、训练或标准化拟合。

## 项目结构

```text
data/
└── aloha/

outputs/
├── features/
├── results/
├── figures/
└── checkpoints/

report_assets/
└── result_tables/

experiments/
├── baseline_v1_random_action_fusion/
├── baseline_v2_add_visual_cluster/
└── baseline_v3_add_fusion_neighbor/

src/
├── 01_load_dataset.py
├── 02_extract_features.py
├── 03_select_random.py
├── 04_select_action_change.py
├── 05_select_fusion_coreset.py
├── 05b_select_visual_cluster.py
├── 05c_select_fusion_neighbor.py
├── 06_train_mlp.py
├── 07_evaluate.py
├── 08_visualize.py
├── 09_archive_experiment.py
└── utils.py

requirements.txt
README.md
run_all.py
```

## Stage 1：数据读取检查

```powershell
python src/01_load_dataset.py --dataset_name lerobot/aloha_sim_transfer_cube_human --cache_dir data/aloha --max_samples 5
python run_all.py --stage load
```

该阶段只验证 LeRobotDataset 能否读取图像、动作和 episode/frame 信息，不做训练。

## Stage 2：ResNet18 特征提取

调试提取前 100 帧：

```powershell
python src/02_extract_features.py --max_samples 100
```

完整提取全部 20000 帧：

```powershell
python src/02_extract_features.py --force_extract
python run_all.py --stage extract-full
```

输出位于 `outputs/features/`：

```text
features.npy
actions.npy
episode_ids.npy
frame_ids.npy
timestamps.npy
split_info.json
feature_info.json
```

## Stage 3：核心集样本选择

基础三种方法：

```powershell
python src/03_select_random.py
python src/04_select_action_change.py
python src/05_select_fusion_coreset.py
python run_all.py --stage select-all
```

消融和扩展方法：

```powershell
python src/05b_select_visual_cluster.py
python src/05c_select_fusion_neighbor.py
python run_all.py --stage select-visual
python run_all.py --stage select-fusion-neighbor
```

方法说明：

- `random`：随机 10% 基准，无认知筛选。
- `action_change`：基于相邻动作变化的动作惊奇度。
- `visual_cluster`：仅使用视觉聚类覆盖，簇内随机采样。
- `fusion`：视觉状态覆盖 + 动作惊奇度。
- `fusion_neighbor`：在 Fusion anchor 周围加入同一 episode 内的时间邻域。

## Stage 4：MLP 训练与测试 MSE

训练基础三种 10% 方法：

```powershell
python run_all.py --stage train-all
```

单独训练：

```powershell
python src/06_train_mlp.py --method random
python src/06_train_mlp.py --method action_change
python src/06_train_mlp.py --method visual_cluster
python src/06_train_mlp.py --method fusion
python src/06_train_mlp.py --method fusion_neighbor
python src/06_train_mlp.py --method full
```

重新评估 checkpoint：

```powershell
python src/07_evaluate.py --method random
python src/07_evaluate.py --method fusion_neighbor
```

统一 MLP 结构：

```text
Linear(512, 256)
ReLU
Dropout(0.1)
Linear(256, 128)
ReLU
Linear(128, 7)
```

所有方法使用同一模型结构和训练参数；特征标准化只使用当前方法训练样本拟合，测试集不参与。

主要输出：

```text
outputs/results/results.csv
outputs/results/eval_*.json
outputs/results/train_log_*.csv
outputs/checkpoints/mlp_*.pt
```

## Stage 5：可视化与报告素材

```powershell
python src/08_visualize.py
python run_all.py --stage visualize
```

输出：

```text
outputs/figures/mse_comparison.png
outputs/figures/action_change_selected.png
outputs/figures/pca_feature_distribution.png
outputs/figures/selected_frame_distribution.png
outputs/figures/joint_mse_comparison.png
report_assets/result_tables/results_summary.csv
report_assets/result_tables/results_summary.md
```

## 实验归档

归档脚本只复制和整理当前结果，不重新运行实验。

Baseline V1：

```powershell
python src/09_archive_experiment.py --experiment_name baseline_v1_random_action_fusion
python run_all.py --stage archive-baseline
```

Baseline V2：

```powershell
python src/09_archive_experiment.py --experiment_name baseline_v2_add_visual_cluster
python run_all.py --stage archive-visual
```

Baseline V3：

```powershell
python src/09_archive_experiment.py --experiment_name baseline_v3_add_fusion_neighbor
python run_all.py --stage archive-fusion-neighbor
```

如果归档目录已存在，默认不覆盖。需要重建时显式添加：

```powershell
--overwrite
```

每个归档目录包含：

```text
results/
figures/
checkpoints/
report_assets/
experiment_note.md
file_manifest.json
README_snapshot.md
```

## 统一入口

`run_all.py` 提供分阶段运行入口，默认只执行轻量数据读取检查，避免误触发耗时任务或覆盖结果。

查看支持的 stage：

```powershell
python run_all.py --help
```
