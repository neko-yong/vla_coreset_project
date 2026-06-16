# 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

本项目用于课程设计题目“基于脑启发核心集选择的轻量级 VLA 机械臂动作预测”。课程指定数据集为：

```text
lerobot/aloha_sim_transfer_cube_human
```

最终目标是从 ALOHA Sim Transfer Cube Human Demonstrations 中读取单视角图像，提取视觉特征，构建 `[视觉特征] -> [单臂 7 自由度动作]` 的回归任务，并比较随机 10% 数据和脑启发核心集选择 10% 数据在固定测试集上的 MSE。

## 当前阶段

当前已实现：

- Stage 1：使用 `LeRobotDataset` 检查数据集读取和字段结构。
- Stage 2：使用冻结 ImageNet 预训练 ResNet18 提取 512 维图像特征。
- Stage 3：固定 episode 级 train/test 划分，并生成 Random、Action-Change、Fusion Coreset 三种训练样本选择结果。
- Stage 4：使用统一 MLP 训练，并在固定测试集上评估整体 MSE 和 7 个关节的单独 MSE。
- Stage 5：生成实验结果可视化图和报告用表格。

当前核心实验流程已完成。

## 推荐环境

```powershell
conda create -n vla311 python=3.11 -y
conda activate vla311
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib tqdm datasets pillow huggingface_hub opencv-python safetensors av
pip install lerobot==0.4.4
```

也可以使用项目依赖文件安装：

```powershell
pip install -r requirements.txt
```

## 数据约定

项目主读取方式使用：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
```

Stage 1 已确认图像字段为 `observation.images.top`，图像 shape 为 `(3, 480, 640)`，范围为 `[0, 1]`。动作字段为 `action`，原始 shape 为 `(14,)`，本项目取 `action[:7]` 作为单臂 7 自由度动作标签。

固定划分规则：按 `episode_index` 排序，前 80% episode 作为训练集，后 20% episode 作为测试集。当前为训练集 16000 帧、测试集 4000 帧。测试集不参与采样、训练或标准化拟合。

## 项目结构

```text
data/
└── aloha/

outputs/
├── features/
├── results/
├── figures/
└── checkpoints/

src/
├── 01_load_dataset.py
├── 02_extract_features.py
├── 03_select_random.py
├── 04_select_action_change.py
├── 05_select_fusion_coreset.py
├── 06_train_mlp.py
├── 07_evaluate.py
├── 08_visualize.py
└── utils.py

requirements.txt
README.md
run_all.py
```

## Stage 1 运行

```powershell
python src/01_load_dataset.py --dataset_name lerobot/aloha_sim_transfer_cube_human --cache_dir data/aloha --max_samples 5
python run_all.py --stage load
```

## Stage 2 运行

调试提取前 100 帧：

```powershell
python src/02_extract_features.py --max_samples 100
```

完整提取全部 20000 帧：

```powershell
python src/02_extract_features.py --force_extract
python run_all.py --stage extract-full
```

Stage 2 输出位于 `outputs/features/`：

```text
features.npy
actions.npy
episode_ids.npy
frame_ids.npy
timestamps.npy
split_info.json
feature_info.json
```

## Stage 3 运行

```powershell
python src/03_select_random.py
python src/04_select_action_change.py
python src/05_select_fusion_coreset.py
```

或一次运行三种选择：

```powershell
python run_all.py --stage select-all
```

Stage 3 输出位于 `outputs/results/`：

```text
selected_indices_random.npy
selected_indices_action_change.npy
selected_indices_fusion.npy
random_selection_info.json
action_change_selection_info.json
fusion_selection_info.json
fusion_sample_table.csv
```

## Stage 4 运行

训练三种 10% 方法：

```powershell
python run_all.py --stage train-all
```

单独训练：

```powershell
python src/06_train_mlp.py --method random
python src/06_train_mlp.py --method action_change
python src/06_train_mlp.py --method fusion
```

可选 full data 上限参考：

```powershell
python src/06_train_mlp.py --method full
```

也可以单独重新评估某个 checkpoint：

```powershell
python src/07_evaluate.py --method random
python src/07_evaluate.py --method action_change
python src/07_evaluate.py --method fusion
```

Stage 4 输出：

```text
outputs/results/results.csv
outputs/results/eval_random.json
outputs/results/eval_action_change.json
outputs/results/eval_fusion.json
outputs/results/train_log_random.csv
outputs/results/train_log_action_change.csv
outputs/results/train_log_fusion.csv
outputs/checkpoints/mlp_random.pt
outputs/checkpoints/mlp_action_change.pt
outputs/checkpoints/mlp_fusion.pt
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

所有方法使用相同模型结构和默认训练参数。特征标准化只用当前方法的训练样本拟合 `StandardScaler`，再 transform 当前训练样本和固定测试集；动作标签不做标准化。

## Stage 5 运行

生成可视化图和报告表格：

```powershell
python src/08_visualize.py
```

或：

```powershell
python run_all.py --stage visualize
```

Stage 5 输出：

```text
outputs/figures/mse_comparison.png
outputs/figures/action_change_selected.png
outputs/figures/pca_feature_distribution.png
outputs/figures/selected_frame_distribution.png
outputs/figures/joint_mse_comparison.png
report_assets/result_tables/results_summary.csv
report_assets/result_tables/results_summary.md
```

## 后续计划

后续可继续整理最终报告文字、补充实验分析和课程展示材料。
