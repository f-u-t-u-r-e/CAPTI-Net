# CAPTI-Net 最终方法源码

完整方法包含两个化学分支：**化学环境丰度交互**与**药效团拓扑交互**。
两分支对冻结的图预测器进行 logit 校正，最终与同划分 RF 的预测按固定 0.5:0.5 概率融合。

本包仅整理这一方法的源码和必要配置。图骨干与 RF 是最终预测流程的组成部分。
模型权重、原始标签、论文文件、图表、对比方法和消融实现均不随包提供。

## 文件

| 文件 | 功能 |
|---|---|
| `capti/model.py` | 最终两分支网络和缺失标签损失 |
| `capti/features.py`、`preprocessing.py` | 丰度与拓扑特征、训练集统计量 |
| `capti/backbone.py` | 图骨干训练与冻结 logit 缓存 |
| `capti/train_head.py` | 最终网络训练 |
| `capti/rf_features.py`、`forest.py` | 固定 RF 特征与训练 |
| `capti/predict.py` | 新分子预测和固定融合 |
| `capti/prepare.py`、`evaluate.py`、`metrics.py`、`io.py` | 数据准备、指标与文件接口 |
| `configs/` | 两数据集共用配置 |
| `assets/` | 任务、描述符、分子顺序及五次骨架划分元数据，不含标签 |
| `scripts/run_five_seeds.py` | 两数据集五次划分的统一训练入口 |

## 环境

Python 3.10；以下命令均从本包根目录执行。化学分支与 RF 特征分别使用其固定 RDKit 版本。

```bash
python3.10 -m venv .venv-core
.venv-core/bin/python -m pip install -r requirements-core.txt
python3.10 -m venv .venv-prep
.venv-prep/bin/python -m pip install -r requirements-preparation.txt
```

核心环境使用 RDKit 2024.09.6、PyTorch 2.5.1、Chemprop 1.6.1；
数据准备与 RF 特征环境使用 RDKit 2026.09.1。CUDA 用户按本机环境安装对应 PyTorch 版本。

## 训练完整方法

下面以 Tox21、seed 13 为例。ToxCast 使用 `toxcast_data.csv.gz`，将数据集名改为 `toxcast`；
五个种子为 13、23、37、53、71，所有训练参数保持一致。
原始 CSV 由用户提供；数据接口详见 `docs/DATA_FORMAT.md`。

```bash
.venv-prep/bin/python -m capti.prepare --raw raw/tox21.csv.gz --dataset tox21 --output work/data/tox21
.venv-core/bin/python -m capti.features --molecules work/data/tox21/molecules.json --output work/data/tox21/chemical_features.npz
.venv-core/bin/python -m capti.backbone train --molecules work/data/tox21/molecules.json --split work/data/tox21/split_13.npz --seed 13 --device cuda --output work/backbones/tox21/13
.venv-core/bin/python -m capti.backbone cache --checkpoint work/backbones/tox21/13/backbone.pt --partition work/backbones/tox21/13/partition.json --molecules work/data/tox21/molecules.json --device cuda --output work/data/tox21/graph_13.npz
.venv-core/bin/python -m capti.train_head --features work/data/tox21/chemical_features.npz --split work/data/tox21/split_13.npz --base work/data/tox21/graph_13.npz --molecules work/data/tox21/molecules.json --seed 13 --device cuda --output work/heads/tox21/13
.venv-prep/bin/python -m capti.rf_features --molecules work/data/tox21/molecules.json --dataset tox21 --output work/data/tox21/rf_features.npz
.venv-core/bin/python -m capti.forest train --features work/data/tox21/rf_features.npz --split work/data/tox21/split_13.npz --molecules work/data/tox21/molecules.json --seed 13 --output work/rf/tox21/13
```

五次划分的图缓存准备完成后，可用 `scripts/run_five_seeds.py --data-root work/data --output-root work/heads_all`
统一启动十次神经头拟合。输出目录必须尚不存在。

## 预测

`examples/molecules.json` 展示输入格式。`--rf-python` 指向准备环境的 Python 绝对路径。

```bash
.venv-core/bin/python -m capti.predict --molecules examples/molecules.json --head work/heads/tox21/13/head.pt --backbone work/backbones/tox21/13/backbone.pt --partition work/backbones/tox21/13/partition.json --rf-dir work/rf/tox21/13 --rf-python /absolute/path/to/.venv-prep/bin/python --dataset tox21 --device cuda --output work/predictions.npz
```

最终概率为输出中的 `p_fused`；`p_neural` 与 `p_rf` 是其两个组成部分。
输入分子顺序与任务顺序均保留。图骨干、神经头及 RF 必须来自同一划分。
不提供 RF 参数时，预测入口只输出神经部分。

## 训练规则

神经头统一使用 rank=128、dropout=0.5、AdamW lr=0.0003、weight decay=0.01，
最多 250 个 epoch，每 5 个 epoch 按验证集宏平均 AP 选取检查点，patience=50。
所有预处理统计量只由训练分子计算，缺失标签按掩码排除。
RF 每终点 400 棵树、最小叶节点样本数 2，采用 `sqrt` 特征采样及 `balanced_subsample`。
训练入口只使用训练/验证标签，不会自动生成新的留出评价。

数据和计算环境相同也不保证重新训练逐位复现；跨设备或精度变化可能带来数值差异。
