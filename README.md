# CAPTI-Net

Chemical Abundance and Pharmacophore Topology Interaction Network。当前代码对应 **2026-10-19 的正文最终口径**：分子图、化学丰度和药效团拓扑三条路径，通过共享 assay head 输出独立神经预测。

```text
z = 0.5 × graph_logits + bias + abundance_readout + topology_readout
p = sigmoid(z)
```

图 logit 的 `0.5` 是模型内部的固定系数。最终概率不再进行 RF 融合。当前主比较不包含 Random forest 或 AttentiveFP。原始数据划分、特征定义、验证集 AP 选模和五划分均值±样本 SD 的口径保持不变。

## 正文结果

| 数据集 | CAPTI-Net AP (%) | AUROC (%) | 全模型参数量 (M) |
|---|---:|---:|---:|
| Tox21 | 50.231 ± 3.495 | 82.812 ± 1.659 | 0.693 |
| ToxCast | 39.594 ± 2.048 | 70.975 ± 1.134 | 1.496 |

以上为五个既有骨架划分的测试结果。未舍入分数、消融、外部 I/II/III 和相似度分箱结果见 [`results/`](results/README.md)。主比较中的 GINE 分数采用作者配置的 5 层、宽度 300、从头监督训练结果。`results/` 是已完成实验的分数档案，运行软件测试不会重做这些实验。

仓库提供 CAPTI-Net 方法、七种路径组合及两种交互控制的训练/预测代码，以及相似度分箱分析。基准模型源码、训练权重、原始标签和论文 PDF 不随此代码发布。自动数据准备支持 Tox21/ToxCast；hERG 的三个外部队列分数作为已完成实验结果提供。

## 环境

基准训练环境为 Python 3.10。核心环境计算化学特征和模型；单独的数据准备环境保留原始清洗时的 RDKit 版本，以重建相同的分子顺序和骨架划分。

```bash
python3.10 -m venv .venv-core
.venv-core/bin/python -m pip install -r requirements-core.txt
python3.10 -m venv .venv-prep
.venv-prep/bin/python -m pip install -r requirements-preparation.txt
```

核心环境：RDKit 2024.09.6、PyTorch 2.5.1、Chemprop 1.6.1。数据准备：RDKit 2026.09.1。Windows 中将 `.venv-core/bin/python` 替换为 `.venv-core\Scripts\python.exe`，准备环境同理。CUDA 用户安装匹配硬件的 PyTorch wheel。

## 训练

所有命令从仓库根目录执行。以 Tox21、seed 13 为例，原始 CSV 由用户提供；ToxCast 使用 `toxcast_data.csv.gz`，并将数据集名改为 `toxcast`。五个种子为 13、23、37、53、71。输出目录必须尚不存在。

```bash
.venv-prep/bin/python -m capti.prepare --raw raw/tox21.csv.gz --dataset tox21 --output work/data/tox21
.venv-core/bin/python -m capti.features --molecules work/data/tox21/molecules.json --output work/data/tox21/chemical_features.npz
.venv-core/bin/python -m capti.backbone train --molecules work/data/tox21/molecules.json --split work/data/tox21/split_13.npz --seed 13 --device cuda --output work/backbones/tox21/13
.venv-core/bin/python -m capti.backbone cache --checkpoint work/backbones/tox21/13/backbone.pt --partition work/backbones/tox21/13/partition.json --molecules work/data/tox21/molecules.json --device cuda --output work/data/tox21/graph_13.npz
.venv-core/bin/python -m capti.train_head --features work/data/tox21/chemical_features.npz --split work/data/tox21/split_13.npz --base work/data/tox21/graph_13.npz --molecules work/data/tox21/molecules.json --seed 13 --variant full --device cuda --output work/heads/tox21/13
```

图预测器先按验证 AP 选模，然后冻结。`train_head` 训练丰度、拓扑路径及共享预测头；`head.pt` 保存这整个阶段的模型与训练集预处理统计量。其统一配置为 rank=128、dropout=0.5、AdamW lr=0.0003、weight decay=0.01、最多 250 epochs，每 5 epochs 按验证集宏 AP 选模，patience=50。所有训练入口只读取训练/验证标签。

两个数据集的五份图缓存准备后，可统一启动十次拟合：

```bash
.venv-core/bin/python scripts/run_five_seeds.py --data-root work/data --output-root work/heads_all --variant full --device cuda
```

## 消融

`--variant` 明确指定保留的路径。每个条件单独拟合并按验证 AP 选模，输出到不同目录；测试数据不参与选择。

| variant | Graph | Abundance | Topology |
|---|:---:|:---:|:---:|
| `full` | ✓ | ✓ | ✓ |
| `abundance_topology` | — | ✓ | ✓ |
| `graph_topology` | ✓ | — | ✓ |
| `graph_abundance` | ✓ | ✓ | — |
| `graph_only` | ✓ | — | — |
| `abundance_only` | — | ✓ | — |
| `topology_only` | — | — | ✓ |

`no_abundance_products` 和 `no_topology_products` 分别关闭丰度 co-occurrence 和拓扑 cross-scale products，并保留对应 first-order 表征。`graph_only` 仅优化共享 assay bias，保留已拟合的图编码器和 0.5 logit 系数。所有拟合使用同一输入缓存与划分核验；不含图路径的条件将图输入清零。

## 预测与评价

`examples/molecules.json` 展示 SMILES 输入。任务顺序来自模型，输入行顺序不变。

```bash
.venv-core/bin/python -m capti.predict --molecules examples/molecules.json --head work/heads/tox21/13/head.pt --backbone work/backbones/tox21/13/backbone.pt --device cuda --output work/predictions.npz
```

最终概率位于 **`p[N,T]`**，另含 `smiles`、`tasks`，以及输入 JSON 提供时的 `ids`。该格式与评价入口一致。不含图路径的模型无需 `--backbone`。新训练的 head 从图缓存元数据保存 checkpoint 哈希，并在预测时核对；旧 head 保持权重格式兼容，但无法补回未保存的来源信息。

验证集预测可直接评价；持有显式留出标签时可单独传入 `ids/y/tasks` 文件：

```bash
.venv-core/bin/python -m capti.evaluate --split work/data/tox21/split_13.npz --part val --predictions work/heads/tox21/13/validation_predictions.npz --output work/validation_metrics.json
.venv-core/bin/python -m capti.evaluate --labels work/test_labels.npz --predictions work/test_predictions.npz --output work/test_metrics.json
```

AP 和 AUROC 按 assay 计算，仅纳入具有两类观测标签的 assay，再等权宏平均。并列分数按相同阈值整体处理，与 scikit-learn 定义一致。五划分汇总使用样本 SD（`ddof=1`）。

## Chemical novelty / applicability domain

`capti.applicability` 仅对已有预测评分，不训练或选择模型。参考集仅包含当前划分的训练分子；指纹为 Morgan radius 2、2048 binary bits、无手性。固定分箱为 `<0.3`、`0.3–0.5`、`0.5–0.7`、`>0.7`，精确边界与数据接口见 [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md)。

```bash
.venv-core/bin/python -m capti.applicability --molecules work/data/tox21/molecules.json --split work/data/tox21/split_13.npz --labels work/test_labels.npz --predictions CAPTI-Net=work/test_predictions.npz --predictions D-MPNN=work/dmpnn_test.npz --predictions MolCLR=work/molclr_test.npz --output work/domain/tox21/13
```

输出包含分子最近邻、逐 assay 指标和分箱宏指标，同时给出 bin-specific 与 common-across-bins 两种共享 assay 口径。对五个划分分别运行后再汇总；各 bin 的样本量、阳性比例和 assay 覆盖一起解释。

## 核验

```bash
.venv-core/bin/python -m unittest discover -s tests -v
.venv-core/bin/python scripts/verify_results.py
```

测试使用明确标记的合成软件输入，检查路径开关、缺失标签与并列分数、预测文件接口、训练集统计量、化学特征和相似度边界。`verify_results.py` 另行核对真实归档分数的均值、SD 和消融配对关系。固定实验配置不保证跨设备或精度变化下逐位重现。

## 从 1.0 迁移

2.0 删除了 RF 训练、特征和融合入口，预测 key 统一为 `p`；请更新读取旧 `p_fused/p_neural` 的调用代码。原有化学路径权重键和三路径完整模型的 logits 计算保持兼容。分子顺序与十组原始划分元数据没有变更。
