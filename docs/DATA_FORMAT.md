# 数据接口

NPZ 使用 `allow_pickle=False` 读取；字符串为 NumPy Unicode 数组。特征和概率必须有限。二值标签为 0/1，缺失标签使用 NaN。

## molecules.json

```json
{"smiles": ["CCO", "CCN"], "tasks": ["endpoint_a", "endpoint_b"], "ids": [0, 1]}
```

完整基准 `smiles` 顺序定义全数据行号；公开准备流程另记录 `scaffold`。新分子预测可省略 `tasks`；模型决定任务顺序。`ids` 可选，提供时必须为每行一个唯一整数，用于与单独标签文件对齐。

## chemical_features.npz

| key | shape | 内容 |
|---|---|---|
| `count` | N × 2048 | typed Morgan 丰度，uint16 |
| `topology` | N × 3 × 152 | 原始药效团实例计数与距离谱，float32 |
| `smiles` | N | 有序 SMILES |

距离尺度为 0–3、4–7、≥8 键，末格聚合 ≥11。每个尺度前 8 维为家族实例数，后 144 维为 36 对家族 × 4 个距离格。断开片段间不计对；家族顺序见 `capti/features.py`。

## split_SEED.npz

`train/val/test` 为不相交的全数据整数行号。`y_train/y_val` 为对应顺序的二维标签。数据准备只输出这两种标签矩阵；训练不会读取测试标签。基准清洗和划分必须与 `assets/` 中的有序元数据一致。

## graph_SEED.npz

`logits[N,T]` 是冻结图编码器的未 sigmoid 输出；另含 `smiles[N]`、`tasks[T]`、`train_ids[N_train]`。训练核对全部对齐信息。图缓存命令还生成 `.npz.json`，包含缓存及 checkpoint 的 SHA256；训练读取并核验它，然后将 checkpoint 哈希保存在 head 中。

## head.pt

包含 `state/config/stats/tasks/task_names/seed`，新模型另有 `train_ids/prediction_mode` 和缓存元数据提供的 `graph_checkpoint_sha256`。`state.fragment_mean` 是训练分子的归一化对数丰度均值；`stats.pharmacophore_mean/std` 是训练分子的对数拓扑统计量。仅载入可信来源的 PyTorch 权重。

`base_scale/i1/i2/product_i1/product_i2` 记录图系数、两条化学路径及其交互项开关。`legacy_input_layout` 只控制旧 GPU 缓存的输入步幅兼容，不增加化学特征。

## 预测和评价标签

最终预测：`p[N,T]`、`smiles[N]`、`tasks[T]`，用于评价时还需 `ids[N]`。所有概率为模型 sigmoid 输出。

显式留出标签：`y[N,T]`、`ids[N]`、`tasks[T]`。评价入口要求 ids 与任务顺序严格相同，不自动重排。验证集也可直接引用 development split 和 `validation_predictions.npz`。

指标输出为 `macro_ap/macro_auroc/eligible_task_ids/eligible_tasks/per_task`。每个 assay 仅使用已观测标签，且必须同时包含阳性和阴性。无可评估 assay 时宏指标为 JSON null。

## 相似度分箱

训练参考索引来自 `split.train`，查询索引来自标签/预测的 `ids`；两者必须不相交。最近邻相似度采用 Morgan radius=2、2048 binary bits、无手性的 Tanimoto 最大值。

- `<0.3`：s < 0.3
- `0.3-0.5`：0.3 ≤ s < 0.5
- `0.5-0.7`：0.5 ≤ s ≤ 0.7
- `>0.7`：s > 0.7

`scores.csv` 包含每个模型/分箱的分子数、可评估 assay 数、宏平均标签比例、AP 和 AUROC。`task_scores.csv` 保留逐 assay 观测数、阳性数及指标。`domains.csv` 保留查询全局行号、最近训练行号、相似度及分箱。

bin-specific 使用每个 bin 内有两类标签的 assay；common-across-bins 使用四个 bin 均满足条件的交集。各模型共享标签和 assay 集合。空 bin 或无共同 assay 不伪造分数，以空宏指标报告。五个划分有重叠，均值和样本 SD 描述划分波动。
