# 数据接口

所有 NPZ 数组均采用 `allow_pickle=False` 读取。字符串采用 NumPy Unicode 数组；不使用对象数组。
除缺失的标签/描述符外，浮点数组必须有限。

## molecules.json

```json
{"smiles": ["CCO", "CCN"], "tasks": ["endpoint_a", "endpoint_b"]}
```

`smiles` 顺序定义全数据行号。预测新分子时可以省略 `tasks`，直接读取模型中的任务顺序。
公开基准数据准备还记录 `scaffold`；随包 `assets/*_molecules.json` 是其有序参照。

## chemical_features.npz

| key | shape | 内容 |
|---|---|---|
| `count` | N × 2048 | 家族语义原子不变量的 Morgan 丰度；uint16 |
| `topology` | N × 3 × 152 | 原始药效团实例计数和距离直方图；float32 |
| `smiles` | N | 与基准分子行顺序一致 |

三个距离尺度分别为 0–3、4–7、8 及以上键；最后一格聚合距离 ≥11。
每尺度前 8 维为家族实例数，后 144 维为 36 对家族 × 4 个距离格。
不同断开片段间不计对。8 个家族的顺序定义在 `capti/features.py` 中。

## split_SEED.npz

`train`、`val`、`test` 为不相交的全数据整数行号。
`y_train` 和 `y_val` 分别为对应顺序的 N_part × T 标签；0/1 表示阴性/阳性，NaN 表示缺失。
本包的数据准备入口只输出这两种标签数组。
使用外部自行生成的文件时，调用者必须保证分子与任务顺序；数字索引不能单独证明输入数据的版本一致。

## graph_SEED.npz

`logits[N,T]` 为冻结图骨干的未 sigmoid 输出；另含 `smiles[N]`、`tasks[T]`、`train_ids[N_train]`。
神经头训练检查这三种对齐信息。构建缓存时传入的 `partition.json` 必须来自该骨干的真实训练划分。

## rf_features.npz

`fp[N,2048]`：二值 Morgan 指纹；`desc[N,217]`：按归档名称排列的描述符。
另含 `smiles`、`descriptor_names`。描述符使用 RDKit 2026.09.1 计算。

## head.pt

包含 `state`、`config`、`stats`、`tasks`、`task_names`、`seed`。
`state.fragment_mean` 是训练分子归一化对数丰度的均值。
`stats.pharmacophore_mean/std` 是训练分子的对数拓扑特征统计量。
既有权重可另含 `dataset` 与输入布局说明。只载入可信来源的 PyTorch/joblib 权重。

## 指标输出

`macro_ap`、`macro_auroc`、`eligible_task_ids`、`eligible_tasks` 和 `per_task`。
各划分的可评估终点集合可能不同，比较时应使用同一划分与相同标签掩码。
五次骨架划分的集合存在重叠，不应将其视为五个相互独立的数据集。
