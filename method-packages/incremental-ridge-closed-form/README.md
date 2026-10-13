# 解析增量岭回归方法包 v0.3

## 文档定位

本文面向需要理解、评审或维护本方法的人，完整说明解析增量岭回归是什么、怎样工作以及本项目具体实现了什么。本文不作为智能体的机器契约：更新决策 Agent 和更新执行 Agent 应读取同目录下的 [`method-card.json`](method-card.json)。面向代码维护者的集成要求见 [`../../docs/product-specs/incremental-ridge-method-package.md`](../../docs/product-specs/incremental-ridge-method-package.md)。

本方法包实现 `incremental_ridge_closed_form` v0.3。它面向固定特征提取器上的解析线性任务头，以未正则化统计矩阵 `G` 和 `C` 累计历史信息，固定求解：

```text
sum_squared_error + lambda * l2_weight_penalty
```

`lambda=0` 是普通最小二乘特例。原模型及原任务头可以由交叉熵等其他损失训练；初始化阶段不沿用原任务头参数，而是根据旧`G`、`C`重新求解解析任务头。初始化后的任务头及其后续增量更新固定使用上述解析目标。

## 1. 方法是什么

本方法用于解决以下更新问题：模型的特征提取器保持固定，只更新最后的仿射线性任务头；下游不能重新访问完整旧训练集，但上游能够提供由全部旧训练样本计算得到的历史充分统计量。方法通过持续累计统计量，在不重新读取旧样本的情况下初始化并更新解析任务头。

## 2. 数学目标

设第 $i$ 个阶段的任务头输入特征为 $Z_i$，对应标签矩阵为 $Y_i$。带bias时，先在特征末列追加全1列，得到增广特征 $\tilde Z_i$。截至阶段 $t$ 的历史统计量定义为：

$$
G_t=\sum_{i=0}^{t}\tilde Z_i^\top\tilde Z_i
$$

$$
C_t=\sum_{i=0}^{t}\tilde Z_i^\top Y_i
$$

解析任务头参数为：

$$
\tilde W_t=(G_t+\lambda D)^{-1}C_t
$$

其中 $D$ 为正则矩阵。带bias且不正则化bias时：

$$
D=
\begin{bmatrix}
I_d & 0\\
0 & 0
\end{bmatrix}
$$

本方法实际求解的目标是平方误差总和加 L2 正则，而不是框架中默认按批次取平均的 MSE：

$$
\min_{\tilde W}
\left\|\tilde Z\tilde W-Y\right\|_F^2
+\lambda\,\operatorname{tr}(\tilde W^\top D\tilde W).
$$

因此，不能在不调整正则强度含义的情况下，把 `sum_squared_error` 擅自换成 mean reduction。

## 3. 完整工作流程

方法包含两个连续阶段：

1. **解析任务头初始化。** 使用上游提供的旧 $G_0$、$C_0$ 重新求解解析任务头，建立能够继续累计统计量的初始解析状态。对于原来使用交叉熵等其他目标训练的分类头，这一步属于任务头的解析重对齐，不保证新任务头一定优于或等同于原任务头。
2. **解析增量更新。** 新数据经过同一个冻结特征提取器得到 $Z_{\text{new}}$，再计算 $\Delta G=\tilde Z_{\text{new}}^\top\tilde Z_{\text{new}}$ 和 $\Delta C=\tilde Z_{\text{new}}^\top Y_{\text{new}}$，累计到历史状态并重新求解任务头。

在特征空间、标签编码、输出维度和解析目标保持不变的条件下，累计统计量后得到的任务头，与将所有旧数据和新数据放在一起进行一次联合岭回归得到的任务头相同：

$$
\tilde W_{\text{incremental}}=\tilde W_{\text{joint-ridge}}
$$

这里的等价仅针对固定特征上的平方误差加L2正则目标，不表示更新后的模型与原交叉熵模型输出相同，也不保证替换任务头后性能一定提高。原任务头、解析初始化任务头和每轮增量候选任务头的实际效果，均须由执行层在模型副本上评估。

## 4. 分类与回归怎样表示

分类任务将固定类别顺序的 one-hot 矩阵作为 $Y$。例如类别顺序为 `[低风险, 中风险, 高风险]` 时，“中风险”的目标行为 `[0, 1, 0]`；该顺序必须在初始化和全部后续更新中保持不变。解析头输出的是平方误差目标下的分数，不应未经验证直接解释为概率。

回归任务使用连续目标矩阵。单指标回归时 $Y\in\mathbb{R}^{n\times1}$；多指标回归时每一列对应一个固定指标，列顺序、业务含义、单位以及训练前变换必须保持不变。

该方法的价值是保留全部历史样本对联合岭回归解的贡献，而不是从旧模型权重中猜测历史数据，也不是通过少量旧样本近似重放完整旧训练集。

## 5. 适用条件与排除条件

只有同时满足以下条件时才能选择本方法：

1. 已确定任务头的直接输入特征及其维度和顺序；
2. 产生该特征的模型部分在解析初始化和后续更新期间保持固定；
3. 上游提供由旧训练数据计算的 $G$、$C$，或者执行层能够访问旧训练数据重新计算；
4. bias 增广规则、正则强度和是否正则化 bias 已确定；
5. 分类类别顺序或回归指标顺序固定；
6. 初始化后的解析头接受平方误差总和加 L2 正则这一目标变化；
7. 能在模型副本上分别评估解析初始化和后续增量结果。

下列情况不得直接选择本方法：

- 只有原模型权重，没有旧 $G$、$C$，也不能访问旧数据；
- 本次更新还需要改变特征提取器；
- 需要 L1、交叉熵或其他非平方误差解析目标；
- 类别集合、输出指标、特征维度、顺序或预处理发生变化；
- 不能接受用新解析头替换原梯度任务头；
- 系统矩阵不可解或产生非有限数值。

### 5.1 当前实现边界

- `method-card.json` 供更新决策 Agent 和执行 Agent 阅读；它不是具体任务适配代码。
- Python 核心只处理内存中的 JSON 兼容矩阵，不搜索上游未声明资产、不读取数据路径、不写 checkpoint。
- Pi 工具只适合有界的小规模状态检查和更新；具体任务的数据接入、适配代码、产物保存和评估仍由执行 Agent 完成。
- 只有旧权重而没有旧`G`、`C`时，本方法不适用。不得通过任意初始化信息矩阵把近似更新称为同一方法。

## 6. 文件与机器契约

- 方法卡：[`method-card.json`](method-card.json)
- 通用方法卡 Schema：[`../../schemas/incremental-learning-method-card.schema.json`](../../schemas/incremental-learning-method-card.schema.json)
- 状态 Schema：[`../../schemas/incremental-ridge-state.schema.json`](../../schemas/incremental-ridge-state.schema.json)
- 更新请求 Schema：[`../../schemas/incremental-ridge-update-request.schema.json`](../../schemas/incremental-ridge-update-request.schema.json)
- 初始化请求 Schema：[`../../schemas/incremental-ridge-initialization-request.schema.json`](../../schemas/incremental-ridge-initialization-request.schema.json)
- 工具报告 Schema：[`../../schemas/incremental-ridge-tool-report.schema.json`](../../schemas/incremental-ridge-tool-report.schema.json)
- 产品规格：[`../../docs/product-specs/incremental-ridge-method-package.md`](../../docs/product-specs/incremental-ridge-method-package.md)

## 7. 本项目怎样实现

### 7.1 三个方法入口

`initialize_incremental_ridge_state`根据原训练数据形成的`G`、`C`直接求解析任务头初始权重和偏置。它不读取也不比较原checkpoint中的任务头参数；返回的解析参数由执行Agent写入模型副本并评估。

`inspect_incremental_ridge_state` 检查已经初始化的解析状态：

- 方法和 Schema 版本；
- 固定目标函数；
- bias增广约定、矩阵shape、有限值以及`G`的对称和半正定；
- 当前参数是否是该状态和目标函数对应的解；
- 参数摘要；
- 系统矩阵条件数。

`update_incremental_ridge` 接收与已批准计划绑定的请求，检查新批次特征和标签矩阵的维度后，根据`has_bias`决定是否在特征末列追加全1列，累计`G`和`C`，求解新参数，并返回`parameter_delta.weight`和`parameter_delta.bias`。它不保存产物，因此返回状态中的 `artifact_ref` 和 `artifact_sha256` 为 `null`；执行 Agent 保存实际 checkpoint 后必须建立新的正式绑定。

分类任务把固定类别顺序的one-hot矩阵作为标签矩阵，回归任务使用连续目标矩阵。当前版本不扩展输出维度，不更新特征提取器，也不加入ACIL的随机扩展层。

### 7.2 初始化输入、状态和更新输入

初始化请求必须包含：

- 方法和 Schema 版本；
- 与已批准 `UpdatePlan` 对应的任务编号、计划版本和计划摘要；
- 基础输入维度、输出维度、是否带 bias 和 `float64` 数值类型；
- 固定目标、正则强度和 bias 正则规则；
- 历史有效样本数、基础特征维度、增广设计维度、bias 位置、$G$ 和 $C$。

初始化成功后生成的状态保存：

- `state_version`；
- 固定模型维度和目标函数；
- 累计样本数、$G$、$C$；
- 当前 `weight` 和 `bias`；
- 参数摘要以及后续需要建立的 checkpoint 绑定；
- 上一状态和更新请求的 lineage。

参数方向与 PyTorch `Linear` 保持一致：状态中的 `weight.shape == [output_features, input_features]`，带 bias 时 `bias.shape == [output_features]`。内部求解矩阵使用 `[design_feature_dim, output_features]`，返回状态时再进行转置和拆分。

更新请求不接收调用方自行计算的增量 $G$、$C$，而是接收已经由固定特征提取器产生的新批次 `features` 和对应 `targets`。工具根据旧状态中的 bias 规则生成设计矩阵、计算本批统计量、累计到旧状态并求解新参数。这避免调用方在增广列、维度和统计方向上采用另一套约定。

### 7.3 一次完整项目执行

```text
批准 UpdatePlan
→ 恢复并验证交付模型
→ 确认任务头输入位置、冻结范围、标签顺序和 bias 规则
→ 使用历史 G、C 初始化解析状态
→ 把解析参数写入模型副本
→ 评估解析初始化模型，决定是否接受任务头替换
→ 检查并保存初始解析状态
→ 使用同一冻结特征提取器处理新数据
→ 调用更新函数累计统计量并求解新参数
→ 把新参数写入新的模型副本
→ 保存新 checkpoint 和新解析状态，不覆盖旧资产
→ 分别评估旧任务和新数据表现
→ 形成执行结果和审阅证据
```

如果解析初始化模型不能满足最低性能要求，应停止后续解析更新并重新选择方法，不能因为数学求解成功就认为任务更新成功。

### 7.4 最小 Python 示例

下面使用一维带 bias 回归演示接口形状。摘要值仅用于示例：

```python
from expert_model_update.methods.incremental_ridge import (
    initialize_incremental_ridge,
    update_incremental_ridge,
)

request = {
    "report_type": "IncrementalRidgeInitializationRequest",
    "schema_version": "0.3",
    "method_id": "incremental_ridge_closed_form",
    "method_version": "0.3",
    "plan_binding": {
        "task_id": "example-update",
        "plan_version": 1,
        "update_plan_sha256": "3" * 64,
    },
    "model": {
        "input_features": 1,
        "output_features": 1,
        "has_bias": True,
        "dtype": "float64",
    },
    "objective": {
        "loss": "sum_squared_error",
        "regularization": "l2",
        "regularization_strength": 0.1,
        "regularize_bias": False,
    },
    "statistics": {
        "sample_count": 2,
        "base_feature_dim": 1,
        "design_feature_dim": 2,
        "bias_feature_position": "last",
        "G": [[1.0, 1.0], [1.0, 2.0]],
        "C": [[3.0], [4.0]],
    },
}

initial_report = initialize_incremental_ridge(request)
assert initial_report["status"] == "completed"

update_request = {
    "report_type": "IncrementalRidgeUpdateRequest",
    "schema_version": "0.3",
    "method_id": "incremental_ridge_closed_form",
    "method_version": "0.3",
    "plan_binding": request["plan_binding"],
    "old_state": initial_report["state"],
    "new_batch": {
        "features": [[2.0]],
        "targets": [[5.0]],
    },
}

update_report = update_incremental_ridge(update_request)
assert update_report["status"] == "completed"
new_weight = update_report["state"]["parameters"]["weight"]
new_bias = update_report["state"]["parameters"]["bias"]
```

真实执行不能使用示例中的任务编号或摘要，也不能只保存 `new_weight` 和 `new_bias`；新状态、checkpoint、输入计划绑定和评估结果必须一起保留。

## 8. 与 ACIL 的关键差异：固定非线性高维特征扩展

ACIL并不是直接在CNN backbone的原始输出上连接一个解析线性分类头。其完整结构在冻结的backbone与解析分类头之间加入一个随机初始化后永久固定的高维特征扩展层：

```text
输入
→ 冻结的backbone
→ 固定随机线性投影
→ ReLU
→ 高维扩展特征
→ 解析线性分类头
```

设backbone输出为 $h\in\mathbb{R}^{d}$，固定随机矩阵为 $A\in\mathbb{R}^{m\times d}$，且通常 $m>d$，则ACIL使用的扩展特征可写为：

$$
z=\operatorname{ReLU}(hA^\top)
$$

随机矩阵 $A$ 不通过反向传播训练，但它定义了后续解析学习所使用的特征空间，因此初始化后必须与backbone一起固定和保存。ReLU提供非线性，高维扩展用于缓解单一线性映射的欠拟合问题。只有线性投影而没有ReLU时，后续线性分类头仍可与投影合并为一个线性映射，不能获得同样的非线性表达能力。

当前方法包没有实现这一高维非线性扩展。上游交付的 $G$、$C$ 是在原任务头前特征 $H$ 上计算的：

$$
G_H=H^\top H,\qquad C_H=H^\top Y
$$

如果下游临时加入随机ReLU扩展 $Z=\operatorname{ReLU}(HA^\top)$，解析更新实际需要的是：

$$
G_Z=Z^\top Z,\qquad C_Z=Z^\top Y
$$

由于ReLU是非线性变换，不能只根据 $G_H$、$C_H$ 推导出 $G_Z$、$C_Z$。因此，在没有完整旧特征或扩展后历史统计量的情况下，本方法包不能在下游补加ACIL的随机非线性扩展层。

这意味着当前实现是“固定原特征提取器上的解析岭回归任务头”，不是完整ACIL实现，也不能直接沿用ACIL论文中的性能结论。原任务头替换为解析任务头后是否仍具有可接受性能，必须在模型副本上通过旧任务验证数据或代表性样例单独验证。

如果后续决定支持该扩展，上游至少需要按照双方固定的映射计算并交付：

- 固定随机投影矩阵及其维度、激活函数和数值类型；
- 使用该固定映射生成的历史扩展特征统计量 $G_Z$、$C_Z$；
- 能够在新数据上复现同一特征映射的代码和配置。

在这些材料齐备前，当前方法包继续直接使用上游声明的任务头前特征，不对历史统计量进行无法验证的非线性变换。

## 9. 数值约定和失败处理

- 数据类型固定为 `float64`；
- `has_bias=true` 时，设计矩阵在基础特征末列追加全1列，`design_feature_dim=base_feature_dim+1`；
- `has_bias=false` 时不追加全1列，bias必须为`null`；
- 是否正则化bias由旧状态固定，更新时不能改变；
- 使用 `numpy.linalg.solve`，不显式求逆；
- 条件数大于 `1e12` 时报告 `ill_conditioned=true`，但只要仍有有限解就保留结果供执行 Agent 判断。

当前工具最多接收 256 个基础输入特征和 64 个输出。矩阵必须具有确定形状且全部是有限数值；$G$ 必须满足对称和半正定检查；状态中的参数必须能由该状态的统计量与目标重新求得。法方程不可解、结果非有限、状态摘要不一致、目标变化、维度变化或 bias 约定变化时，更新被拒绝。

`ill_conditioned=true` 不是“已经验证可以发布”。它表示数值结果对数据扰动可能敏感，执行 Agent 必须结合正则强度、参数变化和真实评估决定是否停止或重新规划。

## 10. 验证要求

测试必须至少覆盖：

- 使用旧 $G$、$C$ 初始化，而不是沿用原梯度任务头；
- 带 bias 和不带 bias 的形状与参数拆分；
- `lambda=0` 的普通最小二乘特例；
- 分类 one-hot 多输出和连续回归目标；
- 增量解与旧、新数据联合岭回归解等价；
- 旧状态不被原地修改；
- 状态版本、摘要和 lineage 正确递增；
- 目标、维度、bias、矩阵和非有限数值错误被拒绝。

运行测试：

```text
python -m unittest tests.test_incremental_ridge
```

算法测试只证明矩阵实现满足方法契约。真实模型仍必须验证解析参数已经正确写回指定任务头、新 checkpoint 可以严格加载、原资产没有被覆盖，并完成新旧指标比较。
