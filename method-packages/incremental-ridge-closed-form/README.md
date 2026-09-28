# 解析增量岭回归方法包

本方法包实现 `incremental_ridge_closed_form` v0.1。它面向输入、输出和预处理契约不变的仿射线性回归，以未正则化充分统计量累计历史信息，固定求解：

```text
sum_squared_error + lambda * l2_weight_penalty
```

`lambda=0` 是普通最小二乘特例。第一版不接受其他损失、L1 正则、特征变化、输出扩展或神经网络内部线性层。

## 边界

- 方法卡供更新决策 Agent 和执行 Agent 阅读；它不是具体任务适配代码。
- Python 核心只处理内存中的 JSON 兼容矩阵，不搜索上游未声明资产、不读取数据路径、不写 checkpoint。
- Pi 工具只适合有界的小规模状态检查和更新；具体任务的数据接入、适配代码、产物保存和评估仍由执行 Agent 完成。
- 只有旧权重而没有旧充分统计量时，本方法不适用。不得通过任意初始化信息矩阵把近似更新称为同一方法。

## 契约

- 方法卡：[`method-card.json`](method-card.json)
- 通用方法卡 Schema：[`../../schemas/incremental-learning-method-card.schema.json`](../../schemas/incremental-learning-method-card.schema.json)
- 状态 Schema：[`../../schemas/incremental-ridge-state.schema.json`](../../schemas/incremental-ridge-state.schema.json)
- 更新请求 Schema：[`../../schemas/incremental-ridge-update-request.schema.json`](../../schemas/incremental-ridge-update-request.schema.json)
- 工具报告 Schema：[`../../schemas/incremental-ridge-tool-report.schema.json`](../../schemas/incremental-ridge-tool-report.schema.json)
- 产品规格：[`../../docs/product-specs/incremental-ridge-method-package.md`](../../docs/product-specs/incremental-ridge-method-package.md)

## 工具

`inspect_incremental_ridge_state` 检查：

- 方法和 Schema 版本；
- 固定目标函数；
- 矩阵 shape、有限值、Gram 对称和半正定；
- 当前参数是否是该状态和目标函数对应的解；
- 参数摘要；
- 系统矩阵条件数。

`update_incremental_ridge` 接收与已批准计划绑定的请求，验证新批次契约后累计充分统计量、求解新参数并返回新状态。它不保存产物，因此返回状态中的 `artifact_ref` 和 `artifact_sha256` 为 `null`；执行 Agent 保存实际 checkpoint 后必须建立新的正式绑定。

## 数值约定

- 数据类型固定为 `float64`；
- intercept 使用增广特征的最后一维；
- 是否正则化 intercept 由旧状态固定，更新时不能改变；
- 使用 `numpy.linalg.solve`，不显式求逆；
- 条件数大于 `1e12` 时报告 `ill_conditioned=true`，但只要仍有有限解就保留结果供执行 Agent 判断。
