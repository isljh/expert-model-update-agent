# 最小模型更新智能体规格

## 1. 最小目标

最小产品接收一个可训练的专业小模型交付包和新数据，完成：

1. 确认任务目标、数据用途和保留能力；
2. 恢复原模型并证明权重完整匹配；
3. 收集方法选择所需证据；
4. 形成版本化 UpdatePlan 并等待人工批准；
5. 在独立工作区实施更新、训练和评估；
6. 保存新 checkpoint、指标、配置和运行记录；
7. 由独立审阅 Agent 给出交付建议。

不要求上游提供原训练项目或训练过程。

## 2. 最小输入

### 2.1 原模型交付

- 包含唯一 PyTorch 模型类的模型文件；
- 符合固定 Schema 的构造配置；
- 与该模型结构对应的 checkpoint。

### 2.2 更新任务

- 新数据及其允许用途；
- 希望获得或改善的能力；
- 必须保留的旧能力，或明确为无；
- 业务成功期望；
- 核心任务、主损失含义、输入输出和预处理语义。

最后一组语义是更新决策必需证据，但其正式机器契约尚待设计。不能因为原训练过程不可得，就默认允许改变这些语义。

## 3. 前置门槛

进入更新决策前必须完成：

```text
TaskInput confirmed
→ AssetPreflightReport passed
→ ConfigInspectionReport inspected
→ CheckpointInspectionReport inspected
→ ModelDeliveryValidationReport ready_for_model_load=true
→ ModelInstantiationReport status=completed
→ checkpoint fully_matched=true
→ ModelStructureAnalysisReport available
```

任何一步失败都不能直接进入训练。

## 4. 三个 Agent 的职责

| 角色 | 负责 | 不负责 | 输出 |
| --- | --- | --- | --- |
| 更新决策 Agent | 明确更新目标；识别不变量；比较增量方法；核对理论与数据前提；形成版本化计划 | 导入未验证模型；猜构造参数；擅改核心目标；直接训练 | UpdatePlan |
| 更新执行 Agent | 在受控工作区建立本次数据适配、训练、评估和保存逻辑；执行已批准计划；前提失败时反馈 | 覆盖原资产；绕过批准；前提不成立仍强行实施 | ExecutionResult 或重规划请求 |
| 独立审阅 Agent | 核对任务、计划、代码、日志、指标和 checkpoint；判断是否满足验收 | 修改实现；依据执行 Agent 摘要自行放宽标准 | ReviewResult |

## 5. 四类逻辑契约

### TaskInput

记录用户原始请求、更新目标、三项原模型资产、新数据用途、保留能力、成功期望和人工确认。结构见 task-input.schema.json。

### UpdatePlan

至少包括：

- 计划版本和关联 TaskInput 版本；
- 原任务目标与本次更新目标；
- 必须保持不变的损失、输入输出和预处理语义；
- 选择的方法及理由；
- 方法前提和已获得证据；
- 对旧数据贡献的替代或近似方式；
- 允许修改和写入的范围；
- 训练、验证、评估方案；
- 成功、退化和停止条件；
- 需要生成的交付物；
- 未解决风险。

UpdatePlan 尚未形成正式 Schema。

### ExecutionResult

至少包括：

- 实际使用的计划版本；
- 数据和模型资产摘要；
- 实际训练配置；
- 新 checkpoint 路径与摘要；
- 原模型和新模型的对比指标；
- 执行日志和失败重试；
- 与计划的偏差；
- 是否请求重规划。

### ReviewResult

至少包括：

- 审阅的任务、计划和执行结果版本；
- 对新 checkpoint 可加载性的复验；
- 指标与日志一致性；
- 旧能力保持情况；
- 未解决问题和风险等级；
- 通过、带风险通过、需要修改或重新规划的建议。

## 6. 状态流

```text
draft
→ waiting_for_confirmation
→ confirmed
→ assets_verified
→ model_restored
→ planning
→ waiting_for_plan_approval
→ executing
→ evaluating
→ reviewing
→ waiting_for_delivery_approval
→ completed
```

失败可以进入 needs_input、needs_replan 或 failed。任何输入或计划内容变化都必须递增版本，并使依赖旧版本的批准失效。

## 7. 人工批准节点

至少保留三个明确节点：

1. TaskInput 确认：用户确认系统正确理解任务和资产；
2. UpdatePlan 批准：人类允许按该方法和资源边界执行训练；
3. 最终交付批准：人类依据独立审阅决定是否接受新模型。

原模型资产永远不能作为输出覆盖目标。

## 8. 完成定义

只有同时具备以下证据才能声明任务完成：

- 新 checkpoint 能在声明模型结构上重新加载；
- 原模型与新模型使用可比较的评估协议；
- 新能力达到批准计划的标准；
- 必须保留的旧能力满足退化约束；
- 训练配置、代码、指标和日志可追溯；
- 独立审阅完成；
- 人工批准最终交付。

仅完成模型加载、代码编写或训练启动都不构成完成。

## 9. 当前实现与缺口

已实现：

- TaskInput v0.3；
- 三项模型资产与新数据预检；
- 配置和 checkpoint 安全检查；
- 模型交付验证；
- 本地子进程实例化与 strict 权重加载；
- 结构和可训练参数报告；
- 解析增量岭回归方法包。

未实现：

- 任务语义和主损失的正式 Schema；
- UpdatePlan、ExecutionResult、ReviewResult Schema；
- 通用训练与评估执行层；
- 新模型交付和回滚；
- 操作系统级隔离；
- 独立审阅闭环。
