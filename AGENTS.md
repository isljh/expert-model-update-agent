# 智能体项目入口

本仓库用于构建“专家模型增量更新智能体”。本文件只提供导航和全局边界，详细内容按任务读取对应文档。

## 项目目标

系统接收上游交付的专业小模型文件、构造配置、原 checkpoint 和新数据，在不依赖原训练工程的前提下恢复模型实例、制定更新策略、执行训练与评估，并交付新权重、效果指标和可追溯记录。

## 阅读地图

- 项目定位和当前状态：[README.md](README.md)
- 主流程与 Agent 分工：[docs/architecture.md](docs/architecture.md)
- 最小产品与批准节点：[docs/product-specs/minimal-model-update-agent.md](docs/product-specs/minimal-model-update-agent.md)
- TaskInput v0.3：[docs/product-specs/task-intake.md](docs/product-specs/task-intake.md)
- 模型配置与 checkpoint 检查：[docs/product-specs/model-asset-evidence-inspection.md](docs/product-specs/model-asset-evidence-inspection.md)
- 模型交付验证：[docs/product-specs/model-delivery-validation.md](docs/product-specs/model-delivery-validation.md)
- 模型加载与结构分析：[docs/product-specs/safe-model-instantiation.md](docs/product-specs/safe-model-instantiation.md)
- 解析增量岭回归方法包：[docs/product-specs/incremental-ridge-method-package.md](docs/product-specs/incremental-ridge-method-package.md)
- Harness Engineering 原则：[docs/design-docs/harness-engineering-principles.md](docs/design-docs/harness-engineering-principles.md)
- Pi Agent Loop：[docs/references/pi-agent-loop.md](docs/references/pi-agent-loop.md)

## 已确定的全局边界

1. 上游必须提供模型文件、构造配置和 checkpoint；三项均为必需资产。
2. 不要求上游提供原训练源码、训练循环、优化器、学习率或训练日志。
3. 系统不扫描完整项目、不生成模型候选、不从权重形状猜测完整构造参数。
4. 模型文件必须声明唯一模型类；配置必须给出类名、完整关键字构造参数和 checkpoint 加载约定。
5. 原模型文件、配置和 checkpoint 只读，不得覆盖；新产物写入独立位置。
6. 只有模型类成功创建、checkpoint 使用安全模式严格加载且无 missing/unexpected keys，才能声称原模型已恢复。
7. 结构和 requires_grad 检查不等于 forward、反向传播或训练链已经验证。
8. 更新方法可以使用新的优化实现，但不能在没有明确决策的情况下改变核心任务目标、主损失含义或输入输出语义。
9. 更新决策 Agent 负责方法选择与理由；更新执行 Agent 围绕交付模型实现本次训练和评估；独立审阅 Agent 核对原始证据。
10. 重要决定必须进入文档、Schema、测试或运行记录，不能只留在聊天中。
11. 当前模型加载使用本地子进程但没有操作系统级隔离，不能宣称外部模型代码只读、断网或无副作用。
12. Pi 工作区只保存正式运行内容；开发测试、参考答案和评估材料在工作区外维护。

## 当前阶段

前置流程已重构为 TaskInput v0.3、四项资产预检、声明配置和 checkpoint 安全检查、唯一模型交付验证、本地子进程模型创建、strict checkpoint 加载、模型结构与可训练参数分析及正式报告保存。

旧的模型项目理解 Skill、多候选 ModelCreationProposalSet 和源码调用链静态验证已删除。首个方法包 incremental_ridge_closed_form v0.1 保留，但只适用于固定平方误差加 L2 正则、契约不变且旧充分统计量可用的仿射线性回归。

下一阶段包括主损失与输入输出语义的交付契约、更新决策、版本化 UpdatePlan、训练评估执行层和独立审阅。尚未确定的技术选型不得根据示意图自行补全。
