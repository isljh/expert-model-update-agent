# 专家模型增量更新智能体

本项目构建一个面向专业小模型的增量更新智能体。上游交付可执行模型定义、构造配置、原权重和新数据，但不需要交付原训练工程或训练过程。系统负责恢复原模型、选择更新方法、实施训练与评估，并最终交付新权重和可追溯证据。

## 当前输入约定

TaskInput v0.3 要求三个原模型资产：

- model_file：包含声明模型类的 Python 文件；
- config：声明模型类名、完整构造参数和 checkpoint 加载约定；
- checkpoint：与声明结构对应的 PyTorch 权重文件。

权重张量形状可以用于核验，但不能作为完整构造参数的通用推断来源。系统不再接收完整目标项目，也不再执行项目理解、模型候选发现或源码调用链还原。

## 当前前置流程

```text
TaskInput 受理与人工确认
→ 四项资产入口预检
→ 构造配置安全解析
→ checkpoint 安全结构检查
→ 唯一模型交付静态验证
→ 本地子进程创建模型并 strict 加载权重
→ 模型结构和可训练参数报告
```

当前实现能够：

- 校验 TaskInput v0.3 和固定必答问题；
- 预检模型文件、配置、checkpoint 和新数据；
- 安全解析声明式构造配置；
- 使用 weights_only 模式检查 checkpoint；
- 静态核对模型类及构造参数；
- 严格加载权重并要求 missing/unexpected keys 均为空；
- 分析注册模块、参数、缓冲区、共享关系和 requires_grad 状态；
- 保存完整 JSON 报告和逐层 Markdown 结构树。

当前加载仍然没有操作系统级隔离。导入上游模型文件和调用构造函数可能产生副作用；本地子进程、超时和离线环境变量不等于沙箱。结构分析不执行 forward，也不证明完整训练链已经可用。

## 文档入口

- [项目全局边界](AGENTS.md)
- [系统架构](docs/architecture.md)
- [任务受理与 TaskInput v0.3](docs/product-specs/task-intake.md)
- [模型配置与 checkpoint 安全检查](docs/product-specs/model-asset-evidence-inspection.md)
- [模型交付验证](docs/product-specs/model-delivery-validation.md)
- [模型加载、严格权重恢复与结构分析](docs/product-specs/safe-model-instantiation.md)
- [最小产品与人工批准边界](docs/product-specs/minimal-model-update-agent.md)
- [解析增量岭回归方法包](docs/product-specs/incremental-ridge-method-package.md)
- [Harness Engineering 原则](docs/design-docs/harness-engineering-principles.md)
- [Pi Agent Loop](docs/references/pi-agent-loop.md)

## 主要机器契约

- [TaskInput Schema](schemas/task-input.schema.json)
- [AssetPreflightReport Schema](schemas/asset-preflight-report.schema.json)
- [Model Load Config Schema](schemas/model-load-config.schema.json)
- [Config Inspection Report Schema](schemas/config-inspection-report.schema.json)
- [Model Delivery Validation Report Schema](schemas/model-delivery-validation-report.schema.json)
- [Model Instantiation Report Schema](schemas/model-instantiation-report.schema.json)
- [Model Structure Analysis Report Schema](schemas/model-structure-analysis-report.schema.json)

## Pi 工具

- validate_task_input
- preflight_task_assets
- inspect_model_config
- inspect_model_checkpoint
- validate_model_delivery
- load_validated_model
- analyze_validated_model_structure
- inspect_incremental_ridge_state
- update_incremental_ridge

## 当前方法包

incremental_ridge_closed_form v0.1 固定实现平方误差总和加 L2 正则的解析增量岭回归，要求旧充分统计量可用且特征、目标、预处理和正则约定不变。它不代表任意模型都能仅凭权重完成解析增量更新。

## 当前未完成

- 更新决策 Agent 和版本化 UpdatePlan；
- 主损失、输入输出语义及预处理的正式交付契约；
- 项目无关的数据适配、训练与评估执行层；
- 新 checkpoint 保存与原模型对比；
- 操作系统级隔离；
- 独立审阅与最终交付。
