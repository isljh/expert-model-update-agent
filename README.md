# 专家模型增量更新智能体

本项目旨在构建一个能够为现有专家模型制定增量更新策略、适配具体项目代码、实际训练并验证结果的智能体系统。

一次完整任务最终应交付：

- 更新后的模型权重 \(f_{\theta'}\)；
- 原模型与新模型的效果对比；
- 可追溯的代码、配置、训练和评估记录；
- 与新模型对应的描述档案。

## 当前阶段

项目处于最小产品原型阶段，前置流程第一版已经从 TaskInput 受理贯通到正式模型结构报告。现有能力包括确定性校验、资产入口预检、声明模型配置与 checkpoint 的安全证据检查、模型项目理解、模型创建方案静态验证、本地 Python 子进程实例化、checkpoint 实际加载和模型实例结构解析。结构解析与实例化在同一 worker 中完成，直接检查内存中的模型，不执行前向推理；完整 JSON 报告和逐层展开的结构树由宿主保存到 `run-records/model-structure/`。

公开验证项目 `pytorch_resnet_cifar10` 已在固定源码版本上完成一次成功运行：声明 checkpoint 在 `model-instantiation-safe` 环境中实际加载且没有 missing/unexpected keys，随后生成结构报告。这只证明当前公开验证案例成立，不代表已经适配其他项目或完成真实业务项目验收。当前本地执行没有操作系统级隔离，不能保证目标项目只读、阻断网络或没有其他副作用。下一阶段是更新决策和版本化 `UpdatePlan` 的设计；项目级改造、训练、评估和独立审阅尚未实现。根目录 `skills/` 也尚未通过项目级 Pi 配置加入自动发现范围。

首个增量学习方法包 `incremental_ridge_closed_form` v0.1 已建立。它固定实现平方误差总和加 L2 正则的解析增量岭回归，要求旧充分统计量与当前参数一致，并提供状态检查和有界更新工具。该方法包不自动发现项目、读取任意数据路径、修改代码或保存 checkpoint；具体项目适配仍属于后续更新执行 Agent。它也不表示更新决策 Agent、`UpdatePlan` 或训练闭环已经实现。

第一版人工编排原型已决定采用 Pi 作为 Agent Harness，但这一选择不形成永久技术锁定。Pi SDK 自动编排、LangGraph、A2A 和 MCP 是否采用仍待后续基于实际需要评估。

## 文档入口

- Agent 和其他自动化执行者从 [`AGENTS.md`](AGENTS.md) 开始。
- 系统主流程、Agent 分工和工具边界见 [`docs/architecture.md`](docs/architecture.md)。
- 最小产品、四类交接契约、任务状态和批准节点见 [`docs/product-specs/minimal-model-update-agent.md`](docs/product-specs/minimal-model-update-agent.md)。
- 任务受理流程与 `TaskInput` 字段见 [`docs/product-specs/task-intake.md`](docs/product-specs/task-intake.md)。
- 模型创建方案静态验证契约见 [`docs/product-specs/model-creation-validation.md`](docs/product-specs/model-creation-validation.md)。
- 模型实例化契约与本地执行边界见 [`docs/product-specs/safe-model-instantiation.md`](docs/product-specs/safe-model-instantiation.md)。
- 新实例化环境的创建命令和依赖文件见 [`environments/README.md`](environments/README.md)；旧 `model-analysis` 环境保留不动。
- 模型配置与 checkpoint 安全证据检查见 [`docs/product-specs/model-asset-evidence-inspection.md`](docs/product-specs/model-asset-evidence-inspection.md)。
- 首个方法包的算法、状态、工具和执行边界见 [`docs/product-specs/incremental-ridge-method-package.md`](docs/product-specs/incremental-ridge-method-package.md)。
- 任务受理建设进度见 [`docs/exec-plans/active/task-intake-v0.md`](docs/exec-plans/active/task-intake-v0.md)。
- 模型项目理解建设进度见 [`docs/exec-plans/active/model-project-understanding-skill-v0.md`](docs/exec-plans/active/model-project-understanding-skill-v0.md)。
- 模型创建、实例化与结构解析建设进度见 [`docs/exec-plans/active/model-creation-static-validator-v0.md`](docs/exec-plans/active/model-creation-static-validator-v0.md)。
- 解析增量岭回归方法包建设进度见 [`docs/exec-plans/active/incremental-ridge-method-package-v0.md`](docs/exec-plans/active/incremental-ridge-method-package-v0.md)。
- Pi 原型 Harness 决定见 [`docs/design-docs/decisions/0001-pi-prototype-harness.md`](docs/design-docs/decisions/0001-pi-prototype-harness.md)。
- TaskInput 校验工具的 Pi Extension 接入决定见 [`docs/design-docs/decisions/0002-task-input-validator-pi-extension.md`](docs/design-docs/decisions/0002-task-input-validator-pi-extension.md)。
- 资产入口预检的报告与工具边界见 [`docs/design-docs/decisions/0003-task-asset-preflight.md`](docs/design-docs/decisions/0003-task-asset-preflight.md)。
- 人类声明模型配置与安全证据检查决定见 [`docs/design-docs/decisions/0004-human-declared-model-config-evidence.md`](docs/design-docs/decisions/0004-human-declared-model-config-evidence.md)。
- 已确认的 Harness Engineering 原则见 [`docs/design-docs/harness-engineering-principles.md`](docs/design-docs/harness-engineering-principles.md)。

仓库目录将根据真实内容逐步扩展，不预先复制完整模板或创建无内容的文档。

当前已有第一版实现的能力：

- [`skills/task-intake/SKILL.md`](skills/task-intake/SKILL.md)：把人类模型更新请求整理成待确认的 `TaskInput` 草案；
- [`skills/model-project-understanding/SKILL.md`](skills/model-project-understanding/SKILL.md)：在 `TaskInput` 确认后只读调查目标项目，形成带证据的模型候选和创建方案；
- [`schemas/task-input.schema.json`](schemas/task-input.schema.json)：`TaskInput` v0.2 的机器结构；
- [`schemas/asset-preflight-report.schema.json`](schemas/asset-preflight-report.schema.json)：只读资产入口预检结果的机器结构；
- [`schemas/config-inspection-report.schema.json`](schemas/config-inspection-report.schema.json)：人类声明模型配置的安全解析结果；
- [`schemas/model-creation-proposal.schema.json`](schemas/model-creation-proposal.schema.json)：项目理解之后提交给静态工具的最小创建方案；
- [`schemas/model-creation-validation-report.schema.json`](schemas/model-creation-validation-report.schema.json)：创建方案静态验证结果；
- [`schemas/model-instantiation-report.schema.json`](schemas/model-instantiation-report.schema.json)：模型实例化与实际权重加载结果；
- [`schemas/model-structure-analysis-report.schema.json`](schemas/model-structure-analysis-report.schema.json)：同一次 worker 执行得到的模型结构事实；
- [`method-packages/incremental-ridge-closed-form/method-card.json`](method-packages/incremental-ridge-closed-form/method-card.json)：首个方法卡，固定解析增量岭回归的适用条件与执行边界；
- [`schemas/incremental-ridge-state.schema.json`](schemas/incremental-ridge-state.schema.json)：解析增量岭回归的充分统计量与参数状态；
- [`schemas/incremental-ridge-update-request.schema.json`](schemas/incremental-ridge-update-request.schema.json)：与任务和计划摘要绑定的有界更新请求；
- [`.pi/extensions/task-input-validator/index.ts`](.pi/extensions/task-input-validator/index.ts)：向 Pi 注册 `validate_task_input` 和 `preflight_task_assets`，并调用各自独立的 Python 逻辑。
- [`.pi/extensions/model-creation-validator/index.ts`](.pi/extensions/model-creation-validator/index.ts)：向 Pi 注册 `validate_model_creation_spec` 并调用独立 Python 验证逻辑。
- [`.pi/extensions/model-instantiation/index.ts`](.pi/extensions/model-instantiation/index.ts)：向 Pi 注册 `instantiate_validated_model` 和 `analyze_validated_model_structure`；后者在同一次本地执行中完成实例化与结构解析，并由宿主保存完整报告和展开树到 `run-records/model-structure/`。
- [`.pi/extensions/model-evidence-inspectors/index.ts`](.pi/extensions/model-evidence-inspectors/index.ts)：向 Pi 注册 `inspect_model_config` 和 `inspect_model_checkpoint`，只读取已确认并通过预检的声明资产。
- [`.pi/extensions/incremental-ridge/index.ts`](.pi/extensions/incremental-ridge/index.ts)：向 Pi 注册 `inspect_incremental_ridge_state` 和 `update_incremental_ridge`，只处理标准化、小规模的内联矩阵与状态。

开发测试、参考答案和评估记录不属于 Pi 运行上下文，在工作区之外维护；正式文档和 Skill 不得引用这些材料。
