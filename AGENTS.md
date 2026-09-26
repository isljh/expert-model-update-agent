# 智能体项目入口

本仓库用于构建“专家模型增量更新智能体”。本文件只提供导航和全局边界，详细内容请按任务读取对应文档。

## 项目目标

系统接收现有模型、任务数据和目标项目资料，通过策略决策与项目级执行，实际完成模型训练和评估，交付更新后的模型权重、效果指标及可追溯记录。

## 阅读地图

- 项目定位、当前状态和文档入口：[`README.md`](README.md)
- Agent、工具、主流程与交付边界：[`docs/architecture.md`](docs/architecture.md)
- 最小产品、交接契约、状态与批准节点：[`docs/product-specs/minimal-model-update-agent.md`](docs/product-specs/minimal-model-update-agent.md)
- 任务受理与 `TaskInput`：[`docs/product-specs/task-intake.md`](docs/product-specs/task-intake.md)
- 模型创建方案静态验证：[`docs/product-specs/model-creation-validation.md`](docs/product-specs/model-creation-validation.md)
- 模型实例化、结构解析与正式报告：[`docs/product-specs/safe-model-instantiation.md`](docs/product-specs/safe-model-instantiation.md)
- 模型配置与 checkpoint 安全证据检查：[`docs/product-specs/model-asset-evidence-inspection.md`](docs/product-specs/model-asset-evidence-inspection.md)
- 首个增量学习方法包（解析增量岭回归）：[`docs/product-specs/incremental-ridge-method-package.md`](docs/product-specs/incremental-ridge-method-package.md)
- 任务受理建设计划：[`docs/exec-plans/active/task-intake-v0.md`](docs/exec-plans/active/task-intake-v0.md)
- 模型项目理解 Skill：[`skills/model-project-understanding/SKILL.md`](skills/model-project-understanding/SKILL.md)
- 模型项目理解 Skill 建设计划：[`docs/exec-plans/active/model-project-understanding-skill-v0.md`](docs/exec-plans/active/model-project-understanding-skill-v0.md)
- 模型创建方案静态验证建设计划：[`docs/exec-plans/active/model-creation-static-validator-v0.md`](docs/exec-plans/active/model-creation-static-validator-v0.md)
- 解析增量岭回归方法包建设计划：[`docs/exec-plans/active/incremental-ridge-method-package-v0.md`](docs/exec-plans/active/incremental-ridge-method-package-v0.md)
- 资产入口预检决定：[`docs/design-docs/decisions/0003-task-asset-preflight.md`](docs/design-docs/decisions/0003-task-asset-preflight.md)
- 人类声明模型配置与安全证据检查决定：[`docs/design-docs/decisions/0004-human-declared-model-config-evidence.md`](docs/design-docs/decisions/0004-human-declared-model-config-evidence.md)
- Harness Engineering 总原则：[`docs/design-docs/harness-engineering-principles.md`](docs/design-docs/harness-engineering-principles.md)
- Pi 的 Agent Loop 与项目映射：[`docs/references/pi-agent-loop.md`](docs/references/pi-agent-loop.md)

后续新增文档时，按用途放入相应目录，并从本文件或上一级索引建立链接。不要把专题知识堆入本文件。

## 已确定的全局边界

1. 人类定义目标、约束和验收标准；Agent 负责规划、实施、训练、评估与修复。
2. 更新决策 Agent 决定采用什么策略及其理由，不负责定位具体项目中的代码修改点。
3. 更新执行 Agent 阅读目标项目、实施更新、实际训练并评估；计划前提不成立时应反馈重规划。
4. 工具提供可信、可复用的分析或算法能力，不假设存在适配所有项目的一键更新规则。
5. 只有新权重可加载、评估完成并且证据齐全，才能声明模型更新完成。
6. 仓库中的文档、代码和运行记录是 Agent 的可信上下文；重要决定不能只保留在聊天中。
7. 可客观判断的重要要求应逐步转化为 Schema、lint、测试或交付检查。
8. 优先选择稳定、透明、可观察、便于 Agent 操作的实现方式；具体技术和协议尚未选定时不得写成既定事实。
9. Pi 工作区只保存正式运行内容；开发测试、参考答案和评估记录在工作区外维护，运行文档不得引用它们。

## 当前阶段

当前前置流程原型已经贯通到模型实例结构报告：TaskInput、资产预检、声明模型资产安全证据检查、模型项目理解、模型创建方案静态验证、本地子进程实例化、checkpoint 实际加载、模型实例结构解析和正式报告保存均已有第一版实现，并已在一个公开验证项目上完成成功运行。首个方法包 `incremental_ridge_closed_form` v0.1 已提供方法卡、充分统计量契约、Python 算法核心以及状态检查和有界更新工具；它只适用于固定平方误差加 L2 正则、特征与目标契约不变且旧充分统计量可用的仿射线性回归，不代表更新决策或具体项目适配已经完成。当前实例化没有操作系统级隔离，不能保证目标项目只读、网络阻断或无其他副作用；一次公开项目成功也不代表已经适配任意项目。下一阶段仍包括更新决策、版本化 `UpdatePlan` 与人工批准边界的设计；项目级改造、训练、评估和独立审阅尚未实现。根目录 `skills/` 尚未通过项目级 Pi 配置加入自动发现范围。尚未确定的技术选型、接口和流程细节应标记为待定，不得根据示意图自行补全。
