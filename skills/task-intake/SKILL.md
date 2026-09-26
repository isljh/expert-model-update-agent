---
name: task-intake
description: 将模型更新请求整理为最小、可校验、待人工确认的 TaskInput；仅在固定最小字段缺失时提出必答问题，并把其他补充问题保持为可跳过。用于任务刚创建时，不用于项目分析、方法选择、代码修改或训练。
---

# 模型更新任务受理

把用户请求整理为最小 `TaskInput` 草案，使人类能够用中文快速核对。人类确认当前版本前，不进入模型或数据分析及策略规划。

## 权威材料

- 字段语义与确认门槛：[`../../docs/product-specs/task-intake.md`](../../docs/product-specs/task-intake.md)
- 机器结构：[`../../schemas/task-input.schema.json`](../../schemas/task-input.schema.json)
- 资产预检报告结构：[`../../schemas/asset-preflight-report.schema.json`](../../schemas/asset-preflight-report.schema.json)

## 工作流程

1. 将用户原始表述原样写入 `request.raw_prompt`。
2. 提取更新目标及原因；未经人类明确确认时，将 `request.update_goal.status` 标为 `proposed`。没有足够信息时标为 `missing`。
3. 记录目标项目、新数据和原模型 checkpoint 的位置。checkpoint 未提供时，只询问“提供 checkpoint”或“授权后续只读发现”二选一。用户明确提供了与原模型或 checkpoint 对应的配置文件时，记录到 `assets.base_model.config`；未提供时使用 `null`，不得搜索或把配置文件升级为必答项。
4. 只记录用户明确授权的新数据用途，不默认数据可用于训练、验证或评估。
5. 记录必须保留的能力；用户也可以明确回答“无需要保留的能力”。
6. 记录业务层面的成功期望，不在任务受理阶段要求用户制定技术指标或阈值。
7. 仅按正式规格中的固定映射生成 `required_questions`。每个问题必须指向对应的 `target_fields`，不得因为模型认为“可能有帮助”而升级为必答问题。
8. 如确有助于后续工作的信息，可生成最多 3 个 `optional_questions`。每题说明提问原因，允许人类跳过，未回答不得阻止确认。
9. Harness 已注册 `validate_task_input` 时，生成或修改草案后用 `draft` 模式检查。修复结构问题时不得编造业务值。工具不可用或执行失败时明确说明“尚未机械校验”，不得通过阅读校验器源码模拟执行结果。
10. `draft` 校验通过、所有必答问题均已回答并同步到正式字段后，调用 `preflight_task_assets`。该工具失败或返回 `overall_status: failed` 时，不请求最终确认；按检查结果说明具体路径和修复动作。不得使用通用命令行或目录扫描代替该工具。
11. 展示完整的 TaskInput JSON 草案和中文确认摘要。摘要固定使用“已理解的信息 / 资产入口预检 / 必须补充 / 可选补充（可跳过） / 当前是否可以确认”五部分。资产预检结果是独立运行证据，不写入 TaskInput；未运行、工具执行失败或检查失败时必须如实说明。
12. 人类回答问题后，把答案同步到目标字段。人类明确确认当前版本后，将更新目标和人工确认状态设为 `confirmed`，并把 `confirmed_task_input_version` 设为当前 `task_input_version`，随后调用 `validate_task_input` 的 `confirmed` 模式。内容发生变化时递增版本并恢复 `pending`，重新确认。
13. `confirmed` 校验通过后、进入项目理解前，再次调用 `preflight_task_assets`。只有报告为 `passed`，且 `task_id`、`task_input_version` 和 `asset_input_digest` 与当前 TaskInput 资产声明匹配时，才能交接。重新检查只改变运行证据，不改变 TaskInput；若用户修改资产位置，则递增版本并重新确认。

## 边界

- “原模型只读、不得覆盖”是系统不变量，不是需要向用户确认的假设，也不写入最小 TaskInput。
- 不询问能够由后续工具发现的事实，包括框架、代码提交、模型结构、参数量、训练入口、损失函数、数据格式、数据分布和漂移程度。
- 任务受理阶段只允许额外调用指定的只读资产预检工具；不调用通用命令行、目录扫描或项目分析工具，不读取项目或配置内容，不修改或写入目标项目。
- 不读取开发测试、参考答案、历史测试对话或校验器实现源码。
- 不选择增量学习方法，不制定 `UpdatePlan`，不实施代码修改、训练或评估。

`TaskInput` 确认后退出任务受理阶段。更新决策 Agent 随后才能调用只读分析工具收集证据并制定 `UpdatePlan`。
