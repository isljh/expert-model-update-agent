---
name: task-intake
description: 将模型更新请求整理为最小、可校验、待人工确认的 TaskInput v0.3；模型文件、构造配置和 checkpoint 均为固定必需资产。只在最小字段缺失时提出必答问题，不进行模型加载、数据分析、方法选择、训练或评估。
---

# 模型更新任务受理

把用户请求整理为最小 `TaskInput` 草案，使人类能够用中文快速核对。人类确认当前版本前，不进入模型或数据分析及策略规划。

## 权威材料

- 字段语义与确认门槛：[`../../docs/product-specs/task-intake.md`](../../docs/product-specs/task-intake.md)
- 机器结构：[`../../schemas/task-input.schema.json`](../../schemas/task-input.schema.json)
- 资产预检报告：[`../../schemas/asset-preflight-report.schema.json`](../../schemas/asset-preflight-report.schema.json)

## 工作流程

1. 将用户原始表述原样写入 `request.raw_prompt`。
2. 提取更新目标及原因；未经人类确认时把 `request.update_goal.status` 设为 `proposed`，信息不足时设为 `missing`。
3. 分别记录上游交付的模型文件、构造配置和 checkpoint 到 `assets.base_model.model_file`、`assets.base_model.config`、`assets.base_model.checkpoint`。三项都是必需资产，不提供自动搜索或权重反推构造参数的替代路径。
4. 记录新数据位置及用户明确授权的用途；不默认允许训练、验证或评估。
5. 记录必须保留的能力；用户可以明确回答“无需要保留的能力”。
6. 记录业务层面的成功期望，不要求用户在受理阶段制定技术阈值。
7. 仅按正式规格中的固定映射生成 `required_questions`。每题必须指向对应 `target_fields`。
8. 可生成最多 3 个 `optional_questions`；说明原因并允许跳过。
9. Harness 已注册 `validate_task_input` 时，每次生成或修改草案后使用 `draft` 模式检查。工具不可用或失败时明确说明“尚未机械校验”。
10. `draft` 校验通过且所有必答答案已同步到正式字段后，调用 `preflight_task_assets`。预检失败时按具体路径问题修复，不请求最终确认。
11. 展示完整 TaskInput JSON 和中文摘要。摘要固定使用“已理解的信息 / 资产入口预检 / 必须补充 / 可选补充（可跳过） / 当前是否可以确认”五部分。
12. 人类明确确认当前版本后，把更新目标和人工确认状态设为 `confirmed`，并把 `confirmed_task_input_version` 设为当前版本，再调用 `validate_task_input` 的 `confirmed` 模式。内容变化时递增版本并恢复 `pending`。
13. `confirmed` 校验通过后、进入模型交付检查前，再次运行 `preflight_task_assets`。只有报告为 `passed` 且任务、版本和资产摘要匹配时才能交接。

## 边界

- 原模型资产只读、不得覆盖是系统不变量，不写入最小 TaskInput。
- 不询问能够由后续检查器获得的 checkpoint 键、张量形状和参数量。
- 不从权重形状猜测模型构造参数；构造参数必须由上游配置明确提供。
- 任务受理阶段不读取模型、配置、checkpoint 或数据内容，不导入模型文件，不选择增量学习方法，不制定 `UpdatePlan`，不训练或评估。

`TaskInput` 确认后退出任务受理阶段。后续依次进行配置检查、checkpoint 检查、模型交付验证和受控加载。
