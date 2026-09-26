# 任务受理规格

## 文档状态

- **状态：** v0.2 实现基线
- **适用阶段：** 人类提出模型更新请求后、更新决策 Agent 开始模型与数据分析前
- **权威结构：** [TaskInput Schema](../../schemas/task-input.schema.json)

## 1. 目标

任务受理把用户的自然语言请求整理为最小、可校验、待人工确认的 `TaskInput`。必答问题只能来自固定最小字段的缺失；Agent 认为有帮助的其他问题必须是可跳过的选答问题。

任务受理不分析项目、模型或数据，不选择增量学习方法，也不制定 `UpdatePlan`。这些工作在 `TaskInput` 确认后由更新决策 Agent 调用只读工具完成。

## 2. 产物和事实来源

正式产物只有：

1. 版本化 `TaskInput` JSON；
2. 与具体 `TaskInput` 版本和资产声明绑定的 `AssetPreflightReport`；
3. 面向用户的中文确认摘要；
4. Schema 与语义校验结果。

中文摘要是 `TaskInput` 和当前资产预检结果的展示视图，不是第二份事实来源。用户原文必须保留；Agent 可以整理目标，但不得把推断标为已确认事实。资产预检报告是可重新生成的运行证据，不写回 `TaskInput`。

开发测试、参考答案、历史测试对话和评估材料由构建者在 Pi 工作区之外保存，正式 Skill 与运行规范不得引用或读取。

## 3. TaskInput v0.2 最小字段

| 分组 | 字段 | 含义 |
| --- | --- | --- |
| 标识 | `schema_version` | 结构版本，v0.2 固定为 `0.2` |
| 标识 | `task_id` | 一次模型更新任务的稳定标识 |
| 标识 | `task_input_version` | 同一任务输入的修订版本，从 1 开始递增 |
| `request` | `raw_prompt` | 用户原始表述，原样保留 |
| `request` | `update_goal` | 整理出的更新目标、原因及确认状态 |
| `assets.target_project` | `location` | 目标项目的位置 |
| `assets.base_model` | `checkpoint` | 原模型 checkpoint 的位置；可暂缺 |
| `assets.base_model` | `config` | 人类明确提供的可选模型配置文件；未提供时为 `null` 或省略，不自动搜索 |
| `assets.base_model` | `discovery_authorized` | checkpoint 缺失时，是否授权后续只读发现 |
| `assets.new_data` | `location` | 新数据的位置 |
| `assets.new_data` | `allowed_uses` | 明确允许的用途：分析、训练、验证、评估 |
| `requirements` | `capabilities_to_preserve` | 必须保留的能力，或明确表示无需保留 |
| `requirements` | `success_expectation` | 用户对更新成功的业务期望 |
| `intake` | `required_questions` | 由最小字段缺失机械决定的必答问题 |
| `intake` | `optional_questions` | Agent 提出的可选问题，最多 3 个，可跳过 |
| `intake` | `human_confirmation` | 人工确认状态及其绑定的输入版本 |

### 3.1 更新目标

`update_goal` 包含：

- `summary`：希望模型获得或改善的能力；
- `reason`：为什么需要更新；
- `status`：`missing`、`proposed` 或 `confirmed`。

从原始 Prompt 整理出的目标默认是 `proposed`；人类明确确认后才是 `confirmed`。

### 3.2 原模型定位

`checkpoint` 与 `discovery_authorized` 构成一个替代条件：

- 用户可以直接给出 checkpoint；或
- 在 checkpoint 未知时，授权后续阶段在目标项目和允许范围内只读发现。

任务受理本身不得执行发现。原模型只读、不得覆盖是全局系统不变量，不放入 `TaskInput`，也不作为待确认假设。

`config` 只接收人类明确提供的、与原模型或 checkpoint 关联的配置文件。它不是最小必答字段；没有提供时，任务受理和项目理解阶段都不得自动搜索替代配置。后续若因缺少配置无法区分候选，再请求人类补充并按正常 TaskInput 变更重新确认。

### 3.3 新数据用途

`allowed_uses` 只能包含：

- `analysis`；
- `training`；
- `validation`；
- `evaluation`。

未明确授权的用途不得默认加入。

### 3.4 保留能力与成功期望

`capabilities_to_preserve` 使用三种模式：

- `unconfirmed`：尚未回答；
- `specified`：`items` 中列出了要保留的能力；
- `none`：人类明确表示没有需要保留的能力，此时 `items` 为空。

`success_expectation` 是业务层面的自然语言期望。具体技术指标、阈值与允许退化幅度可以在后续分析和计划阶段提出，不是任务受理的必答内容。

## 4. 必答问题的固定判定

必答问题不得由 Agent 自由扩张，只能按下表生成：

| 缺失条件 | 必须询问的内容 | `target_fields` |
| --- | --- | --- |
| `raw_prompt` 为空 | 请描述本次模型更新请求 | `request.raw_prompt` |
| 更新目标缺失 | 希望模型获得或改善什么能力 | `request.update_goal` |
| 目标项目位置为空 | 目标项目位于哪里 | `assets.target_project.location` |
| checkpoint 为空且未授权发现 | 提供 checkpoint，或授权后续只读发现 | `assets.base_model.checkpoint`、`assets.base_model.discovery_authorized` |
| 新数据位置为空 | 新数据位于哪里 | `assets.new_data.location` |
| 新数据用途为空 | 数据可用于分析、训练、验证、评估中的哪些用途 | `assets.new_data.allowed_uses` |
| 保留能力未确认 | 列出需要保留的能力，或明确回答“无” | `requirements.capabilities_to_preserve` |
| 成功期望为空 | 业务上怎样算本次更新成功 | `requirements.success_expectation` |

必答问题必须包含唯一 `id`、准确的 `target_fields`、中文问题、回答状态和答案。用户回答后，答案必须同步回相应正式字段，问题不能代替正式字段。

## 5. 可选问题

Agent 可以提出最多 3 个可选问题，但每题必须：

- 对后续分析或计划确有帮助；
- 写明 `reason`；
- 标明 `unanswered`、`answered` 或 `skipped`；
- 允许用户跳过；
- 不得阻止 `TaskInput` 确认。

能够由工具发现的事实不得询问用户，包括框架、代码提交、模型结构、参数量、训练入口、损失函数、数据格式、数据分布和漂移程度。

## 6. 草案与确认门槛

### 草案模式

草案允许最小字段的业务值暂缺，但必须：

- 符合 v0.2 结构；
- 每个最小缺口都有职责匹配的必答问题；
- 必答问题没有指向最小字段之外的内容；
- 可选问题不超过 3 个；
- 已回答的问题具有非空答案；
- 不通过默认权限或猜测补全字段。

### 确认模式

进入策略规划前必须同时满足：

- 所有最小字段有效，checkpoint 替代条件成立；
- 更新目标状态为 `confirmed`；
- 所有必答问题已回答且答案已同步到正式字段；
- 保留能力模式与列表一致；
- 人工确认状态为 `confirmed`；
- `confirmed_task_input_version` 与当前 `task_input_version` 一致。

可选问题可以保持 `unanswered` 或 `skipped`，不影响确认。任何内容变化都应递增 `task_input_version`，并把人工确认恢复为 `pending`。

### 资产入口预检

`draft` 校验通过、所有必答问题均已回答并同步到正式字段后，调用 `preflight_task_assets`，在请求人类最终确认前检查：

- `assets.target_project.location` 存在、是目录且当前可读；
- 明确提供的 `assets.base_model.checkpoint` 存在、是普通文件或目录且当前可读；具体 checkpoint 格式由项目理解阶段按需调用的专用 Checkpoint Inspector 判断；
- checkpoint 未提供但 `discovery_authorized` 为 `true` 时，该项记为 `skipped`，不阻塞预检；
- 明确提供的 `assets.base_model.config` 必须存在、是普通文件且当前可读；未提供时记为 `skipped`，不阻塞预检；
- `assets.new_data.location` 存在、是普通文件或目录且当前可读。

预检不递归扫描目录，不读取项目源码、配置内容、数据内容或 checkpoint 内容，不导入模块，也不修改任何资产。报告结构见 [`../../schemas/asset-preflight-report.schema.json`](../../schemas/asset-preflight-report.schema.json)。

报告通过 `task_id`、`task_input_version` 和只覆盖资产声明的 `asset_input_digest` 绑定 TaskInput。人工确认字段变化不会使报告与资产声明失配；资产位置或其他 `assets` 内容变化会使旧报告失效。

预检失败时不得请求最终确认。用户修改资产声明时按普通 TaskInput 修改递增版本并重新校验；仅修复挂载或权限而未改变 TaskInput 时，重新运行预检即可。

`confirmed` 校验通过后、进入项目理解阶段前必须重新运行预检。只有最新报告为 `passed` 且与当前 TaskInput 资产声明匹配，才能交接。`validate_task_input(mode="confirmed")` 只验证 TaskInput 契约本身，不能替代这项运行环境检查。

## 7. 中文确认摘要

摘要固定按以下顺序展示：

1. 已理解的信息；
2. 资产入口预检；
3. 必须补充；
4. 可选补充（可跳过）；
5. 当前是否可以确认。

摘要必须与 JSON 一致，并明确是否已机械校验。若 Harness 未注册校验工具，只能报告“尚未机械校验”，不得通过阅读校验器实现模拟结果。

## 8. Skill、校验工具与 Harness 边界

- [task-intake Skill](../../skills/task-intake/SKILL.md) 负责从原文整理字段和生成问题。
- [TaskInput 校验 Extension](../../.pi/extensions/task-input-validator/index.ts) 向 Pi 注册 `validate_task_input` 和 `preflight_task_assets`；前者检查结构、固定缺口映射和确认门槛，后者只检查声明资产的存在性、类型和可读性。两者都不推断用户意图。
- 人类明确确认当前版本后，Agent 将确认状态绑定到该 `task_input_version`，再调用 `validate_task_input` 的 `confirmed` 模式；当前流程不需要单独的确认记录工具。
- 任务受理阶段只允许额外调用指定的资产预检工具，不得调用通用命令行、目录扫描、项目分析、编辑或写入工具。
- Skill 中的禁止条款是行为规则；要机械阻止工具调用，仍需在 Pi Harness 或 Extension 中配置阶段级工具权限。该限制尚未实现。
- 当前 Extension 只注册确定性校验与预检工具；未来如需由 Extension 收集输入、管理版本或展示确认界面，应另行设计，且不得承担语义决策。
