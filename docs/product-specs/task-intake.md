# 任务受理规格

## 文档状态

- 状态：v0.3 实现基线
- 适用阶段：人类提出模型更新请求后、模型交付检查和更新决策开始前
- 权威结构：[TaskInput Schema](../../schemas/task-input.schema.json)

## 1. 目标

任务受理把用户自然语言请求整理为最小、可校验、待人工确认的 TaskInput。必答问题只能来自固定最小字段的缺失；其他补充信息必须保持可跳过。

本阶段不读取模型、配置、checkpoint 或数据内容，不导入模型，不选择增量学习方法，也不制定 UpdatePlan。

## 2. 上游交付前提

系统不再假设上游提供原训练项目或训练过程。上游必须明确交付：

1. 模型文件：包含唯一声明的 PyTorch 模型类；
2. 构造配置：给出模型类名、完整构造关键字参数和 checkpoint 加载约定；
3. checkpoint：与该模型结构对应的权重文件。

三项缺一不可。系统不在项目目录中搜索替代文件，也不依靠权重形状猜测完整构造参数。

构造配置的机器格式见 [Model Load Config Schema](../../schemas/model-load-config.schema.json)。

## 3. TaskInput v0.3 最小字段

| 分组 | 字段 | 含义 |
| --- | --- | --- |
| 标识 | schema_version | 固定为 0.3 |
| 标识 | task_id | 一次更新任务的稳定标识 |
| 标识 | task_input_version | 同一任务输入的修订版本 |
| request | raw_prompt | 用户原始表述，原样保留 |
| request | update_goal | 更新目标、原因和确认状态 |
| assets.base_model | model_file | 上游交付的模型类文件 |
| assets.base_model | config | 与模型类和 checkpoint 对应的构造配置 |
| assets.base_model | checkpoint | 原模型权重文件 |
| assets.new_data | location | 新数据位置 |
| assets.new_data | allowed_uses | 明确允许的分析、训练、验证、评估用途 |
| requirements | capabilities_to_preserve | 必须保留的能力，或明确为无 |
| requirements | success_expectation | 业务层面的成功期望 |
| intake | required_questions | 固定最小字段缺失产生的必答问题 |
| intake | optional_questions | 最多 3 个可跳过问题 |
| intake | human_confirmation | 人工确认状态和绑定版本 |

## 4. 必答问题固定映射

| 缺失条件 | 必须询问的内容 | target_fields |
| --- | --- | --- |
| 原始请求为空 | 描述本次模型更新请求 | request.raw_prompt |
| 更新目标缺失 | 希望模型获得或改善什么能力 | request.update_goal |
| 模型文件为空 | 提供包含原模型类的模型文件 | assets.base_model.model_file |
| 构造配置为空 | 提供与模型和权重对应的构造配置 | assets.base_model.config |
| checkpoint 为空 | 提供与声明模型结构匹配的 checkpoint | assets.base_model.checkpoint |
| 新数据位置为空 | 新数据位于哪里 | assets.new_data.location |
| 新数据用途为空 | 数据允许用于哪些用途 | assets.new_data.allowed_uses |
| 保留能力未确认 | 列出需要保留的能力或明确回答无 | requirements.capabilities_to_preserve |
| 成功期望为空 | 业务上怎样算更新成功 | requirements.success_expectation |

用户回答后必须把答案同步回正式字段；问题对象不能替代正式字段。

## 5. 草案、确认和版本

草案允许业务值暂缺，但每个缺口必须有映射准确的未回答问题。可选问题不超过 3 个，未回答不能阻止确认。

确认模式要求：

- 三个模型资产、新数据位置和用途均已明确；
- 更新目标状态为 confirmed；
- 所有必答问题已回答且已同步；
- 保留能力模式与列表一致；
- 人工确认状态为 confirmed；
- confirmed_task_input_version 等于当前版本。

内容变化时递增 task_input_version，并把人工确认恢复为 pending。

## 6. 资产入口预检

在请求最终确认前以及确认后进入模型交付检查前，分别调用 preflight_task_assets。预检只检查：

- assets.base_model.model_file 存在、是普通文件且可读；
- assets.base_model.config 存在、是普通文件且可读；
- assets.base_model.checkpoint 存在、是普通文件且可读；
- assets.new_data.location 存在、类型受支持且可读。

预检不读取内容、不导入代码、不加载权重、不扫描目录。报告通过 task_id、task_input_version 和 asset_input_digest 与当前资产声明绑定。

任何模型资产路径变化都要递增 TaskInput 版本、恢复 pending、重新确认并重新预检。

## 7. 确认后的固定顺序

```text
TaskInput confirmed
→ 再次资产预检
→ inspect_model_config
→ inspect_model_checkpoint
→ validate_model_delivery
→ load_validated_model 或 analyze_validated_model_structure
```

配置检查只把声明文件作为数据安全解析。checkpoint 检查只使用 weights_only 安全模式获取键、形状和候选 state_dict 路径。两项检查都不能证明模型可创建或权重兼容。

## 8. 中文确认摘要

摘要固定按以下顺序展示：

1. 已理解的信息；
2. 资产入口预检；
3. 必须补充；
4. 可选补充（可跳过）；
5. 当前是否可以确认。

工具不可用或执行失败时必须说明“尚未机械校验”，不得根据源码阅读模拟通过结果。

## 9. 边界

- 原模型文件、配置和 checkpoint 只读，不得覆盖。
- 不要求上游交付原训练循环、优化器、学习率或训练日志。
- 不把“没有原训练工程”解释成“可以改变核心任务目标”。
- 模型的核心任务、主损失含义和输入输出语义属于后续更新决策所需信息，不由权重形状推断。
