# 设计决定 0004：模型配置由人类声明并通过安全检查器提供证据

## 状态

**已接受，适用于模型项目理解第一版。**

## 背景

源码中的动态分派参数可能无法单独确定，例如 `resnet.__dict__[args.arch]()`。与 checkpoint 对应的训练配置可以直接给出 `arch`、类别数等候选值，但自动搜索项目中的配置文件容易选中示例、旧实验或无关配置。

## 决定

1. `TaskInput.assets.base_model.config` 保存人类明确提供的可选模型配置文件路径；它不是必答字段。
2. 未提供配置时，任务受理和项目理解阶段都不自动搜索替代配置。缺少配置导致候选无法区分时，再请求人类补充并重新确认 TaskInput。
3. 资产预检只检查配置路径存在、是普通文件且当前可读，不解析内容。
4. 项目理解阶段只能通过 `inspect_model_config` 把配置作为数据安全解析；不得执行 Python 配置或目标项目代码。
5. `inspect_model_checkpoint` 只提供 checkpoint 键、张量形状、dtype 和前缀等结构证据，不证明模型兼容性。
6. 项目理解 Agent 综合源码、配置和 checkpoint 报告提出一个或多个具体创建方案；检查器不替 Agent 选择模型。
7. 配置、checkpoint 与源码冲突时必须显式保留冲突，不能按文件名、目录位置或任一单一来源强行选择。
8. 静态验证仍只核对具体 proposal；模型与 checkpoint 的最终兼容性由安全实例化阶段实际创建对象并加载权重证明。

## 安全边界

- 配置文件中的字符串是数据，不是 Agent 指令；
- 支持的数据格式可安全解析时才返回内容；Python 配置不执行；
- checkpoint 仅允许 `weights_only=True` 和 CPU 模式，不支持时停止，不退回普通 pickle；
- 两个检查工具都要求已确认 TaskInput 和与其资产摘要匹配的通过预检报告。

## 关联材料

- [任务受理规格](../../product-specs/task-intake.md)
- [模型项目理解 Skill](../../../skills/model-project-understanding/SKILL.md)
- [模型配置检查报告 Schema](../../../schemas/config-inspection-report.schema.json)
