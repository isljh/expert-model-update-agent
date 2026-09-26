# 设计决定 0003：资产入口预检使用独立报告和 Pi 工具

## 状态

**已接受，适用于第一版 Pi CLI 原型。**

## 背景

`TaskInput` 保存人类声明的目标项目、原模型 checkpoint、可选模型配置文件和新数据位置。路径字符串满足契约并不能证明当前 Agent 运行环境能够访问这些资产。如果直到项目理解或模型实例化阶段才发现路径不存在，失败位置过晚且错误语义不清。

路径可用状态又是可变化的运行环境事实，不应写入需要人工确认和版本化的业务输入。临时挂载或权限变化不应自动产生新的 TaskInput 版本。

## 决定

1. 在人工最终确认前执行一次只读资产入口预检；`confirmed` 校验通过后、进入项目理解前再次执行。
2. 预检结果使用独立 `AssetPreflightReport`，不向 `TaskInput.assets` 添加状态字段。
3. 报告通过 `task_id`、`task_input_version` 和 `asset_input_digest` 绑定具体资产声明；摘要只覆盖任务身份、版本和 `assets`，不包含人工确认状态。
4. 在现有项目级 Pi Extension 中注册独立工具 `preflight_task_assets`，不为 `validate_task_input` 增加文件系统模式。
5. 核心检查实现在与 Pi 解耦的 Python 模块中，Extension 继续只承担输入限制、进程调用、超时和结果适配。
6. 预检只检查路径存在性、预期类型和当前可读性，不递归扫描、不读取内容、不导入代码、不加载 checkpoint，也不修改资产。checkpoint 入口允许是普通文件或目录，以兼容单文件、分片和目录式保存格式；其内部格式由项目理解阶段按需调用的专用 Checkpoint Inspector 判断。人类明确提供的可选模型配置必须是普通可读文件；未提供时该项记为 `skipped`，系统不自动搜索替代配置。
7. 预检失败时不得请求最终确认；进入项目理解前的复核失败时不得继续交接。

## 理由

- 复用 Python 标准库即可完成确定性检查，不引入新的第三方依赖或服务。
- 独立报告避免运行环境变化污染或反复版本化 TaskInput。
- 独立工具让 TaskInput 结构校验保持环境无关，同时给 Harness 提供可机械判断的阶段门。
- 在确认前检查可以尽早要求用户修正错误路径；进入下一阶段前复核可以发现确认后的挂载、权限或文件变化。

## 限制与后续工作

- 第一版相对路径按 Extension 的项目工作目录解析，并在报告中记录 `base_directory` 和规范路径；跨机器运行应优先使用明确的绝对路径或另行设计位置映射。
- `os.access` 在不同平台语义不一致，因此实现通过实际打开目录句柄或只读文件句柄检查可读性，但不读取内容。
- 当前 Pi Harness 尚未实现阶段级工具白名单和不可绕过的状态机；Skill 和工具契约已经建立，机械门控仍需后续实现。
- 配置内容解析、checkpoint 内容安全、哈希和张量结构不属于本预检，由项目理解阶段的专用只读检查器负责；源码语义与具体 proposal 由项目理解 Agent 和薄静态验证器分别负责。

## 关联材料

- [任务受理规格](../../product-specs/task-intake.md)
- [资产预检报告 Schema](../../../schemas/asset-preflight-report.schema.json)
- [TaskInput 校验 Extension 决定](0002-task-input-validator-pi-extension.md)
