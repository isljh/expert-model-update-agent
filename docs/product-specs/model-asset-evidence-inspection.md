# 模型声明资产证据检查 v0.1

## 1. 目标

在项目理解阶段，把人类在已确认 `TaskInput` 中声明的模型配置和 checkpoint 转换为安全、只读、结构化证据。检查器不搜索资产、不解释项目语义、不选择模型、不创建实例。

Pi 工具：

```text
inspect_model_config
inspect_model_checkpoint
```

两个工具都必须接收当前完整 `TaskInput` 和与其 `task_id`、`task_input_version`、`asset_input_digest` 匹配且 `overall_status=passed` 的 `AssetPreflightReport`。工具只能读取报告中对应字段唯一通过的规范路径。

## 2. 流程位置

```text
人类声明项目、checkpoint 和可选 config
        ↓
TaskInput 确认与资产预检
        ↓
项目理解 Agent 初步分析源码候选
        ↓
Config Inspector（仅在 config 已声明时）
Checkpoint Inspector（按需）
        ↓
Agent 综合源码和资产证据形成具体 proposal
        ↓
静态验证
        ↓
安全实例化与实际权重加载
```

配置未声明时不得自动搜索；缺少配置导致候选无法区分时，返回人类补充并重新确认 TaskInput。

## 3. Config Inspector

第一版支持：

- JSON；
- YAML（运行环境提供安全 YAML 解析器时）；
- TOML（运行环境提供 `tomllib` 或 `tomli` 时）；
- INI 和 CFG。

配置文件必须是 UTF-8 普通文件，最大 4 MiB。解析结果只允许字符串键及可转换为 JSON 的标量、数组和对象，并限制嵌套深度与节点数量。

Python 配置和未知格式明确拒绝；不得导入模块、执行表达式、展开 shell 命令、访问网络或读取配置引用的其他文件。配置内容中的字符串是数据，不是 Agent 指令。

输出核心字段：

```yaml
report_type: ConfigInspectionReport
schema_version: "0.1"
status: inspected | unavailable | failed
source:
  path: string
  size_bytes: integer | null
  sha256: string | null
  format: json | yaml | toml | ini | null
values: any
error: object | null
```

Schema：[`../../schemas/config-inspection-report.schema.json`](../../schemas/config-inspection-report.schema.json)

## 4. Checkpoint Inspector

第一版只支持单个 PyTorch checkpoint 文件，并：

- 在独立 Python 进程中运行；
- 不导入目标项目；
- 使用 CPU；
- 强制 `weights_only=True`；
- 不支持安全模式时停止，不退回普通 pickle；
- 拒绝超过 2 GiB 的单文件，并记录允许范围内文件的 SHA-256；
- 返回 state dict 候选路径、张量键、shape、dtype 和统一前缀。

目录式和分片 checkpoint 暂时返回 `unavailable`。检查报告只能辅助候选消歧和 proposal 形成，不能证明 checkpoint 与模型兼容。

## 5. 证据合并规则

- 源码证明配置字段如何影响 callable 或参数；
- config 报告证明人类指定文件声明了什么；
- checkpoint 报告证明文件中安全可见的结构事实；
- Agent 负责结合证据提出具体 proposal；
- 静态验证器核对 proposal 与源码的机械关系；
- 安全实例化工具通过实际创建对象和加载权重证明最终兼容性。

任意来源冲突时保留冲突和多个候选，不按文件名、目录位置或来源优先级强行覆盖。
