# 设计决定 0002：TaskInput 校验器通过项目级 Pi Extension 注册

## 状态

**已接受，适用于第一版 Pi CLI 原型。**

## 背景

`task-intake` Skill 已能把用户请求整理为 TaskInput v0.2，Python 校验器也能确定性检查结构、固定必答问题映射和人工确认门槛。但仅存在 `validate.py` 不会让 Pi 自动把它作为模型可调用工具；Harness 必须获得工具名称、参数 Schema、执行函数和返回格式。

本项目仍要求业务契约和核心校验逻辑不依赖 Pi，以便未来替换 Harness。当前校验器只在本地项目中使用，不需要独立部署，也没有多客户端共享需求。

## 决定

1. 在 `.pi/extensions/task-input-validator/index.ts` 中使用 `pi.registerTool()` 注册 `validate_task_input`。同一 Extension 后续增加的独立资产预检工具见[设计决定 0003](0003-task-asset-preflight.md)，不改变本工具的环境无关职责。
2. Extension 只承担 Pi 工具适配：声明工具参数、调用校验器、限制输入输出大小与运行时间，并把结构化结果返回给 Agent。
3. 核心规则继续保存在 `src/expert_model_update/tools/task_input/validate.py`；不在 TypeScript 中复制校验逻辑。
4. Python 校验器通过标准输入接收 TaskInput JSON，通过标准输出返回机器可读 JSON。Extension 使用非 Shell 子进程调用，不创建临时 TaskInput 文件。
5. 校验失败属于正常工具结果，返回 `valid: false` 和问题列表；启动失败、超时或返回格式损坏才作为工具执行错误抛出。
6. 当前不为该本地校验器引入 MCP。若未来需要跨 Harness、跨进程或跨机器共享，再单独评估 MCP Server。

## 工具契约

输入：

```json
{
  "task_input": {},
  "mode": "draft"
}
```

`mode` 只能是：

- `draft`：检查结构、最小字段缺口与必答问题映射；
- `confirmed`：在草案检查基础上检查人工确认与进入策略规划的门槛。

输出至少包含：

```json
{
  "valid": false,
  "schema_version": "0.2",
  "mode": "draft",
  "task_id": null,
  "task_input_version": null,
  "issues": []
}
```

## 理由

- 项目级 Extension 能由 Pi 自动发现，适合当前人工 CLI 原型。
- 薄适配保留 Python 校验器的独立性，避免两套规则漂移。
- 直接注册比为单个本地确定性函数建立 MCP Server 更简单。
- JSON 标准输入/输出便于独立测试，也避免文件路径、临时文件和 Shell 命令拼接。

## 限制与后续工作

- 运行环境必须提供可执行的 Python；默认 Windows 使用 `python`，其他平台使用 `python3`，可通过 `TASK_INPUT_PYTHON` 指定。
- Extension 注册校验工具不等于阶段级权限隔离。任务受理阶段仍需进一步限制目录扫描、命令执行和文件修改工具。
- 后续若改变 TaskInput Schema 版本，必须同步更新 Schema、Python 校验器、工具说明和自动化测试。
- 资产存在性、类型和可读性不属于 `validate_task_input`；由独立的 `preflight_task_assets` 工具处理。

## 关联材料

- [任务受理规格](../../product-specs/task-intake.md)
- [Pi Harness 原型决定](0001-pi-prototype-harness.md)
- [Pi Agent Loop](../../references/pi-agent-loop.md)
