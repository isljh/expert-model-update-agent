# 设计决定 0001：第一版原型采用 Pi Harness，但不形成永久锁定

## 状态

**已接受，适用于第一版人工编排原型。**

## 背景

第一版需要验证的核心问题不是复杂工作流框架能否运行，而是以下纵向闭环能否成立：决策 Agent 形成有证据的更新计划，执行 Agent 在真实项目中核对前提并完成训练与评估，审阅 Agent 独立核验代码和产物，人类在关键节点批准继续。

现有 Pi 实践已经验证了模型连接、Agent Loop、文件与命令工具、Session、Skill、Extension、检查点和独立审阅等基础能力。Pi 核心以单 Agent 会话为中心，不直接提供本项目所需的完整多 Agent 编排；这不妨碍用三个独立 Session 验证职责和交接契约。

## 决定

第一版原型使用同一次 Pi 安装建立三个独立 Session：

- 更新决策 Session；
- 更新执行 Session；
- 独立审阅 Session。

三个 Session 通过仓库或受管理任务目录中的版本化契约与证据交接，不依赖彼此的临时聊天记忆。原型阶段由人类创建任务、批准 `UpdatePlan`、批准正式训练并完成最终交付决定。

Pi 只承担 Harness 职责：连接模型与工具、组装上下文、运行 Agent Loop、保存 Session，并提供文件、命令、Skill、Extension 或 SDK 扩展点。任务状态、跨 Agent 交接、训练产物追踪和完成验收由本项目定义。

## 不锁定原则

Pi 不是永久实现承诺。业务契约、任务状态和证据模型不得直接依赖 Pi 私有 Session 结构、事件格式或 CLI 交互。后续可以在保持这些外部语义稳定的前提下替换 Harness 或增加编排层。

以下事项仍为待定：

- CLI 原型通过后是否使用 Pi SDK 自动创建和调度 `AgentSession`；
- 是否以及何时引入专门的工作流编排框架；
- 工具采用直接接口、Extension、MCP 或其他接入方式；
- Agent 是否需要部署为独立服务及是否采用 A2A。

## 理由

1. 复用已经掌握并验证的 Agent Loop 与 Session 能力，降低首个闭环的非业务复杂度。
2. 三个独立 Session 足以先验证角色隔离、交接契约、重规划和独立审阅。
3. 人工批准适合原型阶段高成本训练与不可逆外部影响的风险水平。
4. 将业务契约与 Harness 解耦，可以避免学习性原型演变成未经评估的永久架构。

## 后果

### 正面后果

- 可以较早用真实代码、训练命令、权重和指标检验系统设计。
- 每个角色拥有独立上下文和工具权限，便于发现职责泄漏。
- 交接依赖持久化材料，有利于未来自动编排和审计。

### 成本与限制

- 原型阶段需要人类切换 Session、传递批准结果并启动下一阶段。
- Pi 的单 Agent 循环不等于本项目状态机，必须另行记录任务状态。
- Session 记录不能替代正式任务记录、训练日志和产物清单。

## 重新评估触发条件

出现下列任一情况时，应新增设计决定，而不是静默扩大本决定：

- 人工切换 Session 成为稳定复现的主要瓶颈；
- 需要跨进程恢复、复杂并行分支或大量人工介入点；
- Pi 无法提供所需的权限隔离、可观测性或长任务控制；
- 需要将 Agent 暴露为独立服务；
- Harness 私有能力开始渗入业务契约。

## 关联材料

- [`../../references/pi-agent-loop.md`](../../references/pi-agent-loop.md)
- [`../harness-engineering-principles.md`](../harness-engineering-principles.md)
- [`../../product-specs/minimal-model-update-agent.md`](../../product-specs/minimal-model-update-agent.md)
