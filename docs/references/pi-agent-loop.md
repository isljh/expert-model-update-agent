# Pi Agent Loop 运行原理

本文记录项目采用 Pi 时必须保持一致的运行心智模型。主要参考：

- [Pi 蓝皮书：从一条 Prompt 到一次完整 Agent Loop](https://pi.xiaomovps.com/guide/how-pi-works)
- [Pi 官方 SDK 文档](https://pi.dev/docs/latest/sdk)
- [Pi 官方 Extensions 文档](https://pi.dev/docs/latest/extensions)
- [Pi 官方 Skills 文档](https://pi.dev/docs/latest/skills)
- [Pi 官方 Sessions 文档](https://pi.dev/docs/latest/sessions)
- [Pi 官方 Compaction 文档](https://pi.dev/docs/latest/compaction)

## 一句话模型

Pi 负责组织运行过程；模型判断下一步；工具接触真实环境；Session 保存过程；Context 是某一次模型调用真正收到的输入。

## 单次任务的真实循环

```text
启动并加载运行资源
  → 用户提交 Prompt，写入 Session
  → Pi 从活动 Session 分支、系统提示、项目规则、工具定义和 Skill 索引组装 Context
  → 调用模型
  → 模型直接回复，或返回 Tool Call
  → Pi 执行工具并得到 Tool Result
  → Tool Result 写入 Session，并进入下一次 Context
  → 再次调用模型
  → 重复工具循环，直到模型给出普通回复
  → Pi 保存最终回复并等待下一条消息
  → 真实环境中的产物由独立机制验收
```

一次用户 Prompt 可以触发多次模型调用。模型只提出 Tool Call；文件读写、命令执行和训练等真实动作由 Pi 工具或项目 Extension 完成。Tool Result 是下一次判断的证据，不是最终交付。

## 不能混为一谈的概念

- System Prompt、工具定义和 Skill 不是依次执行的流水线节点。它们共同参与 Context 组装；工具仅在模型请求时执行，Skill 的完整说明通常按需读取。
- Session 是磁盘上的过程记录；Context 是本次模型调用的输入快照。Session 仍保留旧内容，不代表所有旧内容每次都完整进入 Context。
- Compaction 只在上下文接近限额或被手动触发时发生。它以摘要替换部分早期上下文，存在信息损失，不恢复或回滚文件。
- Cache 是 Provider 对相同输入前缀的计算复用，影响费用和延迟，不保存 Session、不决定下一步，也不执行工具。
- Agent Loop 回到等待输入状态，只表示本轮停止继续调用工具，不代表业务结果正确。

## 对本项目的直接含义

1. 决策、执行与独立审阅 Agent 可以运行在各自的 AgentSession 中；三者通过版本化契约和证据交接，而不是依赖临时对话记忆。
2. 决策 Agent 的 Context 应包含角色边界、任务目标、方法约束、模型交付证据和允许使用的分析工具；不向它开放训练产物写入能力。
3. 执行 Agent 的 Context 应包含已验证的 `UpdatePlan`、只读模型交付物和实施约束；它在隔离工作区中创建本次更新代码、训练并评估。
4. 执行 Agent 发现计划前提不成立时，应提交结构化证据，由编排层返回决策 Agent 重规划。
5. Agent 的普通最终回复不能充当完成信号。编排层必须检查训练退出状态、新权重可加载性、评估指标、日志、配置、模型档案和原始产物保护情况。
6. 重要目标、决定、进度与验证结果必须写入仓库或受管理的产物记录；Session 和 Compaction 摘要都不是正式记录系统。
7. 审阅 Agent 的 Context 应包含任务、已批准计划、执行结果和原始证据的只读引用；审阅结论不能只依据执行 Agent 的自然语言总结。

## Pi 已提供与项目仍需提供的部分

| Pi 提供 | 本项目提供 |
| --- | --- |
| 模型连接与认证 | 决策 Agent、执行 Agent 的角色和边界 |
| 单个 Agent 的模型—工具循环 | 三个 Agent 的角色边界、跨 Agent 编排、重规划和任务状态机 |
| Session、Context、事件和压缩 | `UpdatePlan`、执行报告和验证报告契约 |
| 基础读写、编辑和命令工具 | 模型结构、训练、评估与产物管理能力 |
| Skill、Extension 和 SDK 扩展点 | 项目级 Skills、Extensions、lint、测试与 CI |
| 工作目录和资源发现 | 沙箱、训练资源控制和交付验收 |

CLI 适合学习、观察和人工原型验证。正式自动化可以通过 Pi SDK 创建和配置 AgentSession、限制工具集合、订阅事件并接入项目编排层；具体实现方式仍需在原型验证后确定。
