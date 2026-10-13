# 增量学习方法包目录

本目录保存可以被更新决策 Agent 选择、并由更新执行层调用的版本化增量学习方法。每个方法目录必须同时提供“面向人”和“面向智能体”的两种说明，但两者职责不同，不能互相代替。

## 文件职责

| 文件 | 主要读者 | 必须回答的问题 |
|---|---|---|
| `method-packages/<method>/README.md` | 研究人员、项目教师、开发者和审阅者 | 方法是什么、解决什么问题、数学目标、完整流程、项目如何实现、怎样调用、何时适用、限制和验证方式 |
| `method-packages/<method>/method-card.json` | 更新决策 Agent、更新执行 Agent | 方法标识与版本、硬前提、排除条件、所需状态、执行约束、重规划条件、组件入口和验证规则 |
| `docs/product-specs/<method>.md` | 负责维护和集成代码的开发者 | 方法怎样接入项目流程、工具边界、数据和报告契约、批准节点及验收要求 |
| `src/expert_model_update/methods/<method>/` | 执行环境 | 实际算法实现，不承担完整方法说明 |
| `tests/test_<method>.py` | 开发者和审阅者 | 实现是否满足方法契约和失败边界 |

## 使用规则

1. 人阅读某个方法时，从该方法的 `README.md` 开始。README 必须能够独立解释方法，不能只写一句话再让读者跳转到其他文档。
2. 智能体进行方法选择和执行检查时，以版本化的 `method-card.json` 为机器契约，不从自然语言 README 猜测硬前提。
3. 开发者修改算法、接口或状态格式时，同时更新实现规格、方法卡、README 和测试；其中任何一个未同步，都不能把新行为视为已正式定义。
4. README 中的示例用于帮助人理解；实际运行必须服从方法卡、Schema、已批准的 `UpdatePlan` 和当前版本代码。
5. 方法包只描述一种已经明确边界的方法，不以“兼容所有模型”为目标。超出适用条件时必须重新选择方法或新增方法版本。

## 当前方法

- [`incremental-ridge-closed-form`](incremental-ridge-closed-form/README.md)：固定特征提取器上的解析增量岭回归；
- [`derpp-replay`](derpp-replay/README.md)：使用历史样本、真实标签和历史输出的 DER++ 回放正则化；
- [`star-weight-perturbation`](star-weight-perturbation/README.md)：组合在分类回放方法上的 STAR 局部权重扰动稳定性组件。
