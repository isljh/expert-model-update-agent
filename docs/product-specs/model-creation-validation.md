# 模型创建方案静态验证

## 1. 目标

模型项目理解 Agent 已负责发现模型候选、理解对象角色并提出创建方案。静态验证工具不重新理解整个项目，也不验证整份自然语言调查报告；它只对 Agent 提交的 `ModelCreationProposalSet` 做定向、确定性的机械检查，并在条件满足时生成 `validated_instantiation_plan`。

第一版注册的 Pi 工具名称为：

```text
validate_model_creation_spec
```

输入 Schema：[`../../schemas/model-creation-proposal.schema.json`](../../schemas/model-creation-proposal.schema.json)

输出 Schema：[`../../schemas/model-creation-validation-report.schema.json`](../../schemas/model-creation-validation-report.schema.json)

## 2. 流程位置

```text
项目理解 Agent
  ├─ 面向人类和决策 Agent 的项目理解报告
  └─ 一个或多个 ModelCreationProposal
                    ↓
validate_model_creation_spec
                    ↓
ModelCreationValidationReport
  └─ validated_instantiation_plan（仅在就绪时存在）
                    ↓
安全实例化工具
```

项目理解报告中的候选理由、文档冲突、优化器关系和排除过程不属于本工具输入。`ModelCreationProposalSet` v0.2 只保留创建实例所需的完整对象创建链和用于核验这些字段的源码锚点。

## 3. 输入边界

顶层输入包含：

- `task_binding`：绑定已确认任务版本和资产声明摘要；
- `asset_preflight_report`：限制工具允许读取的目标项目与 checkpoint 范围；
- `project`：目标项目路径和可选 Git revision；
- `selected_proposal_id`：上游已经作出选择时使用；尚未选择时为 `null`；
- `proposals`：一个或多个具体创建方案。

每个方案包含：

- `construction_steps`：安全实例化工具必须按顺序执行的外部创建调用；
- 每一步的精确 callable 来源、模块、符号、位置参数、关键字参数和源码锚点；
- `object_ref`：后续步骤参数对前序步骤结果的显式引用；
- `result.runtime_object`：后续运行流程持有的对象；
- `result.target_model`：真正需要做结构分析的模型对象；
- checkpoint 加载前后的必要操作；
- checkpoint 文件、选择路径、键转换、对象加载目标和 `strict`；
- 每个创建调用及 checkpoint 加载调用的源码锚点。

参数值使用显式类型：

- `literal`：可 JSON 序列化的具体值；
- `symbol`：目标项目内的模块和符号引用。
- `object_ref`：已经完成的前序创建步骤所产生的对象，包含 `step_id` 和以 `$` 开头的对象路径。

例如源码：

```python
model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
```

不能压成一个 `resnet20` callable。proposal 至少包含基础模型步骤和 `DataParallel` 包装步骤；包装步骤的第一个实参引用基础步骤的 `$`，运行时对象引用包装步骤的 `$`，目标模型引用包装步骤的 `$.module`。

`construction_steps` 是安全实例化工具需要重放的外部调用图，不要求机械展开项目工厂内部已经封装的每个 Python 调用。工厂返回后发生的独立包装、容器组装或其他对象创建不能省略。

## 4. 工具检查内容

工具只围绕方案声明检查：

1. `TaskInput` 绑定与 `AssetPreflightReport` 是否一致；
2. 项目路径是否与资产预检授权路径一致；
3. 当前 Git revision 是否与方案声明一致；
4. `step_id` 是否唯一，对象引用是否只指向已存在的前序步骤，所有步骤是否都参与最终运行时对象的依赖链；
5. 项目 callable 模块、顶层类或函数是否存在；
6. 外部 PyTorch 包装 callable 是否位于明确允许列表；
7. 每一步参数是否能绑定对应 callable 签名；
8. 符号参数是否能在项目源码中定位；
9. 每一步创建调用锚点是否能绑定到对应 callable；
10. 调用参数表达式是否与方案一致；
11. `object_ref` 实参是否确实来自声明的前序步骤；
12. 支持范围内的常量、赋值链和导入符号是否能反向解析到方案值；
13. 运行时对象是否为源码直接赋值或返回的外层结果，目标模型路径是否属于声明步骤的可证明结果路径；
14. 生命周期函数调用和接收对象是否与声明的对象引用一致；
15. checkpoint 加载调用是否使用 proposal 声明的对象引用和选择路径；
16. checkpoint 加载指令中的路径是否绑定到资产预检授权范围；
17. selector、键转换和 `strict` 是否已经由 Agent 作为具体加载指令提供。

工具不会重新搜索其他模型、判断哪个模型承担业务角色或按源码顺序选择第一个方案。

## 5. 两种就绪状态

```yaml
readiness:
  ready_for_model_construction: boolean
  ready_for_checkpoint_load_attempt: boolean
  blockers: []
```

`ready_for_model_construction` 表示已选方案的所有创建步骤、步骤间对象依赖、源码绑定、运行时对象、目标模型和必要操作足以交给安全实例化工具。

`ready_for_checkpoint_load_attempt` 表示 Agent 提出的 checkpoint 路径、选择路径、键转换、加载目标和 `strict` 已通过契约、资产绑定、对象引用及源码锚点核验，可以交给安全实例化工具尝试。它不表示静态验证器打开或检查过 checkpoint，不表示权重已经成功加载，也不证明 checkpoint 参数与实例兼容。

未提供 checkpoint 时，前者可以为 `true`，后者必须为 `false`。

## 6. 多方案规则

- 没有验证通过的方案：不能实例化；
- 只有一个方案通过且未显式选择：自动形成唯一选择；
- 多个方案通过且 `selected_proposal_id=null`：报告 `ambiguous`，不能实例化；
- 上游显式选择的方案通过：可以形成实例化计划；
- 上游显式选择的方案未通过：不能退回其他方案，也不能按顺序选择。

工具验证“方案能否成立”，不决定“业务上应该选择哪个方案”。

## 7. checkpoint 加载指令边界

静态验证器只核验 proposal 中已经形成的 checkpoint 加载指令：

- 路径是否属于资产预检授权范围；
- `load_state_dict` 源码锚点是否存在；
- 加载接收对象是否与 `load_target` 对象引用一致；
- 可机械提取 selector 时，它是否与 proposal 一致；
- selector、键转换和 `strict` 是否符合输入 Schema。

静态验证器不打开 checkpoint，不计算其摘要，不运行 checkpoint 检查器，不发现 state dict，不检查张量形状或键前缀，也不判断键转换能否应用。checkpoint 的结构事实由项目理解阶段显式调用的 `inspect_model_checkpoint` 提供；Agent 结合这些事实形成具体 proposal；实际加载与兼容性验证属于安全实例化工具。

## 8. v0.2 已知限制

- 项目 callable 仅支持项目中的顶层类或函数；外部 callable 当前只允许 `torch.nn.DataParallel` 和 `torch.nn.parallel.DistributedDataParallel`；
- 创建锚点支持直接调用、导入别名、`module.__dict__[selector]()` 和 `getattr(module, selector)()`；
- 参数反向追溯当前支持字面量、同一作用域内的简单名称赋值和导入符号；复杂配置对象、跨函数变更和数据派生值会返回 `unresolved`；
- `$` 总能表示某一步的直接返回结果；受支持包装器的公开路径（当前为 `$.module`）可确定性验证，其他非根属性路径返回 `unresolved`；
- 对象引用目前要求生产步骤与使用位置位于同一源码文件和词法作用域，或直接以内嵌调用形式相连；跨函数返回值传播仍会返回 `unresolved`；
- 无法从源码机械提取 selector 时只记录 warning，不重新读取 checkpoint 或替 Agent 推导；
- checkpoint 加载指令通过只表示 proposal 可交给安全实例化工具尝试，不静态证明与模型实例兼容；
- 工具不执行目标项目，不创建模型实例，不调用模型工厂。

这些限制必须通过明确状态暴露，不能通过执行目标代码或不安全加载来绕过。
