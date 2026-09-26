# 模型项目理解 Skill v0 建设计划

**状态：** Skill 与其下游前置流程第一版已贯通，保留在 active 目录继续处理 Pi 自动发现和运行边界加固。

## 目标

在 `TaskInput` 确认后，为更新决策 Agent 提供一个只读、证据化的项目理解流程，输出模型候选和模型创建方案，供后续静态验证工具使用。

## 已确认边界

```text
TaskInput confirmed
  → 资产入口快速复核通过
  → 项目理解 Agent
  → 模型候选与创建方案
  → 静态验证工具
  → 安全实例化工具
  → 结构解析工具
  → 更新决策 Agent
```

- Skill 不执行目标项目代码，不创建实例，不加载权重。
- Agent 不通过临时代码替代安全实例化工具。
- 原工具的三个入口字段由项目理解阶段发现，不再要求用户提供。
- 多候选不能按源码顺序默认选择第一个。
- `ready_for_static_validation` 与 `ready_for_instantiation` 必须严格区分。

## 当前实现

- [x] 创建 `skills/model-project-understanding/SKILL.md`。
- [x] 定义逻辑输出模板和证据状态。
- [x] 明确与 TaskInput、静态验证、实例化和结构解析的边界。
- [x] 将与当前 TaskInput 版本和资产声明匹配的已通过预检报告设为进入条件。
- [x] 用真实项目案例验证 Skill 的候选发现质量。
- [x] 根据失败案例收紧候选、创建链、参数状态和运行时表述。
- [x] 将实例创建所需字段拆为独立 `ModelCreationProposalSet` v0.2 Schema，用有序步骤与对象引用表达完整创建链，并实现对应静态验证工具。
- [x] 接入人类声明配置和 checkpoint 的安全只读证据检查，不自动搜索配置文件。
- [x] 接入本地实例化 worker、严格 checkpoint 加载、实例结构解析及正式报告保存。
- [x] 使用公开验证项目从 TaskInput 确认后的项目理解运行到结构报告，证明第一版交接链可以贯通。

## 尚未完成或不属于本 Skill

- 静态验证工具后续增强；
- 不同项目结构、checkpoint 格式和依赖环境的通用兼容；
- 前向轨迹、动态分支和真实输入输出关系验证；
- 操作系统级实例化隔离及副作用独立审计；
- 更新方法选择、`UpdatePlan`、项目改造、训练和评估；
- 将根目录 `skills/` 通过项目级 Pi 配置加入自动发现范围；
- Pi 阶段级工具权限控制。

## 首个公开验证候选

选择 [`akamaster/pytorch_resnet_cifar10`](https://github.com/akamaster/pytorch_resnet_cifar10) 作为 Skill 的首个公开只读演练候选，初始目标使用仓库内的：

```text
pretrained_models/resnet20-12fca82f.th
```

选择依据：

- PyTorch 源码、训练入口和多个预训练权重位于同一公开仓库；
- ResNet20 权重约 1.05 MB，适合低成本验证；
- 模型由 `resnet.__dict__[args.arch]()` 动态选择，可检验 Agent 是否沿配置和调用链确定候选；
- 训练入口使用 `torch.nn.DataParallel`，可检验包装对象、实际模型对象和 checkpoint 键绑定；
- 训练入口包含 CUDA 初始化和 CIFAR-10 下载，适合检验 Skill 是否在只读调查后停止而不执行项目；
- 模型本身可由源码构造，不需要先取得数据集维度。

该项目当前只作为开发期公开验证案例，不代表首个正式模型更新项目，也不构成隐藏评估答案。外部源码和权重保存在智能体项目之外；正式工具运行报告保存在本仓库的 `run-records/model-structure/pytorch_resnet_cifar10/`，开发测试输入和参考答案仍不得写入 Pi 正式工作区。

## 公开案例验收结果

- [x] 固定并记录实际使用的仓库 revision；
- [x] 从 TaskInput、资产预检和本 Skill 的只读调查形成具体创建方案，没有由用户预填模型入口；
- [x] 使用支持 `weights_only=True` 的独立环境检查并实际加载声明 checkpoint；
- [x] 通过静态验证后按两步创建链构造 ResNet20 与 `DataParallel` 包装对象；
- [x] 严格加载 97 个 checkpoint 张量，missing/unexpected keys 均为 0；
- [x] 在同一 worker 中解析 61 个注册模块和 269722 个参数，并保存完整 JSON 与展开树；
- [ ] 独立检查本地执行是否产生目标项目写入、网络访问或遗留子进程。当前成功报告不能证明没有副作用；
- [ ] 使用结构不同的第二个项目验证候选发现和创建方案的通用性。
