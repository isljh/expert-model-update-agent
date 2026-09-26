# 模型实例化 v0.1（本地执行）

## 目标与输入

`instantiate_validated_model` 接收原始 `ModelCreationProposalSet` v0.2 和对应的 `ModelCreationValidationReport` v0.2。宿主进程重新运行薄静态验证，并要求任务绑定、选择结果、就绪状态和 `validated_instantiation_plan` 与传入报告完全一致。工具不接受 Agent 手写的单独计划。

通过复核后，按计划中的有序 `construction_steps` 创建对象，解析 `object_ref`，执行声明的生命周期操作，确定运行时对象与目标 `torch.nn.Module`。提供 checkpoint 加载指令时，仅用 `torch.load(..., weights_only=True, map_location="cpu")` 读取；不支持该模式或加载失败时不退回普通 pickle。selector、键转换、加载目标和 `strict` 均来自已核验计划，不由执行器重新选择。

执行器输出 [`../../schemas/model-instantiation-report.schema.json`](../../schemas/model-instantiation-report.schema.json) 所定义的报告。`status=completed` 表示计划执行完成；`checkpoint.status=loaded` 才表示权重实际装入。若模型已创建但加载失败，报告保留已知模型元数据并明确 `checkpoint.status=failed`；不能把新初始化参数称为原模型权重。

## 当前执行方式与风险边界

Pi 工具复核计划后，直接启动本地 Python 子进程运行 `instantiate_worker.py`。在当前 Windows 工作站，Pi 扩展默认使用 `D:\software\envs\model-instantiation-safe\python.exe`；项目理解阶段的声明资产检查器也默认使用该环境。显式设置 `MODEL_INSTANTIATION_PYTHON` 或 `MODEL_EVIDENCE_PYTHON` 可以分别覆盖选择。worker 接收 `validated_instantiation_plan` JSON，导入项目模块、调用声明的模型创建步骤，并在退出前返回一个 `ModelInstantiationReport`。报告记录实际使用的 Python 路径、Python 版本与 PyTorch 版本。不需要额外配置操作系统级隔离运行器。

导入或调用目标项目的 Python 代码本身就可能下载数据、写入缓存、初始化 GPU、启动训练、修改项目文件或访问网络。本地子进程、临时工作目录、禁用字节码、环境变量和 120 秒超时只能减少部分意外影响，**不能保证目标项目只读、网络隔离、凭据隔离或其子进程被清理**。工具报告中的 `execution_boundary.kind` 必须明确写为 `local_subprocess_without_os_isolation`，不能称为沙箱。

工具不会运行整个训练入口，而是调用已验证方案中指定的模型创建 callable；但 callable 及其导入依赖仍可能执行任意 Python 代码。使用前应明确接受这一执行边界。`blocked` 只表示方案、执行环境或报告不满足要求，不再代表缺少隔离运行器。

## 源码与权重状态

worker 在导入前核对每个 callable 源文件、创建锚点和 checkpoint 加载锚点的 SHA256 是否仍与静态验证时一致。它不主动运行项目的训练入口，但无法阻止项目模块在导入或工厂调用时自行执行其他操作。

checkpoint 文件在执行阶段必须为普通文件且不超过 2 GiB。目标模型和加载目标必须是 `torch.nn.Module`。严格加载按 proposal 的 `strict` 执行；报告保留缺失/意外键，但不输出权重值。实例对象只存在于 worker 内存，报告不序列化模型实例。需要结构事实时，调用 `analyze_validated_model_structure`，由同一 worker 在创建和加载后直接分析实例；不要先调用 `instantiate_validated_model` 再用其 JSON 报告尝试恢复实例，也不要顺序调用两工具导致重复创建。结构报告见 [`../../schemas/model-structure-analysis-report.schema.json`](../../schemas/model-structure-analysis-report.schema.json)。

## 结构解析第一版

结构解析逻辑参考旧工具的模块树、层属性、唯一参数统计、共享对象、量化迹象和模型家族证据，适配到当前 `ModelCreationProposalSet` 创建链。输出是 `ModelInstantiationReport.structure_analysis`，其中记录 checkpoint 是否真正加载、缺失/意外键以及是否完全匹配；非严格加载有键不匹配时不能宣称原模型权重完整恢复。没有 checkpoint 时明确标记为新初始化参数。模型家族是有证据的启发式分类，不是业务策略判断。

第一版只检查实例中注册的模块、参数与缓冲区；不执行 `forward()`、训练或数据加载，因此不提供真实输入输出签名、动态分支覆盖和前向轨迹。自定义层的未注册属性不会猜测。结构报告受 worker 输出 4 MiB 上限约束，超限时返回 `blocked`，不能声称结构分析完成。

## 正式运行记录

`analyze_validated_model_structure` 的宿主端在核对 worker 报告后，将本次完整 `ModelInstantiationReport`（含 `structure_analysis`）与从同一数据生成的逐层展开 Markdown 树保存到智能体工作区的 `run-records/model-structure/<目标项目目录名>/`。同一次运行使用相同的 `<UTC时间>-<随机运行号>` 文件名前缀，分别生成 `<运行号>.model-instantiation-and-structure-report.json` 与 `<运行号>.model-structure-tree.md`；不会覆盖以前的文件。返回的 `record` 字段包含项目目录名、绝对路径和唯一运行号。不同路径但同名的目标项目会共用同名文件夹，因此 Pi 摘要和完整报告都保留目标项目绝对路径以供区分。Pi 的工具正文只返回执行状态、项目路径、checkpoint/结构摘要和这两个报告路径；后续 Agent 如需完整证据，应读取 JSON 文件，不能把摘要当作完整报告。

写入目录由宿主代码固定选择，不接受 proposal、目标项目代码或 Agent 指定的输出路径；若解析后的目录位于目标项目内，工具在执行目标代码前拒绝。worker 不负责写正式报告。模型创建或结构解析失败但 worker 返回有效失败报告时，也会保存失败记录；输入复核阶段即被阻断时不创建运行目录。若正式报告写入失败，工具返回 `REPORT_WRITE_FAILED`，不得声称交接产物已经保存。当前报告仍受 worker 的 4 MiB 输出上限约束，超大模型需要后续单独设计流式产物交付。

## 当前边界

- 项目模块导入可能需要第三方依赖；缺失时明确失败，不自动安装。
- `DataParallel` 与 `DistributedDataParallel` 是否可在当前本地环境创建取决于运行条件；失败时不擅自跳过包装。
- 当前只支持单文件 PyTorch checkpoint；目录/分片格式需另行设计。
- 新建的 `model-instantiation-safe` 环境是 Python 3.11.16、PyTorch 2.14.0+cpu、torchvision 0.29.0+cpu。受控临时 checkpoint 的 `weights_only=True` 加载和实例化单元测试已通过；公开验证项目 `pytorch_resnet_cifar10` 的声明 checkpoint 也已在固定源码版本上实际严格加载，报告显示 97 个张量且没有 missing/unexpected keys，并完成后续结构解析。该结果只验收这一公开案例，不代表其他项目或真实业务项目已经兼容；目标代码副作用也尚未得到操作系统级约束或独立审计。旧 `model-analysis` 环境保持 Python 3.9/PyTorch 1.12，不再作为新工具默认运行环境。当前仍不自动退回普通 pickle。
- 未提供 checkpoint 时可报告新初始化模型结构，但不能宣称原权重已加载。
- 前向推理、训练和数据加载不属于当前结构解析；第一版不迁移旧工具的技术示例输入与前向 Hook。
