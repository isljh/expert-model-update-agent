# 模型加载、严格权重恢复与结构分析

## 1. 输入与重新验证

load_validated_model 和 analyze_validated_model_structure 接收：

- 传给 validate_model_delivery 的原始 delivery_inputs；
- 对应的 ModelDeliveryValidationReport v0.1。

宿主进程重新运行模型交付验证，并要求任务绑定、就绪状态和 validated_load_plan 与传入报告完全一致。工具不接受 Agent 手写的单独加载计划。

## 2. 本地执行边界

Pi 工具复核后启动本地 Python 子进程。Windows 原型默认使用：

```text
D:\software\envs\model-instantiation-safe\python.exe
```

可以通过 MODEL_INSTANTIATION_PYTHON 和 MODEL_EVIDENCE_PYTHON 分别覆盖加载和证据检查环境。

子进程不是操作系统沙箱。导入模型文件和调用模型构造函数可能写文件、访问网络、初始化设备或启动其他进程。临时工作目录、禁用字节码、离线环境变量和超时只能缩小风险，不能保证资产只读、网络隔离或子进程完全清理。

## 3. 固定执行顺序

```text
重新验证交付物
→ 核对模型、配置、checkpoint 的 SHA-256
→ 导入声明模型文件
→ 取得唯一声明模型类
→ 使用 constructor_kwargs 创建实例
→ 确认实例是 torch.nn.Module
→ 使用 weights_only=True 读取 checkpoint
→ 选择声明的 state_dict
→ 应用声明的键前缀转换
→ load_state_dict(strict=True)
→ 确认 missing_keys 和 unexpected_keys 均为空
→ 统计总参数与 requires_grad=true 的可训练参数
→ 可选：在同一进程分析完整注册模块结构
```

任一步失败都不能声称已经恢复原模型。

## 4. 权重加载

- checkpoint 必须是普通文件且不超过 2 GiB；
- 不支持 weights_only 时不回退普通 pickle；
- strict 固定为 true；
- 实际张量数量必须与检查证据一致；
- missing_keys 或 unexpected_keys 非空即失败；
- 报告不输出张量值。

## 5. 结构与可训练性分析

analyze_validated_model_structure 在创建并严格加载权重后直接分析同一内存对象，不需要也不应该先调用 load_validated_model。

分析包括：

- 模型类和 Python 模块；
- 注册模块树及常见结构属性；
- 唯一参数张量、形状、dtype、requires_grad；
- 总参数、可训练参数和冻结参数数量；
- 缓冲区；
- 共享模块与共享参数；
- 量化迹象；
- 启发式模型家族。

第一版不执行 forward，因此不能证明输入样例可运行、输出语义正确或反向传播成功。has_trainable_parameters 只表示至少有一个参数允许梯度，不等于已经证明完整训练链可用。

## 6. 报告保存

结构分析宿主把完整 ModelInstantiationReport 和逐层 Markdown 树保存到：

```text
run-records/model-structure/<模型交付目录名>/
```

每次运行使用唯一时间和随机标识，不覆盖历史记录。输出目录不能位于模型交付目录中，worker 不负责写正式报告。

## 7. 成功判据

只有同时满足以下条件才可表述为“原模型实例已恢复”：

- status=completed；
- 模型实例是 torch.nn.Module；
- checkpoint.status=loaded；
- weights_only=true；
- strict=true；
- fully_matched=true；
- missing_keys 和 unexpected_keys 均为空。

这仍不代表模型已经适合某种增量学习方法。方法选择还需要任务目标、主损失含义、新数据、旧能力保持要求以及方法特定状态。
