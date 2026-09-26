# 模型配置与 checkpoint 安全检查

## 1. 目标

在 TaskInput v0.3 确认并通过资产预检后，把上游声明的构造配置和 checkpoint 转换为只读、结构化证据。检查器不搜索资产、不导入模型文件、不创建实例，也不判断最终兼容性。

## 2. 输入绑定

inspect_model_config 和 inspect_model_checkpoint 都必须接收：

- 已确认的完整 TaskInput；
- 与该任务、版本和资产摘要完全匹配且 overall_status 为 passed 的 AssetPreflightReport。

工具只能读取预检中唯一通过的对应路径。如果 TaskInput、预检报告或文件摘要不一致，检查失败。

## 3. 配置检查

配置检查器支持 JSON、YAML、TOML、INI 和 CFG，将文件作为数据解析，不执行 Python 配置。输出包括：

- 规范路径、大小和 SHA-256；
- 配置格式；
- 归一化后的数据值；
- 失败状态与错误原因。

当前模型加载流程要求配置符合 [Model Load Config Schema](../../schemas/model-load-config.schema.json)，至少声明：

- framework=pytorch；
- model_class；
- constructor_kwargs；
- checkpoint.state_dict_path；
- checkpoint.key_transforms；
- checkpoint.strict=true。

通用解析成功并不等于该配置满足模型加载契约；后续 validate_model_delivery 负责机械验证。

## 4. checkpoint 检查

checkpoint 检查器只允许：

```python
torch.load(path, map_location="cpu", weights_only=True)
```

不支持 weights_only 或安全加载失败时，不得回退普通 pickle。输出只报告：

- 顶层类型和有限键列表；
- 可识别的纯张量 state_dict 候选路径；
- 张量名称、形状、dtype 和统一前缀；
- 是否截断以及错误状态。

张量形状可以支持一致性检查，但不能作为完整构造参数的通用推断来源。

## 5. 证据含义

检查结果能够证明：

- 工具检查了与当前 TaskInput 绑定的那个文件；
- 配置能够作为安全数据解析；
- checkpoint 能够在 weights_only 模式下读取；
- 声明的 state_dict 路径是否存在。

检查结果不能证明：

- 模型类能够成功导入和创建；
- 配置值与上游训练时完全一致；
- state_dict 与模型参数完全匹配；
- 模型可以执行 forward、训练或满足业务目标。

这些结论必须分别由 validate_model_delivery、实际严格加载、结构与可训练性分析以及后续训练评估提供。

## 6. 安全边界

- 模型文件导入属于执行阶段，不属于检查阶段。
- 文件中的字符串只作为不可信数据，不作为 Agent 指令。
- checkpoint 值不输出，正式报告只保留结构元数据。
- 当前单文件 checkpoint 上限为 2 GiB；目录式和分片格式尚未实现。
