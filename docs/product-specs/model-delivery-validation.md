# 模型交付验证规格

## 1. 目标

validate_model_delivery 对上游唯一声明的模型文件、构造配置和 checkpoint 证据进行确定性验证，并在全部条件满足时生成 validated_load_plan。

它取代旧的“项目理解 → 多个模型候选 → 创建方案静态验证”流程。工具不搜索项目、不生成候选，也不从权重形状猜测构造参数。

## 2. 输入

工具接收四项完整输入：

1. 已确认的 TaskInput v0.3；
2. 与当前任务和资产摘要匹配且已通过的 AssetPreflightReport；
3. inspect_model_config 的成功报告；
4. inspect_model_checkpoint 的成功报告。

任何输入缺失、绑定不一致或文件在检查后发生变化，均阻止生成加载计划。

## 3. 固定验证内容

### 3.1 任务与资产绑定

- TaskInput 更新目标和人工确认均为 confirmed；
- confirmed_task_input_version 等于当前版本；
- 预检报告任务、版本和 asset_input_digest 一致；
- 模型、配置和 checkpoint 三项检查均唯一通过；
- 配置和 checkpoint 的当前 SHA-256 与检查报告一致。

### 3.2 构造配置

配置必须符合 [Model Load Config Schema](../../schemas/model-load-config.schema.json)：

- 版本为 0.1；
- 框架为 pytorch；
- model_class 是简单 Python 标识符；
- constructor_kwargs 是 JSON 安全的关键字参数对象；
- state_dict_path 是 $ 或 $.name 形式；
- 键转换仅允许 strip_prefix 和 add_prefix；
- strict 固定为 true。

第一版只支持关键字构造参数。需要必须按位置传入的模型类不进入加载阶段，必须由上游调整为受支持的交付接口。

### 3.3 模型文件静态检查

验证器只把模型文件解析为 Python AST，不导入执行。它检查：

- 文件可解码且语法有效；
- 顶层存在唯一的声明模型类；
- 配置中的构造参数名被构造函数接受；
- 所有可静态识别的必需构造参数都已提供；
- 不存在第一版不支持的必需位置参数。

静态签名匹配不证明构造函数运行成功，也不证明该类是 torch.nn.Module；这些由实际加载工具验证。

### 3.4 checkpoint 证据

配置声明的 state_dict_path 必须在 checkpoint 检查报告中唯一存在，并对应非空的纯张量映射。验证器不加载权重到模型，也不根据张量形状补构造参数。

## 4. 输出

成功报告类型为 ModelDeliveryValidationReport v0.1，包含：

- task_binding；
- findings；
- validated_load_plan；
- ready_for_model_load；
- ready_for_structure_analysis；
- blockers。

validated_load_plan 只包含执行所需的固定信息：

- load_spec_id；
- 模型文件路径、摘要、类名和构造参数；
- 配置路径和摘要；
- checkpoint 路径、摘要、state_dict 选择路径、键转换和 strict=true。

报告 Schema 见 [Model Delivery Validation Report](../../schemas/model-delivery-validation-report.schema.json)。

## 5. 执行门槛

只有 ready_for_model_load=true 时，load_validated_model 才能执行。执行工具必须用原始四项输入重新运行验证，并要求重新生成的加载计划与传入报告完全一致。

实际兼容性的唯一证据是：

```text
模型类成功创建
+ checkpoint 使用 weights_only 读取
+ load_state_dict(strict=True) 成功
+ missing_keys 为空
+ unexpected_keys 为空
```

静态验证通过不能替代以上运行证据。

## 6. 非目标

- 不读取原训练项目或训练流程；
- 不发现模型类、工厂或注册表；
- 不支持多个候选或自动选择；
- 不执行 forward；
- 不判断应该采用哪种增量学习方法；
- 不证明核心任务目标、主损失或预处理语义。
