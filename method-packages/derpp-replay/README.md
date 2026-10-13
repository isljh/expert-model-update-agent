# DER++ 版本式回放正则化方法包 v0.1

## 文档定位

本文面向需要理解、评审或维护本方法的人，完整说明 DER++ 是什么、怎样工作以及本项目具体实现了什么。本文不作为智能体的机器契约：更新决策 Agent 和更新执行 Agent 应读取同目录下的 [`method-card.json`](method-card.json)。面向代码维护者的集成要求见 [`../../docs/product-specs/derpp-method-package.md`](../../docs/product-specs/derpp-method-package.md)。

## 1. 方法是什么

DER++ 是一种基于样本回放的持续学习方法。模型获得新数据并继续训练时，不只学习新样本，还会同时读取少量历史样本，并利用两类历史信息约束当前模型：

1. 历史样本的真实标签；
2. 历史样本加入回放池时，旧模型产生的输出。

这样做的目标是在学习新数据的同时，降低模型对旧知识的遗忘。

本项目的分类损失采用 DER++ 的标准三项形式；回放池采用适合模型版本交付场景的“批准后追加”规则，不实现在线数据流中常用的 reservoir sampling。回归任务沿用相同的三项结构，但属于本项目明确标注的 DER++ 风格扩展，不表述为原论文标准定义。

## 2. 为什么需要历史样本和历史输出

只使用新数据继续训练，模型参数可能朝新数据分布移动，导致旧数据性能下降。DER++ 同时使用两种约束：

- 真实标签监督：提醒模型历史样本应得到什么正确结果；
- 历史输出约束：提醒模型旧版本曾怎样区分各类别或怎样给出连续预测。

对于分类任务，历史 logits 不只是最终类别，还包含各类别相对分数。即使两个模型预测出相同类别，它们的 logits 分布也可能不同，因此历史输出约束比只保存预测类别包含更多旧模型行为信息。

上游不需要额外交付 logits。本项目已经取得模型定义、配置、权重和代表性历史样本后，可以使用已恢复的旧模型自行生成并保存 logits。但是如果没有实际历史样本，仅有模型权重，就无法建立 DER++ 回放池。

## 3. 数学目标

设当前新样本批次为 \((X_n,Y_n)\)，历史回放批次为 \((X_r,Y_r,O_r)\)，其中 \(O_r\) 是这些样本加入回放池时由当时模型生成并冻结保存的输出，当前待更新模型为 \(f_\theta\)。一次更新的损失为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}(f_\theta(X_n),Y_n)
+\alpha\,\operatorname{MSE}(f_\theta(X_r),O_r)
+\beta\,\mathcal{L}_{\mathrm{task}}(f_\theta(X_r),Y_r).
$$

三项含义分别是：

1. 新样本主损失：学习本次新增数据；
2. 历史输出一致性损失：约束当前模型在历史样本上的输出不要过度偏离旧模型；
3. 历史标签监督损失：继续使用历史样本真实标签进行监督。

\(\alpha\) 控制历史输出约束强度，\(\beta\) 控制历史标签监督强度。当前实现要求二者均为非负数，并且至少一个大于零。具体取值必须由本次 `UpdatePlan` 给出，并通过验证集选择或实验确认，不能把示例值直接当作项目默认值。

分类任务通常令主损失为原任务使用的交叉熵等分类损失，\(O_r\) 保存主损失之前的 logits。回归任务保存主损失之前的连续预测值，并使用原回归主损失处理新样本和历史标签；输出一致性项仍使用 MSE。

## 4. 完整工作流程

### 4.1 建立初始回放池

1. 按模型交付契约恢复并验证初始模型；
2. 从上游提供的代表性历史训练样本中选择回放样本；
3. 使用与原模型一致的预处理生成实际模型输入；
4. 将模型切换为评估模式，在 `torch.no_grad()` 下生成训练输出；
5. 保存输入、真实标签、固定输出、样本编号和生成输出的模型版本；
6. 恢复模型调用前每个子模块的训练或评估状态。

初始模型的固定输出只生成一次。后续模型更新后，不得使用新模型重新计算这些记录的固定输出。

### 4.2 训练一个候选新版本

每个训练步骤同时取得一个当前新样本批次和一个历史回放批次。模型分别对两个批次前向计算，形成三项损失，执行一次反向传播和优化器更新。回放池中的固定输出只作为监督目标读取，不参与梯度计算，也不会被训练过程修改。

### 4.3 完成并批准一个版本

候选模型训练完成后，必须分别评估新数据能力和旧数据能力。只有模型通过评估并获得批准，才能：

1. 固定该版本的最终权重；
2. 从该版本的新数据中选择后续需要保留的代表性样本；
3. 使用最终模型为这些新选样本生成固定输出；
4. 将新记录追加到正式回放池。

训练过程中的临时 checkpoint、评估失败的候选版本和未批准版本不得写入正式回放池。

### 4.4 多版本关系

```text
初始模型 v1
→ 保存 v1 代表性样本及 v1 输出
→ 训练并批准 v2
→ 保存 v2 新增代表性样本及 v2 输出
→ 训练 v3 时，从包含 v1、v2 记录的回放池抽样
```

一条记录的 `source_version` 表示生成该条 `stored_outputs` 的模型版本，不表示当前正在训练的候选版本。

## 5. 回放池具体保存什么

每条回放记录对应一个样本：

| 内容 | 项目实现中的含义 |
|---|---|
| `inputs` | 能够再次传给模型的实际 Tensor 输入，可以是单个 Tensor、Tensor 字典或 Tensor 元组 |
| `targets` | 真实类别或真实回归目标，用于历史标签监督损失 |
| `stored_outputs` | 记录加入回放池时的模型训练输出；分类通常为 logits，回归为连续预测值 |
| `sample_id` | 回放池内全局唯一的非空字符串 |
| `source_version` | 生成固定输出的已确认模型版本 |

`stored_outputs` 不是 `argmax` 后的类别编号，也不是面向业务用户的后处理结果。它应与模型主损失实际接收的预测量对应。若业务结果需要 softmax、反标准化、阈值判断或其他后处理，回放池保存的仍是这些后处理之前的训练输出。

同一个实际样本不应仅通过更换 `sample_id` 被重复加入，否则会增加该样本在抽样和损失中的权重。核心库只检查编号是否重复，实际内容去重由执行层完成。

正式执行建议显式提供稳定的 `sample_id`。如果调用方省略，核心库会按照 `<source_version>:<递增编号>` 生成池内编号；该自动编号只保证当前回放池内唯一，不表示业务数据中的稳定身份。

## 6. 回放池容量和保留规则

### 6.1 容量怎样计算

`VersionedReplayBuffer(capacity=...)` 的容量按样本记录条数计算：

- 一个样本计为一条记录；
- 图片、掩码等多个 Tensor 组成的组合输入仍计为一个样本；
- `capacity=200` 表示最多保存 200 条记录；
- `capacity=None` 表示 API 不设条数上限，不表示机器拥有无限内存。

容量不是 DER++ 自动推导出的固定数字。`UpdatePlan` 应根据可用 CPU 内存或持久化空间、单条记录大小、预计版本数量、类别或场景覆盖要求以及回放批次大小确定容量。单条 Tensor 的基础内存可以用 `numel × element_size` 估算，同时还要考虑标签、固定输出、编号和 Python 对象开销。

### 6.2 当前保留策略

v0.1 的回放池采用：

```text
按批准模型版本追加
+ 已有记录保持不变
+ 不自动删除和替换
```

它没有实现 reservoir sampling、FIFO、随机淘汰、按类别配额、按版本配额、自动类别均衡或自动扩大容量。这与原始 DER++ 实验中常见的在线缓存维护方式不同。当前项目保留的是 DER++ 损失机制，缓存维护改为可审计的版本式规则。

### 6.3 容量不足时怎样处理

每次追加前检查：

```text
已有记录数 + 本次新增记录数 <= capacity
```

如果不成立，本次追加整体失败。系统不会保留其中一部分，也不会擅自删除旧样本。

正式流程只能选择以下处理方式：

1. 按已批准的代表性样本选择规则，在调用追加前减少本次新增样本；
2. 资源允许时，修改并重新批准 `UpdatePlan` 中的容量；
3. 如果需要删除或替换旧样本，先定义新的保留策略，再作为新方法版本实现、测试和审阅。

当前代码没有删除接口。调用方不能在报错后随意删除旧记录并继续把结果称为同一方法状态。

## 7. 写入、检查和抽样规则

### 7.1 写入检查

一批记录写入前会整体检查：

- 输入、标签和固定输出的样本数一致；
- 输入中所有 Tensor 的第一维均为样本维且长度一致；
- 新旧输入具有相同的字段结构、单样本形状和 dtype；
- 新旧标签具有相同的单样本形状和 dtype；
- 新旧固定输出具有相同的单样本形状和 dtype；
- 新增和历史 `sample_id` 均不重复；
- `source_version` 为非空字符串；
- 追加后不超过容量。

检查失败时不会只写入部分记录。成功写入时，Tensor 会与计算图分离、复制到 CPU，并形成独立副本。

### 7.2 抽样规则

`buffer.sample(batch_size, generator=..., device=...)`：

- 从全部历史记录中等概率随机抽样；
- 单次调用内部不放回，同一批次不会重复样本；
- 不同训练步骤间可以再次抽到同一样本；
- 不自动按照类别、版本、时间或来源分层；
- 不保证一次 epoch 恰好遍历全部回放记录；
- `batch_size` 必须大于零且不超过现有记录数；
- 可以使用 `torch.Generator` 控制可复现的随机序列；
- 返回批次可以移动到指定设备，池内正式副本仍保存在 CPU。

如果任务需要类别均衡或版本配额，必须在 `UpdatePlan` 中明确，并由执行层增加相应抽样实现，不能假定当前 `sample()` 已经包含该行为。

### 7.3 状态保存

- `buffer.all()` 返回全部记录的副本；
- `buffer.sample()` 返回抽样记录的副本；
- `buffer.state_dict()` 导出容量、格式版本、自动编号计数器、输入、标签、固定输出、样本编号和来源版本；
- `VersionedReplayBuffer.from_state_dict(...)` 恢复并重新检查状态；
- 核心库不规定磁盘文件名，也不自动执行 `torch.save`。

执行层负责把回放状态保存在独立运行产物目录中，并记录它与模型 checkpoint、方法版本和 `UpdatePlan` 的绑定关系。

## 8. 本项目怎样实现一次训练更新

`derpp_train_step` 执行以下过程：

1. `optimizer.zero_grad(set_to_none=True)`；
2. 对新样本执行模型前向；
3. 对历史回放样本执行模型前向；
4. 使用调用方提供的 `primary_loss` 计算新样本主损失；
5. 使用同一个 `primary_loss` 计算历史标签监督损失；
6. 使用 PyTorch 默认 `mean` reduction 的 MSE 计算历史输出一致性损失；
7. 按 \(1\)、\(\alpha\)、\(\beta\) 合并损失；
8. 执行 `backward()` 和 `optimizer.step()`；
9. 返回总损失、三项损失、权重和批次大小。

`primary_loss` 必须返回标量 Tensor。它使用 mean 还是 sum、是否使用类别权重、是否忽略某些标签，都由任务执行层根据原任务目标配置。

单步函数不会自动创建优化器、设置学习率、切换模型模式、冻结参数、使用混合精度、进行梯度累积或裁剪、调度学习率、运行 epoch 循环或保存 checkpoint。这些内容必须由获批准的任务执行代码负责。

## 9. 模型输入和输出怎样接入

当前实现支持：

- 单个 Tensor：调用 `model(inputs)`；
- Tensor 字典：调用 `model(**inputs)`；
- Tensor 元组：调用 `model(*inputs)`。

输出选择规则：

- 模型直接返回 Tensor：不需要选择器；
- 返回字典或具名对象：用字符串选择字段，例如 `"logits"`；
- 返回元组或列表：用整数选择位置；
- 多输出但没有提供选择器：直接报错，不猜测；
- v0.1 一次只处理一个被选中的输出 Tensor。

因此 CNN、ResNet、MLP 和 Transformer 都可能使用本方法，关键不在架构名称，而在于输入能够重放、训练输出能够唯一选定、主损失能够复现且输出语义和维度在版本间保持不变。

## 10. 适用条件与排除条件

必须同时满足：

1. 有少量可重新读取的历史代表性样本和真实标签；
2. 能恢复一个确定旧模型版本并为历史样本生成固定输出；
3. 增量前后是同一任务；
4. 输入预处理、输出含义、输出顺序和输出维度不变；
5. 能提供与原任务一致的主损失函数；
6. `UpdatePlan` 明确更新参数范围、优化器、学习率、批次、\(\alpha\)、\(\beta\)、容量和样本保留规则。

不能直接使用的情况包括：没有历史样本；类别、回归指标或输出维度变化；无法复现历史输入处理；主任务目标变化；输出过大无法回放；多任务头没有任务专用封装；无法评估旧数据能力。

## 11. 最小调用示例

```python
from torch import nn

from expert_model_update.methods.derpp import (
    VersionedReplayBuffer,
    derpp_train_step,
)

# replay_capacity、alpha、beta 和优化器参数来自已批准的 UpdatePlan。
buffer = VersionedReplayBuffer(capacity=replay_capacity)
buffer.add_from_model(
    model=old_model,
    inputs=old_inputs,
    targets=old_targets,
    source_version="1.0",
    sample_ids=old_sample_ids,
    output_selector="logits",  # 模型直接返回 Tensor 时省略。
)

model.train()
replay_batch = buffer.sample(
    replay_batch_size,
    generator=replay_generator,
    device=device,
)
metrics = derpp_train_step(
    model=model,
    optimizer=optimizer,
    current_inputs=new_inputs,
    current_targets=new_targets,
    replay_batch=replay_batch,
    primary_loss=nn.CrossEntropyLoss(),
    alpha=alpha,
    beta=beta,
    task_type="classification",
    output_selector="logits",
)
```

示例只展示方法调用关系。实际执行还必须处理设备迁移、随机种子、冻结范围、完整训练循环、验证集评估、checkpoint 保存和运行记录。

## 12. 已实现组件和当前状态

Python 入口：`expert_model_update.methods.derpp`

| 组件 | 作用 |
|---|---|
| `VersionedReplayBuffer` | 写入、检查、抽样和恢复版本式回放记录 |
| `capture_model_outputs` | 在 eval/no_grad 下生成固定输出，并逐模块恢复原训练状态 |
| `select_output` | 从模型返回结果中显式选择训练输出 Tensor |
| `compute_derpp_loss` | 计算三项损失及其明细 |
| `derpp_train_step` | 使用现有模型、优化器和两个批次完成一次更新 |

当前尚未实现完整训练作业、自动样本选择、类别均衡、跨版本容量分配、磁盘级回放数据集、断点续训、训练报告 Schema 和 Pi 工具入口。

## 13. 验证和相关文件

测试覆盖分类公式、回归扩展、历史输出不可变、混合 train/eval 状态恢复、字典输入、输出选择、状态恢复、容量失败、维度变化拒绝和一次参数更新。

- 智能体方法卡：[`method-card.json`](method-card.json)
- 开发者实现规格：[`../../docs/product-specs/derpp-method-package.md`](../../docs/product-specs/derpp-method-package.md)
- Python 实现：[`../../src/expert_model_update/methods/derpp/`](../../src/expert_model_update/methods/derpp/)
- 测试：[`../../tests/test_derpp.py`](../../tests/test_derpp.py)
- 通用方法卡 Schema：[`../../schemas/incremental-learning-method-card.schema.json`](../../schemas/incremental-learning-method-card.schema.json)
- 原始论文：Buzzega et al., *Dark Experience for General Continual Learning: a Strong, Simple Baseline*, NeurIPS 2020, <https://arxiv.org/abs/2004.07211>

```text
python -m unittest tests.test_derpp
```
