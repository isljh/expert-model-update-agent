# STAR 分类权重扰动稳定性组件

本目录实现 `star_weight_perturbation` v0.1。它是回放式持续学习方法的可选增强组件，不是能够单独完成模型更新的基础方法。更新决策 Agent 和执行 Agent 应读取 [`method-card.json`](method-card.json)；维护者的精确集成规则见 [`../../docs/product-specs/star-method-package.md`](../../docs/product-specs/star-method-package.md)。

## 1. 方法解决什么问题

ER、DER++ 等方法让模型在学习新样本时重新训练少量历史样本，但“此刻仍能正确分类”不代表模型对后续参数变化稳定。STAR 在当前参数附近主动寻找一种会显著改变历史预测的权重扰动，再利用该扰动产生的梯度把模型推向局部预测更稳定的参数区域。

它不冻结参数，也不保证未来更新恰好沿当前找到的扰动方向。它使用局部最坏扰动作为未来遗忘风险的近似。

## 2. 与基础方法的关系

STAR 必须组合在一个能够提供历史样本和真实类别标签的回放方法之上：

```text
历史回放批次
  → STAR 计算并累积稳定性梯度
  → ER、DER++ 等基础方法计算训练梯度
  → 同一次 optimizer.step 更新真实模型
```

本项目第一版优先与 `derpp_replay` 组合，但 STAR 方法包不依赖 DER++ 保存的历史 logits，也不拥有 DER++ 的回放池。

## 3. 数学目标

设当前模型参数为 \(\theta\)，回放批次为 \(\mathcal M\)。先选出当前正确分类的样本：

$$
\mathcal M^*=\{(x,y)\in\mathcal M\mid\arg\max f_\theta(x)=y\}.
$$

将 logits 通过 softmax 转为概率 \(q_\theta(x)\)，STAR 的理论目标是：

$$
\mathcal L_{\mathrm{STAR}}(\theta)
=
\max_{\|\delta\|_2\le d}
\frac{1}{|\mathcal M^*|}
\sum_{(x,y)\in\mathcal M^*}
D_{\mathrm{KL}}
\left(q_\theta(x)\parallel q_{\theta+\delta}(x)\right).
$$

组合目标为：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{base}}
+\lambda\mathcal L_{\mathrm{STAR}}.
$$

其中基础损失和 STAR 的超参数必须由本次 `UpdatePlan` 给出。

## 4. 本项目怎样近似最坏扰动

对每个被批准的参数 \(\theta^{(l)}\)，先生成小随机扰动：

$$
\delta_0^{(l)}=z^{(l)}\epsilon\|\theta^{(l)}\|_2,
\qquad z^{(l)}\sim\mathcal N(0,I).
$$

然后计算使 KL 增大的梯度，并按层归一化：

$$
\delta^{(l)}
\leftarrow
\delta^{(l)}
+\frac{\gamma}{K}
\frac{\|\theta^{(l)}\|_2}
{\|g^{(l)}\|_2+10^{-20}}
g^{(l)}.
$$

其中 \(K\) 是扰动梯度上升步数。最后在扰动参数点计算加权 STAR 梯度，恢复真实参数，再把该梯度累积到基础方法梯度中。

与作者公开实现一致，v0.1 默认只扰动 `requires_grad=True` 且维度大于 1 的浮点参数，因此不扰动偏置和 BatchNorm 等归一化层的一维仿射向量。调用方也可以通过 `parameter_names` 提交更小的已批准范围，但不能把一维或冻结参数加入范围。

## 5. 分类边界

v0.1 只支持单标签分类：

- 输出必须是损失函数之前的二维 logits，形状为 `[batch, classes]`；
- 类别数至少为 2；
- 标签必须是一维整数类别索引；
- 增量前后类别集合、顺序和输出维度不变；
- 概率固定使用 `softmax(logits, dim=1)`；
- KL 只在当前预测正确的回放样本上求平均。

普通回归输出没有类别概率，也没有“当前分类正确”的定义，因此不能使用这个方法包。若未来研究回归扩展，必须使用新的方法标识、目标函数和验证证据。

## 6. 模型状态与参数安全

STAR 需要多次临时前向。为避免 Dropout 随机性或 BatchNorm 可变缓冲区被误当作权重敏感性，v0.1 在 STAR 专用前向期间临时使用 `eval` 模式，并在结束后逐模块恢复原训练状态。基础方法仍按执行层配置的训练状态运行。

临时扰动使用受保护的添加/恢复区间。正常返回或前向、KL、梯度计算发生异常时，已经添加的扰动都会在异常继续抛出前撤销。`compute_star_gradients` 不改变参数的 `.grad`；`accumulate_star_gradients` 只把计算结果加到现有梯度上，从不调用 `zero_grad` 或 `optimizer.step`。

## 7. 正确样本不足

官方实现会在正确样本少于一个最低数量时跳过 STAR。v0.1 将该数量作为 `min_correct_samples` 显式写入 `UpdatePlan`。不足时返回：

```json
{
  "skipped": true,
  "skip_reason": "insufficient_correct_replay_samples"
}
```

本次调用不增加梯度。若连续多步都跳过，说明回放样本、基础模型能力或阈值假设不成立，应触发重新规划，而不是静默宣称 STAR 已生效。

## 8. 公共接口

入口：`expert_model_update.methods.star`

- `STARConfig`：保存必须由计划确定的超参数和参数范围；
- `compute_star_gradients`：计算并返回脱离计算图的 STAR 梯度，不修改 `.grad`；
- `accumulate_star_gradients`：把 STAR 梯度累积到模型现有 `.grad`；
- `STARGradientResult`：返回损失、正确样本数、扰动范数、跳过状态和梯度。

## 9. 组合调用示例

下面的变量值应来自已批准的 `UpdatePlan`，不能照抄论文或示例默认值：

```python
from expert_model_update.methods.star import STARConfig, accumulate_star_gradients

star_config = STARConfig(
    regularization_weight=approved_star_lambda,
    perturbation_ratio=approved_gamma,
    noise_scale=approved_noise_scale,
    ascent_steps=approved_ascent_steps,
    min_correct_samples=approved_minimum,
    parameter_names=approved_parameter_names,
)

optimizer.zero_grad(set_to_none=True)

star_result = accumulate_star_gradients(
    model=model,
    replay_inputs=replay_batch.inputs,
    replay_targets=replay_batch.targets,
    config=star_config,
    output_selector="logits",
    generator=approved_generator,
)

base_breakdown = compute_approved_base_loss(...)
base_breakdown.total.backward()
optimizer.step()
```

顺序不能颠倒：若在 STAR 之后再次调用 `zero_grad`，STAR 梯度会被清除；若 STAR 自行执行一次 `optimizer.step`，则不再是论文定义的组合梯度更新。

## 10. 验证和报告

每次正式训练至少记录：

- 回放批次大小与当前正确样本数；
- 是否跳过及原因；
- 原始 KL 和乘以 \(\lambda\) 后的值；
- 被扰动参数数量及逐参数扰动 L2 范数；
- 随机种子策略和 STAR 超参数；
- 基础方法单独运行与加入 STAR 后的新旧验证指标；
- 额外训练时间和显存开销。

STAR 训练 KL 不是旧能力验证指标。最终仍需在独立的新数据与历史验证数据上评估，并遵守 `UpdatePlan` 的退化阈值。

## 11. 当前限制

- 不支持回归、多标签分类、生成任务或类别集合变化；
- 不管理回放池、不保存历史 logits、不选择样本；
- 不创建优化器、不执行完整训练循环、不保存 checkpoint；
- 不自动选择 \(\lambda\)、\(\gamma\)、噪声尺度、步数或参数范围；
- 最坏扰动是一阶局部近似，不能保证覆盖未来真实更新方向；
- 官方论文的实验收益不能直接外推到本项目模型，必须进行有/无 STAR 的受控对照。

## 12. 来源

- Masih Eskandar, Tooba Imtiaz, Davin Hill, Zifeng Wang, Jennifer Dy. *STAR: Stability-Inducing Weight Perturbation for Continual Learning*. ICLR 2025.
- 官方实现：[`Gnomy17/STAR_CL`](https://github.com/Gnomy17/STAR_CL)，MIT License。

本项目的来源与重新实现说明见 [`NOTICE.md`](NOTICE.md)。
