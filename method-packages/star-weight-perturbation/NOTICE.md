# 来源与许可证说明

本方法包依据 Eskandar 等人在 ICLR 2025 发表的 STAR 论文，以及作者公开的
[`Gnomy17/STAR_CL`](https://github.com/Gnomy17/STAR_CL) MIT 许可证实现进行设计核对。

本项目没有复制 Mammoth 训练框架，也没有逐行复制官方 `star_perturber.py`。核心代码按照本项目的组合式方法包接口重新实现，并增加了输入契约、异常恢复、显式跳过结果、独立梯度返回和测试。

原论文和官方实现的著作权归原作者所有。使用或分发上游代码时仍须遵守其仓库中的 MIT License。
