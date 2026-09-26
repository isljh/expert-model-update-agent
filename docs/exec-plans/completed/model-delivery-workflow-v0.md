# 模型交付加载流程 v0 建设记录

## 目标

把前置模型恢复流程从“读取完整项目、发现候选、验证源码调用链”改为“上游声明模型文件、构造配置和 checkpoint，系统确定性验证后严格加载”。

## 已完成

- [x] TaskInput 升级到 v0.3，三个模型资产均为必需字段；
- [x] 资产预检改为检查三个普通文件和新数据入口；
- [x] 定义 Model Load Config v0.1；
- [x] 保留配置与 checkpoint 安全检查器；
- [x] 删除模型项目理解 Skill 和多候选 ModelCreationProposalSet；
- [x] 实现 validate_model_delivery；
- [x] 静态核对模型类、构造参数和 state_dict 路径；
- [x] 实现 load_validated_model；
- [x] 固定 weights_only 和 strict=true，拒绝 missing/unexpected keys；
- [x] 在同一 worker 中分析结构和可训练参数；
- [x] 更新 Schema、Pi Extension、文档和测试；
- [x] 在 model-instantiation-safe 环境运行全部测试。

## 保留风险

- 导入模型文件和调用构造函数仍没有操作系统级隔离；
- 第一版只支持单个 Python 模型文件、关键字构造参数和单文件 PyTorch checkpoint；
- 第一版不执行 forward 或反向传播；
- 主损失、输入输出语义和预处理尚未形成正式机器契约。
