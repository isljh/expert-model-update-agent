# 0003：任务资产入口预检

## 状态

已接受；字段集合已随 TaskInput v0.3 更新。

## 背景

TaskInput 中的路径字符串满足结构契约，并不能证明当前运行环境可以访问相应资产。若到模型检查或训练阶段才发现路径不存在，失败过晚且错误语义不清。

## 决定

1. 在人工最终确认前运行一次只读资产预检。
2. confirmed 校验通过后、进入模型交付检查前再次运行。
3. 报告独立于 TaskInput，通过 task_id、task_input_version 和 asset_input_digest 绑定。
4. v0.3 固定检查：
   - 模型文件：普通可读文件；
   - 构造配置：普通可读文件；
   - checkpoint：普通可读文件；
   - 新数据：普通文件或目录。
5. 预检只检查存在性、类型和可读性，不读取内容、不递归扫描、不导入代码、不加载 checkpoint、不修改资产。
6. 预检失败时不得请求最终确认；确认后的复核失败时不得继续交接。

## 边界

配置内容由 inspect_model_config 检查，checkpoint 安全性与结构由 inspect_model_checkpoint 检查，模型类和构造参数由 validate_model_delivery 核对，实际兼容性由 strict 权重加载证明。
