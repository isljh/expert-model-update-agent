# 模型实例化与资产证据环境

当前 Windows 工作站使用 `model-instantiation-safe` Conda 环境，服务于 `instantiate_validated_model`、`analyze_validated_model_structure`、`inspect_model_checkpoint` 和 `inspect_model_config`。它不替代保留不动的旧 `model-analysis` 环境，也不是模型训练环境。

在仓库根目录的 PowerShell 中重建：

```powershell
conda env create -f .\environments\model-instantiation-safe.yml
conda run -n model-instantiation-safe python -m pip install -r .\environments\requirements-model-instantiation-cpu.txt
conda run -n model-instantiation-safe python -c "import torch, torchvision, yaml; print(torch.__version__, torchvision.__version__, yaml.__version__)"
```

直接使用的解释器路径为 `D:\software\envs\model-instantiation-safe\python.exe`。Pi Extension 在此工作站默认使用该路径；部署到其他机器时，可分别设置 `MODEL_INSTANTIATION_PYTHON` 和 `MODEL_EVIDENCE_PYTHON` 为新环境的绝对解释器路径。

依赖文件仅固定 CPU 版 PyTorch、torchvision 和配置解析所需 PyYAML。目标项目特有的依赖必须根据该项目的导入错误和已确认的依赖清单单独加入，不把旧环境的所有训练依赖未经测试地复制过来。`weights_only=True` 减少 checkpoint 反序列化风险，但本地执行目标项目代码不具备操作系统级隔离。

## 增量方法组件

解析增量岭回归组件依赖 NumPy，最小依赖见 [`requirements-incremental-methods.txt`](requirements-incremental-methods.txt)。Pi Extension 默认使用当前 `python`/`python3`；部署环境可通过 `INCREMENTAL_METHODS_PYTHON` 指定解释器。该环境只运行标准化矩阵计算，不因此获得读取目标项目或业务数据的额外授权。
