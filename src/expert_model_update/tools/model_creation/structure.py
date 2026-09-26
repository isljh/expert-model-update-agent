"""Inspect an already-created PyTorch module without executing forward().

The extraction logic is adapted from the earlier model_structure_analyzer. It
does not locate, import, construct, or reload a model; the caller supplies the
live object in the same worker process that created it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


MAX_MODULE_PATHS = 10_000
MAX_PARAMETER_PATHS = 50_000


def _path(value: str) -> str:
    return value or "<root>"


def _registered_modules(model: Any) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []

    def visit(module: Any, path: str, ancestors: frozenset[int]) -> None:
        if len(records) >= MAX_MODULE_PATHS:
            raise ValueError("注册模块路径超过 10000 条上限。")
        records.append((path, module))
        next_ancestors = ancestors | {id(module)}
        for name, child in module._modules.items():
            if child is None:
                continue
            child_path = f"{path}.{name}" if path else name
            if id(child) in next_ancestors:
                if len(records) >= MAX_MODULE_PATHS:
                    raise ValueError("注册模块路径超过 10000 条上限。")
                records.append((child_path, child))
            else:
                visit(child, child_path, next_ancestors)

    visit(model, "", frozenset())
    return records


def _layer_attributes(module: Any, nn: Any) -> dict[str, Any]:
    if isinstance(module, nn.Linear):
        return {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return {
            "in_channels": module.in_channels, "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size), "stride": list(module.stride),
            "padding": list(module.padding) if isinstance(module.padding, tuple) else module.padding,
            "dilation": list(module.dilation), "groups": module.groups, "bias": module.bias is not None,
        }
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return {"num_features": module.num_features, "eps": module.eps, "momentum": module.momentum,
                "affine": module.affine, "track_running_stats": module.track_running_stats}
    if isinstance(module, nn.LayerNorm):
        return {"normalized_shape": list(module.normalized_shape), "eps": module.eps,
                "elementwise_affine": module.elementwise_affine}
    if isinstance(module, nn.GroupNorm):
        return {"num_groups": module.num_groups, "num_channels": module.num_channels,
                "eps": module.eps, "affine": module.affine}
    if isinstance(module, nn.Embedding):
        return {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim,
                "padding_idx": module.padding_idx}
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
        return {"p": module.p, "inplace": module.inplace}
    if isinstance(module, nn.MultiheadAttention):
        return {"embed_dim": module.embed_dim, "num_heads": module.num_heads,
                "head_dim": module.head_dim, "dropout": module.dropout,
                "batch_first": module.batch_first, "bias": module.in_proj_bias is not None}
    if isinstance(module, (nn.RNN, nn.GRU, nn.LSTM)):
        return {"input_size": module.input_size, "hidden_size": module.hidden_size,
                "num_layers": module.num_layers, "bias": module.bias,
                "batch_first": module.batch_first, "dropout": module.dropout,
                "bidirectional": module.bidirectional}
    return {}


def _module_tree(named: list[tuple[str, Any]], nn: Any) -> tuple[list[dict[str, Any]], list[list[str]]]:
    aliases: dict[int, list[str]] = defaultdict(list)
    for path, module in named:
        aliases[id(module)].append(_path(path))
    records = []
    for path, module in named:
        children = [f"{path}.{name}" if path else name for name, child in module._modules.items() if child is not None]
        shared_paths = aliases[id(module)]
        records.append({
            "module_path": _path(path),
            "module_name": path.rsplit(".", 1)[-1] if path else type(module).__name__,
            "module_type": type(module).__name__, "python_module": type(module).__module__,
            "parent_module_path": _path(path.rpartition(".")[0]) if path else None,
            "child_module_paths": children, "depth": path.count(".") + 1 if path else 0,
            "is_leaf": not children, "shared_module_instance": len(shared_paths) > 1,
            "shared_module_paths": shared_paths if len(shared_paths) > 1 else [],
            "executed_in_forward": None, "structural_attributes": _layer_attributes(module, nn),
        })
    return records, [paths for paths in aliases.values() if len(paths) > 1]


def _parameter_role(name: str, owner: Any, nn: Any) -> str:
    if name.rsplit(".", 1)[-1] == "bias":
        return "bias"
    if isinstance(owner, nn.Embedding):
        return "embedding"
    if isinstance(owner, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
                          nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return "normalization"
    return "weight_or_other"


def _parameters(named: list[tuple[str, Any]], nn: Any) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    aliases: dict[int, list[str]] = defaultdict(list)
    objects: dict[int, Any] = {}
    owners: dict[str, Any] = {}
    count_paths = 0
    for module_path, module in named:
        for local_name, parameter in module._parameters.items():
            if parameter is None:
                continue
            count_paths += 1
            if count_paths > MAX_PARAMETER_PATHS:
                raise ValueError("注册参数路径超过 50000 条上限。")
            path = f"{module_path}.{local_name}" if module_path else local_name
            aliases[id(parameter)].append(path)
            objects[id(parameter)] = parameter
            owners[path] = module
    records: list[dict[str, Any]] = []
    shared: list[dict[str, Any]] = []
    by_dtype: Counter[str] = Counter()
    by_top: Counter[str] = Counter()
    total = trainable = 0
    for identity, paths in aliases.items():
        parameter = objects[identity]
        canonical = paths[0]
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
        dtype = str(parameter.dtype).removeprefix("torch.")
        by_dtype[dtype] += count
        top = canonical.split(".", 1)[0] if "." in canonical else "<root>"
        by_top[top] += count
        records.append({
            "parameter_path": canonical,
            "owner_module_path": _path(canonical.rpartition(".")[0]),
            "parameter_name": canonical.rsplit(".", 1)[-1],
            "shape": list(parameter.shape), "parameter_count": count,
            "dtype": dtype, "device": str(parameter.device),
            "requires_grad": bool(parameter.requires_grad), "frozen": not parameter.requires_grad,
            "shared_or_tied": len(paths) > 1, "alias_paths": paths[1:],
            "role": _parameter_role(canonical, owners[canonical], nn),
            "is_quantized": bool(getattr(parameter, "is_quantized", False)),
        })
        if len(paths) > 1:
            shared.append({"canonical_parameter_path": canonical, "all_paths": paths})
    summary = {
        "total_parameter_count": total, "trainable_parameter_count": trainable,
        "frozen_parameter_count": total - trainable,
        "unique_parameter_tensor_count": len(records),
        "dtype_element_counts": dict(sorted(by_dtype.items())),
        "parameter_count_by_top_level_module": dict(sorted(by_top.items())),
    }
    return records, summary, shared


def _buffers(named: list[tuple[str, Any]]) -> list[dict[str, Any]]:
    records = []
    seen: set[int] = set()
    for module_path, module in named:
        for name, buffer in module._buffers.items():
            if buffer is None or id(buffer) in seen:
                continue
            seen.add(id(buffer))
            path = f"{module_path}.{name}" if module_path else name
            records.append({"buffer_path": path, "shape": list(buffer.shape),
                            "dtype": str(buffer.dtype).removeprefix("torch."),
                            "device": str(buffer.device),
                            "persistent": name not in module._non_persistent_buffers_set})
    return records


def _family(named: list[tuple[str, Any]], nn: Any) -> tuple[str, dict[str, Any]]:
    unique = list({id(module): module for _, module in named}.values())
    conv = sum(isinstance(item, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) for item in unique)
    recurrent = sum(isinstance(item, (nn.RNN, nn.GRU, nn.LSTM)) for item in unique)
    attention = sum(isinstance(item, nn.MultiheadAttention) for item in unique)
    linear = sum(isinstance(item, nn.Linear) for item in unique)
    embedding = sum(isinstance(item, nn.Embedding) for item in unique)
    transformer_named = any("transformer" in type(item).__name__.lower() or "attention" in type(item).__name__.lower() for item in unique)
    patch_named = any(
        any(token in f"{path}.{type(module).__name__}".lower() for token in
            ("patch_embed", "patch_projection", "patchembed"))
        for path, module in named
    )
    root_transformer = transformer_named and any("transformer" in type(item).__name__.lower() or "attention" in type(item).__name__.lower() for path, item in named if not path)
    direct = {name: parameter for name, parameter in named[0][1]._parameters.items() if parameter is not None}
    direct_weight = direct.get("weight")
    direct_bias = direct.get("bias")
    direct_affine = bool(
        direct_weight is not None and direct_weight.ndim <= 2
        and (direct_bias is None or direct_bias.ndim <= 1)
        and set(direct).issubset({"weight", "bias"})
    )
    if (root_transformer or (transformer_named and patch_named)) and conv <= 1 and not recurrent:
        family = "Transformer"
    elif sum(bool(value) for value in (conv, recurrent, attention or transformer_named)) > 1:
        family = "Hybrid"
    elif attention or transformer_named:
        family = "Transformer"
    elif conv:
        family = "CNN"
    elif recurrent:
        family = "RNN"
    elif linear >= 2:
        family = "MLP"
    elif linear == 1:
        family = "Linear"
    elif direct_affine:
        family = "Linear"
    else:
        family = "Unknown"
    return family, {
        "conv_module_count": conv, "recurrent_module_count": recurrent,
        "attention_module_count": attention, "linear_module_count": linear,
        "embedding_module_count": embedding,
        "transformer_or_attention_name_detected": transformer_named,
        "patch_projection_name_detected": patch_named,
        "direct_affine_parameter_pattern": direct_affine,
        "classification_is_heuristic": True,
    }


def analyze_model_structure(model: Any, *, load_spec_id: str, checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return structure facts for the live target model; never call forward()."""
    import torch

    if not isinstance(model, torch.nn.Module):
        raise TypeError("结构解析目标不是 torch.nn.Module。")
    named = _registered_modules(model)
    modules, shared_modules = _module_tree(named, torch.nn)
    parameters, summary, shared_parameters = _parameters(named, torch.nn)
    buffers = _buffers(named)
    family, family_evidence = _family(named, torch.nn)
    quantized_modules = [record["module_path"] for record in modules
                         if "quantized" in record["python_module"].lower() or "quantization" in record["python_module"].lower()]
    quantized_parameters = [record["parameter_path"] for record in parameters if record["is_quantized"]]
    checkpoint_loaded = checkpoint["status"] == "loaded"
    missing_keys = checkpoint.get("missing_keys", [])
    unexpected_keys = checkpoint.get("unexpected_keys", [])
    return {
        "report_type": "ModelStructureAnalysisReport", "schema_version": "0.2",
        "status": "passed", "load_spec_id": load_spec_id,
        "source_model_state": {
            "checkpoint_status": checkpoint["status"],
            "checkpoint_parameters_loaded": checkpoint_loaded,
            "checkpoint_fully_matched": checkpoint_loaded and not missing_keys and not unexpected_keys,
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "general": {
            "framework": "pytorch", "framework_version": torch.__version__,
            "model_class_name": type(model).__name__, "model_python_module": type(model).__module__,
            "training_mode_at_analysis_start": bool(model.training),
            "model_family": family, "model_family_evidence": family_evidence,
        },
        "module_tree": modules, "parameters": parameters,
        "parameter_summary": summary, "buffers": buffers,
        "sharing_and_quantization": {
            "shared_module_groups": shared_modules,
            "shared_parameter_groups": shared_parameters,
            "quantized_module_paths": quantized_modules,
            "quantized_parameter_paths": quantized_parameters,
            "quantization_indicators_found": bool(quantized_modules or quantized_parameters),
        },
        "forward_trace": {"status": "not_run", "reason": "未提供技术示例输入；未执行 forward。"},
    }
