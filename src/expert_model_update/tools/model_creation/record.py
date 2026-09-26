"""Persist verified model-analysis results outside the model delivery directory.

Only the host calls this module. The worker that imports delivered model code
never receives the record directory or writes the formal report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class RecordWriteError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _inline(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_structure_tree(report: dict[str, Any]) -> str:
    """Render every registered module and parameter without running the model."""
    structure = report.get("structure_analysis")
    binding = report.get("task_binding") or {}
    delivery = report.get("delivery") or {}
    checkpoint = report.get("checkpoint") or {}
    model = report.get("model") or {}
    lines = [
        "# 模型结构解析运行报告",
        "",
        f"- 任务：{_inline(binding.get('task_id'))}，TaskInput v{binding.get('task_input_version', '?')}",
        f"- 加载计划：{_inline(report.get('load_spec_id'))}",
        f"- 模型文件：{_inline(delivery.get('model_file'))}",
        f"- 构造配置：{_inline(delivery.get('config'))}",
        f"- checkpoint：{_inline(delivery.get('checkpoint'))}",
        f"- 本次执行：{_inline(report.get('status'))}；checkpoint：{_inline(checkpoint.get('status'))}",
        f"- 模型类：{_inline(model.get('class_name'))}；总参数：{_inline(model.get('parameter_count'))}；可训练参数：{_inline(model.get('trainable_parameter_count'))}",
        "",
        "本报告由工具返回的结构化结果生成；不是完整前向计算图。工具未主动调用训练或评估入口；导入模型文件和调用构造函数仍可能有副作用。",
        "",
    ]
    if not isinstance(structure, dict) or structure.get("status") != "passed":
        lines.extend([
            "## 结构解析状态",
            "",
            "本次没有可用的完整结构树。",
            f"错误：{_inline(report.get('error'))}",
            "",
        ])
        return "\n".join(lines)

    general = structure["general"]
    summary = structure["parameter_summary"]
    state = structure["source_model_state"]
    modules = structure["module_tree"]
    parameters = structure["parameters"]
    buffers = structure["buffers"]
    lines.extend([
        "## 总览",
        "",
        f"- 模型类：{_inline(str(general.get('model_python_module')) + '.' + str(general.get('model_class_name')))}",
        f"- 家族（启发式）：{_inline(general.get('model_family'))}",
        f"- 注册模块路径：{len(modules)}；唯一参数张量：{len(parameters)}；唯一缓冲区：{len(buffers)}",
        f"- 参数元素：总计 {summary.get('total_parameter_count')}，可训练 {summary.get('trainable_parameter_count')}，冻结 {summary.get('frozen_parameter_count')}",
        f"- checkpoint 已加载：{_inline(state.get('checkpoint_parameters_loaded'))}；加载后无不兼容键：{_inline(state.get('checkpoint_fully_matched'))}",
        f"- checkpoint 选择路径：{_inline(checkpoint.get('selector_path'))}；严格加载：{_inline(checkpoint.get('strict'))}；权重张量数：{_inline(checkpoint.get('tensor_count'))}",
        f"- 加载缺失键：{_inline(state.get('missing_keys', []))}；多余键：{_inline(state.get('unexpected_keys', []))}",
        f"- 前向轨迹：{_inline(structure.get('forward_trace', {}).get('status'))}",
        "",
        "## 完整注册模块树",
        "",
        "以下逐层列出报告中的全部模块路径；函数式操作和未注册对象不在此树中。",
        "",
        "```text",
    ])
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for item in modules:
        children[item.get("parent_module_path")].append(item)

    rendered: set[str] = set()

    def label(item: dict[str, Any]) -> str:
        attributes = item.get("structural_attributes") or {}
        suffix = f" {_inline(attributes)}" if attributes else ""
        shared = " [共享模块]" if item.get("shared_module_instance") else ""
        return f"{_inline(item['module_path'])}  {item['module_type']}{suffix}{shared}"

    def visit(parent: str, prefix: str) -> None:
        siblings = children.get(parent, [])
        for index, item in enumerate(siblings):
            path = item["module_path"]
            if path in rendered:
                continue
            rendered.add(path)
            last = index == len(siblings) - 1
            lines.append(f"{prefix}{'└─' if last else '├─'} {label(item)}")
            visit(path, prefix + ("   " if last else "│  "))

    roots = children.get(None, [])
    for root in roots:
        rendered.add(root["module_path"])
        lines.append(label(root))
        visit(root["module_path"], "")
    if len(rendered) != len(modules):
        # Keep orphaned/duplicate paths visible rather than silently dropping them.
        for item in modules:
            if item["module_path"] not in rendered:
                lines.append(f"[未连接] {label(item)}")
                rendered.add(item["module_path"])
    lines.extend(["```", "", "## 参数张量清单", "", "```text"])
    for item in parameters:
        lines.append(
            f"{_inline(item['parameter_path'])} shape={_inline(item['shape'])} "
            f"count={item['parameter_count']} dtype={item['dtype']} "
            f"requires_grad={_inline(item['requires_grad'])} aliases={_inline(item['alias_paths'])}"
        )
    lines.extend(["```", "", "## 缓冲区清单", "", "```text"])
    for item in buffers:
        lines.append(
            f"{_inline(item['buffer_path'])} shape={_inline(item['shape'])} "
            f"dtype={item['dtype']} persistent={_inline(item['persistent'])}"
        )
    sharing = structure["sharing_and_quantization"]
    lines.extend([
        "```", "", "## 共享与量化迹象", "",
        f"- 共享模块组：{_inline(sharing.get('shared_module_groups', []))}",
        f"- 共享参数组：{_inline(sharing.get('shared_parameter_groups', []))}",
        f"- 量化模块路径：{_inline(sharing.get('quantized_module_paths', []))}",
        f"- 量化参数路径：{_inline(sharing.get('quantized_parameter_paths', []))}",
        "",
    ])
    return "\n".join(lines)


def save_model_analysis_record(
    report: dict[str, Any], *, record_root: Path, model_delivery_root: Path
) -> dict[str, Any]:
    """Create a new record without accepting output paths from the load plan or worker."""
    root = record_root.resolve(strict=False)
    target = model_delivery_root.resolve(strict=True)
    if root == target or target in root.parents:
        raise RecordWriteError("RECORD_ROOT_IN_MODEL_DELIVERY", "运行记录目录位于模型交付目录内，拒绝写入。")
    delivery_name = target.name or "model-delivery"
    delivery_dir = (root / delivery_name).resolve(strict=False)
    if root not in delivery_dir.parents or delivery_dir == target or target in delivery_dir.parents:
        raise RecordWriteError("RECORD_DIRECTORY_INVALID", "模型运行记录目录不在允许的智能体目录内，拒绝写入。")
    try:
        delivery_dir.mkdir(parents=True, exist_ok=True)
        recorded_at = datetime.now(timezone.utc)
        run_id = recorded_at.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex
        json_path = delivery_dir / f"{run_id}.model-instantiation-and-structure-report.json"
        tree_path = delivery_dir / f"{run_id}.model-structure-tree.md"
        record = {
            "run_id": run_id,
            "recorded_at_utc": recorded_at.isoformat(),
            "delivery_name": delivery_name,
            "json_report_path": str(json_path),
            "structure_tree_path": str(tree_path),
        }
        saved_report = {**report, "record": record}
        tree = render_structure_tree(saved_report)
        pending_tree = delivery_dir / f".{run_id}.model-structure-tree.md.pending"
        pending_json = delivery_dir / f".{run_id}.model-instantiation-and-structure-report.json.pending"
        with pending_tree.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(tree)
        pending_tree.rename(tree_path)
        with pending_json.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(saved_report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        pending_json.rename(json_path)
        return saved_report
    except OSError as exc:
        raise RecordWriteError("REPORT_WRITE_FAILED", f"无法保存模型分析运行记录：{type(exc).__name__}: {exc}") from exc
