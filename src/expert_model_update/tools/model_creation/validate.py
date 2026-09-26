"""Validate one explicitly declared model delivery without project discovery."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPORT_TYPE = "ModelDeliveryValidationReport"
REPORT_VERSION = "0.1"
TASK_INPUT_VERSION = "0.3"
CLASS_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SELECTOR_PATTERN = re.compile(r"^\$(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
MODEL_FIELDS = {
    "model_file": "assets.base_model.model_file",
    "config": "assets.base_model.config",
    "checkpoint": "assets.base_model.checkpoint",
}


@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    status: str
    message: str
    details: dict[str, Any] | None = None


class DeliveryValidationError(ValueError):
    pass


def _nonempty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_digest(task_id: str, version: int, assets: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"task_id": task_id, "task_input_version": version, "assets": assets},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _resolved_check(preflight: dict[str, Any], field: str) -> Path:
    checks = preflight.get("checks")
    matches = [
        item for item in checks
        if isinstance(item, dict) and item.get("field") == field
    ] if isinstance(checks, list) else []
    if len(matches) != 1 or matches[0].get("status") != "passed":
        raise DeliveryValidationError(f"资产预检没有唯一通过的 {field} 检查。")
    value = _nonempty(matches[0].get("resolved_path"))
    if value is None:
        raise DeliveryValidationError(f"资产预检没有返回 {field} 的规范路径。")
    return Path(value).resolve(strict=True)


def _bind_inputs(payload: Any) -> tuple[dict[str, Any], dict[str, Path]]:
    required = {
        "task_input",
        "asset_preflight_report",
        "config_inspection_report",
        "checkpoint_inspection_report",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise DeliveryValidationError("必须提供 TaskInput、资产预检、配置检查和 checkpoint 检查四项输入。")
    task = payload["task_input"]
    preflight = payload["asset_preflight_report"]
    if not isinstance(task, dict) or task.get("schema_version") != TASK_INPUT_VERSION:
        raise DeliveryValidationError("只接受 TaskInput v0.3。")
    task_id = _nonempty(task.get("task_id"))
    version = task.get("task_input_version")
    assets = task.get("assets")
    if task_id is None or isinstance(version, bool) or not isinstance(version, int) or version < 1 or not isinstance(assets, dict):
        raise DeliveryValidationError("TaskInput 的任务标识、版本或 assets 无效。")
    goal = task.get("request", {}).get("update_goal") if isinstance(task.get("request"), dict) else None
    confirmation = task.get("intake", {}).get("human_confirmation") if isinstance(task.get("intake"), dict) else None
    if (
        not isinstance(goal, dict)
        or goal.get("status") != "confirmed"
        or not isinstance(confirmation, dict)
        or confirmation.get("status") != "confirmed"
        or confirmation.get("confirmed_task_input_version") != version
    ):
        raise DeliveryValidationError("TaskInput 尚未确认，或确认版本与当前版本不一致。")
    binding = {
        "task_id": task_id,
        "task_input_version": version,
        "asset_input_digest": _asset_digest(task_id, version, assets),
    }
    if not isinstance(preflight, dict) or preflight.get("report_type") != "AssetPreflightReport" or preflight.get("overall_status") != "passed":
        raise DeliveryValidationError("必须提供已经通过的 AssetPreflightReport。")
    for key, value in binding.items():
        if preflight.get(key) != value:
            raise DeliveryValidationError(f"AssetPreflightReport 的 {key} 与 TaskInput 不一致。")
    paths = {name: _resolved_check(preflight, field) for name, field in MODEL_FIELDS.items()}
    for name, report_key in (("config", "config_inspection_report"), ("checkpoint", "checkpoint_inspection_report")):
        report = payload[report_key]
        if not isinstance(report, dict) or report.get("report_type") != "DeclaredModelAssetInspectionReport":
            raise DeliveryValidationError(f"{name} 检查报告结构无效。")
        if report.get("asset_kind") != name or report.get("status") != "inspected":
            raise DeliveryValidationError(f"{name} 尚未得到成功检查。")
        if report.get("task_binding") != binding:
            raise DeliveryValidationError(f"{name} 检查报告与当前任务绑定不一致。")
        source = report.get("source")
        if not isinstance(source, dict) or Path(str(source.get("path"))).resolve(strict=True) != paths[name]:
            raise DeliveryValidationError(f"{name} 检查报告的源文件与预检路径不一致。")
        digest = _nonempty(source.get("sha256"))
        if digest is None or digest != _sha256(paths[name]):
            raise DeliveryValidationError(f"{name} 文件在检查后发生变化。")
    return binding, paths


def _validate_config(values: Any, findings: list[Finding]) -> dict[str, Any] | None:
    path = "$.config"
    if not isinstance(values, dict):
        findings.append(Finding("CONFIG_NOT_OBJECT", path, "failed", "模型配置必须是对象。"))
        return None
    expected = {"schema_version", "framework", "model_class", "constructor_kwargs", "checkpoint"}
    missing = expected - values.keys()
    unknown = values.keys() - expected
    for key in sorted(missing):
        findings.append(Finding("CONFIG_FIELD_MISSING", f"{path}.{key}", "failed", "缺少必填配置字段。"))
    for key in sorted(unknown):
        findings.append(Finding("CONFIG_FIELD_UNKNOWN", f"{path}.{key}", "failed", "存在未定义的配置字段。"))
    if missing:
        return None
    if values.get("schema_version") != "0.1":
        findings.append(Finding("CONFIG_VERSION_UNSUPPORTED", f"{path}.schema_version", "failed", "只支持模型加载配置 v0.1。"))
    if values.get("framework") != "pytorch":
        findings.append(Finding("FRAMEWORK_UNSUPPORTED", f"{path}.framework", "failed", "当前只支持 pytorch。"))
    class_name = values.get("model_class")
    if not isinstance(class_name, str) or not CLASS_PATTERN.fullmatch(class_name):
        findings.append(Finding("MODEL_CLASS_INVALID", f"{path}.model_class", "failed", "model_class 必须是简单 Python 标识符。"))
    kwargs = values.get("constructor_kwargs")
    if not isinstance(kwargs, dict) or not all(isinstance(key, str) and key.isidentifier() for key in kwargs):
        findings.append(Finding("CONSTRUCTOR_KWARGS_INVALID", f"{path}.constructor_kwargs", "failed", "constructor_kwargs 必须是以 Python 标识符为键的对象。"))
    checkpoint = values.get("checkpoint")
    if not isinstance(checkpoint, dict):
        findings.append(Finding("CHECKPOINT_CONFIG_INVALID", f"{path}.checkpoint", "failed", "checkpoint 配置必须是对象。"))
    else:
        allowed = {"state_dict_path", "key_transforms", "strict"}
        if set(checkpoint) != allowed:
            findings.append(Finding("CHECKPOINT_CONFIG_FIELDS_INVALID", f"{path}.checkpoint", "failed", "checkpoint 配置必须且只能包含 state_dict_path、key_transforms、strict。"))
        selector = checkpoint.get("state_dict_path")
        if not isinstance(selector, str) or not SELECTOR_PATTERN.fullmatch(selector):
            findings.append(Finding("STATE_DICT_PATH_INVALID", f"{path}.checkpoint.state_dict_path", "failed", "state_dict_path 必须是 $ 或 $.name 形式。"))
        if checkpoint.get("strict") is not True:
            findings.append(Finding("STRICT_LOAD_REQUIRED", f"{path}.checkpoint.strict", "failed", "v0.1 要求 strict=true，以证明权重完全匹配。"))
        transforms = checkpoint.get("key_transforms")
        if not isinstance(transforms, list):
            findings.append(Finding("KEY_TRANSFORMS_INVALID", f"{path}.checkpoint.key_transforms", "failed", "key_transforms 必须是数组。"))
        else:
            for index, item in enumerate(transforms):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"operation", "value"}
                    or item.get("operation") not in {"strip_prefix", "add_prefix"}
                    or not _nonempty(item.get("value"))
                ):
                    findings.append(Finding("KEY_TRANSFORM_INVALID", f"{path}.checkpoint.key_transforms[{index}]", "failed", "键转换必须声明 strip_prefix/add_prefix 和非空 value。"))
    if any(item.status == "failed" for item in findings):
        return None
    return values


def _class_node(tree: ast.Module, class_name: str) -> ast.ClassDef | None:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name]
    return matches[0] if len(matches) == 1 else None


def _validate_constructor(model_path: Path, class_name: str, kwargs: dict[str, Any], findings: list[Finding]) -> str | None:
    try:
        raw = model_path.read_bytes()
        source = raw.decode("utf-8-sig")
        tree = ast.parse(source, filename=str(model_path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        findings.append(Finding("MODEL_FILE_PARSE_FAILED", "$.model_file", "failed", f"模型文件无法静态解析：{type(exc).__name__}: {exc}"))
        return None
    node = _class_node(tree, class_name)
    if node is None:
        findings.append(Finding("MODEL_CLASS_NOT_UNIQUE", "$.config.model_class", "failed", "模型文件顶层不存在唯一的声明模型类。"))
        return None
    init = next((item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__"), None)
    if init is None:
        if kwargs:
            findings.append(Finding("CONSTRUCTOR_REJECTS_KWARGS", "$.config.constructor_kwargs", "failed", "模型类没有自定义 __init__，但配置提供了构造参数。"))
            return None
    else:
        args = init.args
        positional = [*args.posonlyargs, *args.args]
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        keyword_only = list(args.kwonlyargs)
        accepted = {item.arg for item in [*positional, *keyword_only]}
        if args.kwarg is None:
            unknown = sorted(set(kwargs) - accepted)
            for name in unknown:
                findings.append(Finding("CONSTRUCTOR_ARGUMENT_UNKNOWN", f"$.config.constructor_kwargs.{name}", "failed", "该参数不在模型类构造函数签名中。"))
        defaults_start = len(positional) - len(args.defaults)
        required = {
            item.arg for index, item in enumerate(positional)
            if index < defaults_start and item not in args.posonlyargs
        }
        required.update(
            item.arg for item, default in zip(keyword_only, args.kw_defaults)
            if default is None
        )
        missing = sorted(required - set(kwargs))
        for name in missing:
            findings.append(Finding("CONSTRUCTOR_ARGUMENT_MISSING", f"$.config.constructor_kwargs.{name}", "failed", "模型类构造函数的必需参数没有配置值。"))
        required_positional_only = [
            item.arg for index, item in enumerate(args.posonlyargs)
            if not (index == 0 and item.arg in {"self", "cls"})
            and index < max(0, len(args.posonlyargs) - len(args.defaults))
        ]
        if required_positional_only:
            findings.append(Finding("POSITIONAL_ONLY_CONSTRUCTOR_UNSUPPORTED", "$.config.constructor_kwargs", "failed", "v0.1 不支持必须按位置传入的构造参数。", {"parameters": required_positional_only}))
    if any(item.status == "failed" for item in findings):
        return None
    findings.append(Finding("MODEL_CLASS_DECLARED", "$.config.model_class", "passed", "模型文件中存在唯一声明类，构造参数与静态签名一致。"))
    return hashlib.sha256(raw).hexdigest()


def _validate_checkpoint_evidence(report: dict[str, Any], selector: str, findings: list[Finding]) -> dict[str, Any] | None:
    checkpoint_report = report.get("inspection")
    if not isinstance(checkpoint_report, dict) or checkpoint_report.get("report_type") != "CheckpointInspectionReport":
        findings.append(Finding("CHECKPOINT_REPORT_INVALID", "$.checkpoint_inspection_report", "failed", "checkpoint 检查报告结构无效。"))
        return None
    candidates = checkpoint_report.get("state_dict_candidates")
    matches = [
        item for item in candidates
        if isinstance(item, dict) and item.get("selector_path") == selector
    ] if isinstance(candidates, list) else []
    if len(matches) != 1:
        findings.append(Finding("STATE_DICT_PATH_NOT_FOUND", "$.config.checkpoint.state_dict_path", "failed", "声明路径没有唯一对应 checkpoint 中的张量 state_dict。"))
        return None
    findings.append(Finding("STATE_DICT_PATH_CONFIRMED", "$.config.checkpoint.state_dict_path", "passed", "checkpoint 检查证据确认了声明的 state_dict 路径。", {"tensor_count": matches[0].get("tensor_count")}))
    return matches[0]


def validate_model_delivery(payload: Any, *, base_directory: Path | None = None) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        binding, paths = _bind_inputs(payload)
        config_wrapper = payload["config_inspection_report"]
        config_report = config_wrapper.get("inspection")
        values = config_report.get("values") if isinstance(config_report, dict) else None
        config = _validate_config(values, findings)
        model_digest = None
        candidate = None
        if config is not None:
            model_digest = _validate_constructor(paths["model_file"], config["model_class"], config["constructor_kwargs"], findings)
            candidate = _validate_checkpoint_evidence(
                payload["checkpoint_inspection_report"],
                config["checkpoint"]["state_dict_path"],
                findings,
            )
    except (DeliveryValidationError, FileNotFoundError, OSError, RuntimeError) as exc:
        binding = None
        paths = {}
        config = None
        model_digest = None
        candidate = None
        findings.append(Finding("INPUT_BINDING_FAILED", "$", "failed", str(exc)))

    ready = (
        binding is not None
        and config is not None
        and model_digest is not None
        and candidate is not None
        and not any(item.status == "failed" for item in findings)
    )
    plan = None
    if ready:
        config_source = payload["config_inspection_report"]["source"]
        checkpoint_source = payload["checkpoint_inspection_report"]["source"]
        core = {
            "task_binding": binding,
            "model": {
                "path": str(paths["model_file"]),
                "sha256": model_digest,
                "class_name": config["model_class"],
                "constructor_kwargs": config["constructor_kwargs"],
            },
            "config": {
                "path": str(paths["config"]),
                "sha256": config_source["sha256"],
            },
            "checkpoint": {
                "path": str(paths["checkpoint"]),
                "sha256": checkpoint_source["sha256"],
                "selector_path": config["checkpoint"]["state_dict_path"],
                "key_transforms": config["checkpoint"]["key_transforms"],
                "strict": True,
                "inspected_tensor_count": candidate.get("tensor_count"),
            },
        }
        load_spec_id = hashlib.sha256(
            json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        plan = {"load_spec_id": load_spec_id, **core}
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_VERSION,
        "task_binding": binding,
        "status": "validated" if ready else "failed",
        "findings": [asdict(item) for item in findings],
        "validated_load_plan": plan,
        "readiness": {
            "ready_for_model_load": ready,
            "ready_for_structure_analysis": ready,
            "blockers": [] if ready else [item.code for item in findings if item.status == "failed"],
        },
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="验证上游声明的模型文件、构造配置与 checkpoint。")
    parser.add_argument("--stdin", action="store_true", required=True)
    parser.add_argument("--output", choices=("json",), default="json")
    parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        report = validate_model_delivery(payload)
    except Exception as exc:
        report = {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_VERSION,
            "task_binding": None,
            "status": "execution_error",
            "findings": [],
            "validated_load_plan": None,
            "readiness": {"ready_for_model_load": False, "ready_for_structure_analysis": False, "blockers": ["execution_error"]},
            "execution_error": True,
            "error": {"code": "EXECUTION_ERROR", "message": f"{type(exc).__name__}: {exc}"},
        }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["readiness"]["ready_for_model_load"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
