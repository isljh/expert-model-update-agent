"""Execute one validated construction plan in a local Python subprocess.

This module is not a sandbox. Importing and calling target project code can
write files, access the network, or start other processes.
"""

from __future__ import annotations

import hashlib
import argparse
import importlib
import importlib.util
import json
import re
import sys
from contextlib import redirect_stdout
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ALLOWED_EXTERNAL = {
    ("torch.nn", "DataParallel"),
    ("torch.nn.parallel", "DistributedDataParallel"),
}


class InstantiationFailure(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_project_file(root: Path, path_text: str, expected_digest: str) -> None:
    path = Path(path_text).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise InstantiationFailure("SOURCE_OUTSIDE_PROJECT", "源码快照越过目标项目边界。") from exc
    if not path.is_file() or _sha256(path) != expected_digest:
        raise InstantiationFailure("SOURCE_SNAPSHOT_CHANGED", "创建计划使用的项目源码已变化。", {"path": str(path)})


def _check_source_snapshots(plan: dict[str, Any], root: Path) -> None:
    for entry in plan["project"]["step_sources"]:
        anchor = entry["construction_anchor_source"]
        _assert_project_file(root, anchor["path"], anchor["sha256"])
        definition = entry.get("callable_source")
        if definition is not None:
            _assert_project_file(root, definition["path"], definition["sha256"])
    checkpoint = plan.get("checkpoint_load_attempt")
    if checkpoint is not None:
        anchor = checkpoint["source_anchor_snapshot"]
        _assert_project_file(root, anchor["path"], anchor["sha256"])


def _project_module_path(root: Path, module_name: str) -> Path:
    if not MODULE_PATTERN.fullmatch(module_name):
        raise InstantiationFailure("INVALID_MODULE_NAME", "项目模块名格式无效。")
    parts = module_name.split(".")
    candidates = [root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts, "__init__.py")]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise InstantiationFailure("MODULE_OUTSIDE_PROJECT", "项目模块越过目标项目边界。") from exc
            return resolved
    raise InstantiationFailure("PROJECT_MODULE_NOT_FOUND", "找不到计划声明的项目模块。", {"module": module_name})


def _import_project_module(root: Path, module_name: str) -> Any:
    path = _project_module_path(root, module_name)
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None or Path(existing_path).resolve() != path:
            raise InstantiationFailure("MODULE_NAME_COLLISION", "模块名已被非目标项目模块占用。", {"module": module_name})
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise InstantiationFailure("PROJECT_MODULE_IMPORT_FAILED", "无法创建项目模块导入器。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _callable(root: Path, reference: dict[str, str]) -> Any:
    module_name, symbol = reference["module"], reference["symbol"]
    if not SYMBOL_PATTERN.fullmatch(symbol):
        raise InstantiationFailure("INVALID_SYMBOL", "callable 符号格式无效。")
    if reference["origin"] == "project":
        module = _import_project_module(root, module_name)
    elif reference["origin"] == "pytorch" and (module_name, symbol) in ALLOWED_EXTERNAL:
        module = importlib.import_module(module_name)
    else:
        raise InstantiationFailure("CALLABLE_NOT_APPROVED", "计划中的外部 callable 未获准执行。")
    value = getattr(module, symbol, None)
    if not callable(value):
        raise InstantiationFailure("CALLABLE_NOT_FOUND", "计划声明的 callable 在运行时不存在。", {"module": module_name, "symbol": symbol})
    return value


def _object_ref(objects: dict[str, Any], reference: dict[str, str]) -> Any:
    step_id = reference["step_id"]
    if step_id not in objects:
        raise InstantiationFailure("OBJECT_REF_MISSING", "对象引用指向尚未创建的步骤。", {"step_id": step_id})
    value = objects[step_id]
    path = reference["path"]
    if path == "$":
        return value
    for part in path[2:].split("."):
        if not SYMBOL_PATTERN.fullmatch(part):
            raise InstantiationFailure("OBJECT_PATH_INVALID", "对象路径格式无效。")
        value = getattr(value, part)
    return value


def _argument(root: Path, objects: dict[str, Any], item: dict[str, Any]) -> Any:
    value = item["value"]
    kind = value["kind"]
    if kind == "literal":
        return value["value"]
    if kind == "object_ref":
        return _object_ref(objects, value["object"])
    if kind == "symbol":
        return getattr(_import_project_module(root, value["module"]), value["symbol"])
    raise InstantiationFailure("ARGUMENT_KIND_INVALID", "计划包含不支持的实参类型。")


def _arguments(root: Path, objects: dict[str, Any], spec: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    positional = [_argument(root, objects, item) for item in spec["positional_arguments"]]
    keywords = {item["parameter"]: _argument(root, objects, item) for item in spec["keyword_arguments"]}
    return positional, keywords


def _apply_operations(root: Path, objects: dict[str, Any], operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = []
    for index, operation in enumerate(operations):
        receiver = _object_ref(objects, operation["target"])
        method = getattr(receiver, operation["method"])
        if not callable(method):
            raise InstantiationFailure("OPERATION_NOT_CALLABLE", "生命周期操作在运行时不可调用。", {"index": index})
        positional, keywords = _arguments(root, objects, operation)
        method(*positional, **keywords)
        completed.append({"index": index, "method": operation["method"], "target": operation["target"]})
    return completed


def _select_checkpoint_value(loaded: Any, path: str | None) -> Any:
    if path is None:
        raise InstantiationFailure("CHECKPOINT_SELECTOR_UNRESOLVED", "加载指令未指定 state_dict 选择路径。")
    if path == "$":
        return loaded
    value = loaded
    for key in path[2:].split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise InstantiationFailure("CHECKPOINT_SELECTOR_NOT_FOUND", "checkpoint 中不存在声明的选择路径。", {"selector_path": path})
        value = value[key]
    return value


def _transform_keys(state_dict: Any, transforms: list[dict[str, str]], torch: Any) -> dict[str, Any]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InstantiationFailure("STATE_DICT_INVALID", "选择结果不是非空映射。")
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise InstantiationFailure("STATE_DICT_INVALID", "state_dict 必须只包含字符串键和张量值。")
        output_key = key
        for transform in transforms:
            if transform["operation"] == "strip_prefix":
                prefix = transform["value"]
                if not output_key.startswith(prefix):
                    raise InstantiationFailure("KEY_PREFIX_MISMATCH", "某个权重键不具有声明的剥离前缀。", {"key": key, "prefix": prefix})
                output_key = output_key[len(prefix):]
            elif transform["operation"] == "add_prefix":
                output_key = transform["value"] + output_key
            else:
                raise InstantiationFailure("KEY_TRANSFORM_INVALID", "不支持的键转换操作。")
        if output_key in result:
            raise InstantiationFailure("KEY_TRANSFORM_COLLISION", "键转换产生重复键。", {"key": output_key})
        result[output_key] = value
    return result


def _load_checkpoint(objects: dict[str, Any], recipe: dict[str, Any], torch: Any) -> dict[str, Any]:
    path = Path(recipe["path"]).resolve(strict=True)
    if not path.is_file():
        raise InstantiationFailure("CHECKPOINT_NOT_FILE", "checkpoint 不是普通文件。")
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise InstantiationFailure("CHECKPOINT_TOO_LARGE", "checkpoint 超过当前 2 GiB 安全上限。")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise InstantiationFailure("WEIGHTS_ONLY_UNSUPPORTED", "当前 PyTorch 不支持安全权重加载；不会退回普通 pickle。") from exc
    except Exception as exc:
        raise InstantiationFailure("SAFE_CHECKPOINT_LOAD_FAILED", f"安全读取 checkpoint 失败：{type(exc).__name__}: {exc}") from exc
    state_dict = _transform_keys(_select_checkpoint_value(loaded, recipe["selector_path"]), recipe["key_transforms"], torch)
    target = _object_ref(objects, recipe["load_target"])
    if not isinstance(target, torch.nn.Module):
        raise InstantiationFailure("LOAD_TARGET_NOT_MODULE", "权重加载目标不是 torch.nn.Module。")
    try:
        incompatible = target.load_state_dict(state_dict, strict=recipe["strict"])
    except Exception as exc:
        raise InstantiationFailure("STATE_DICT_LOAD_FAILED", f"模型拒绝该 state_dict：{type(exc).__name__}: {exc}") from exc
    return {
        "status": "loaded",
        "path": str(path),
        "selector_path": recipe["selector_path"],
        "load_target": recipe["load_target"],
        "strict": recipe["strict"],
        "tensor_count": len(state_dict),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "weights_only": True,
        "map_location": "cpu",
    }


def instantiate(plan: dict[str, Any], *, analyze_structure: bool = False) -> dict[str, Any]:
    import torch

    objects: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "report_type": "ModelInstantiationReport",
        "schema_version": "0.1",
        "proposal_id": plan["proposal_id"],
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
        "status": "failed",
        "steps": steps,
        "checkpoint": {"status": "not_attempted"},
        "model": None,
        "structure_analysis": None,
        "error": None,
    }
    root: Path | None = None
    try:
        root = Path(plan["project"]["source_project_path"]).resolve(strict=True)
        _check_source_snapshots(plan, root)
        sys.path.insert(0, str(root))
        for step in plan["construction_steps"]:
            callable_value = _callable(root, step["callable"])
            positional, keywords = _arguments(root, objects, step)
            objects[step["step_id"]] = callable_value(*positional, **keywords)
            steps.append({"step_id": step["step_id"], "status": "created", "type": type(objects[step["step_id"]]).__name__})
        _apply_operations(root, objects, plan["pre_checkpoint_load_operations"])
        target_model = _object_ref(objects, plan["result"]["target_model"])
        runtime_object = _object_ref(objects, plan["result"]["runtime_object"])
        if not isinstance(target_model, torch.nn.Module):
            raise InstantiationFailure("TARGET_NOT_TORCH_MODULE", "目标模型不是 torch.nn.Module。")
        result["model"] = {
            "target_type": type(target_model).__name__,
            "target_module": type(target_model).__module__,
            "runtime_type": type(runtime_object).__name__,
            "target_is_torch_module": True,
            "parameter_count": sum(parameter.numel() for parameter in target_model.parameters()),
        }
        recipe = plan.get("checkpoint_load_attempt")
        if recipe is not None:
            result["checkpoint"] = _load_checkpoint(objects, recipe, torch)
        else:
            result["checkpoint"] = {"status": "not_provided"}
        _apply_operations(root, objects, plan["post_checkpoint_load_operations"])
        if analyze_structure:
            try:
                try:
                    from .structure import analyze_model_structure
                except ImportError:  # direct worker entry
                    from structure import analyze_model_structure
                result["structure_analysis"] = analyze_model_structure(
                    target_model, proposal_id=plan["proposal_id"], checkpoint=result["checkpoint"]
                )
            except Exception as exc:
                raise InstantiationFailure(
                    "STRUCTURE_ANALYSIS_FAILED",
                    f"模型已创建，但结构解析失败：{type(exc).__name__}: {exc}",
                ) from exc
        result["status"] = "completed"
    except InstantiationFailure as exc:
        result["error"] = {"code": exc.code, "message": str(exc), "details": exc.details}
        if result["model"] is not None and result["checkpoint"]["status"] == "not_attempted":
            result["checkpoint"] = {"status": "failed", "reason_code": exc.code}
    except Exception as exc:
        result["error"] = {"code": "PROJECT_EXECUTION_FAILED", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if root is not None and sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="执行已验证模型创建方案；可在同一进程解析结构。")
    parser.add_argument("--analyze-structure", action="store_true")
    args = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    plan: Any = None
    try:
        plan = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            report = instantiate(plan, analyze_structure=args.analyze_structure)
    except Exception as exc:
        report = {
            "report_type": "ModelInstantiationReport",
            "schema_version": "0.1",
            "proposal_id": plan.get("proposal_id") if isinstance(plan, dict) else None,
            "runtime": {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "torch_version": None,
            },
            "status": "failed",
            "steps": [],
            "checkpoint": {"status": "not_attempted"},
            "model": None,
            "structure_analysis": None,
            "error": {"code": "WORKER_FAILED", "message": f"{type(exc).__name__}: {exc}"},
        }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
