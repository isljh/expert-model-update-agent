"""Load one validated model delivery in a bounded local Python subprocess.

This process is not an OS sandbox. Importing the supplied model file and
calling its constructor can still have side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Mapping
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any


MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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


def _assert_snapshot(spec: dict[str, Any], label: str) -> Path:
    try:
        path = Path(spec["path"]).resolve(strict=True)
    except (KeyError, OSError, RuntimeError) as exc:
        raise InstantiationFailure("ASSET_PATH_INVALID", f"{label} 路径无效。") from exc
    if not path.is_file():
        raise InstantiationFailure("ASSET_NOT_FILE", f"{label} 不是普通文件。")
    if _sha256(path) != spec.get("sha256"):
        raise InstantiationFailure("ASSET_SNAPSHOT_CHANGED", f"{label} 在验证后发生变化。", {"path": str(path)})
    return path


def _import_model_file(path: Path, load_spec_id: str) -> Any:
    module_name = f"_delivered_model_{load_spec_id[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise InstantiationFailure("MODEL_IMPORTER_UNAVAILABLE", "无法为声明模型文件创建导入器。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _select_checkpoint_value(loaded: Any, path: str) -> Any:
    if path == "$":
        return loaded
    value = loaded
    for key in path[2:].split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise InstantiationFailure("CHECKPOINT_SELECTOR_NOT_FOUND", "checkpoint 中不存在声明的 state_dict 路径。", {"selector_path": path})
        value = value[key]
    return value


def _transform_keys(state_dict: Any, transforms: list[dict[str, str]], torch: Any) -> dict[str, Any]:
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise InstantiationFailure("STATE_DICT_INVALID", "选择结果不是非空 state_dict。")
    result: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise InstantiationFailure("STATE_DICT_INVALID", "state_dict 必须只包含字符串键和张量值。")
        output_key = key
        for transform in transforms:
            operation = transform["operation"]
            prefix = transform["value"]
            if operation == "strip_prefix":
                if not output_key.startswith(prefix):
                    raise InstantiationFailure("KEY_PREFIX_MISMATCH", "权重键不具有声明的剥离前缀。", {"key": key, "prefix": prefix})
                output_key = output_key[len(prefix):]
            elif operation == "add_prefix":
                output_key = prefix + output_key
            else:
                raise InstantiationFailure("KEY_TRANSFORM_INVALID", "不支持的键转换操作。")
        if output_key in result:
            raise InstantiationFailure("KEY_TRANSFORM_COLLISION", "键转换产生重复键。", {"key": output_key})
        result[output_key] = value
    return result


def _load_checkpoint(model: Any, spec: dict[str, Any], path: Path, torch: Any) -> dict[str, Any]:
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise InstantiationFailure("CHECKPOINT_TOO_LARGE", "checkpoint 超过当前 2 GiB 安全上限。")
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise InstantiationFailure("WEIGHTS_ONLY_UNSUPPORTED", "当前 PyTorch 不支持安全权重加载；不会退回普通 pickle。") from exc
    except Exception as exc:
        raise InstantiationFailure("SAFE_CHECKPOINT_LOAD_FAILED", f"安全读取 checkpoint 失败：{type(exc).__name__}: {exc}") from exc
    state_dict = _transform_keys(
        _select_checkpoint_value(loaded, spec["selector_path"]),
        spec["key_transforms"],
        torch,
    )
    inspected_count = spec.get("inspected_tensor_count")
    if isinstance(inspected_count, int) and inspected_count != len(state_dict):
        raise InstantiationFailure(
            "CHECKPOINT_TENSOR_COUNT_CHANGED",
            "实际加载的权重张量数与检查证据不一致。",
            {"inspected": inspected_count, "actual": len(state_dict)},
        )
    try:
        incompatible = model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise InstantiationFailure("STATE_DICT_LOAD_FAILED", f"模型拒绝该 state_dict：{type(exc).__name__}: {exc}") from exc
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing or unexpected:
        raise InstantiationFailure("STATE_DICT_NOT_FULLY_MATCHED", "权重加载后仍有 missing 或 unexpected keys。", {"missing_keys": missing, "unexpected_keys": unexpected})
    return {
        "status": "loaded",
        "path": str(path),
        "selector_path": spec["selector_path"],
        "strict": True,
        "tensor_count": len(state_dict),
        "missing_keys": [],
        "unexpected_keys": [],
        "fully_matched": True,
        "weights_only": True,
        "map_location": "cpu",
    }


def instantiate(plan: dict[str, Any], *, analyze_structure: bool = False) -> dict[str, Any]:
    import torch

    load_spec_id = plan.get("load_spec_id")
    result: dict[str, Any] = {
        "report_type": "ModelInstantiationReport",
        "schema_version": "0.2",
        "load_spec_id": load_spec_id,
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
        "status": "failed",
        "steps": [],
        "checkpoint": {"status": "not_attempted"},
        "model": None,
        "structure_analysis": None,
        "error": None,
    }
    parent_added: str | None = None
    try:
        if not isinstance(load_spec_id, str) or not re.fullmatch(r"[a-f0-9]{64}", load_spec_id):
            raise InstantiationFailure("LOAD_SPEC_ID_INVALID", "加载计划标识无效。")
        model_path = _assert_snapshot(plan["model"], "模型文件")
        _assert_snapshot(plan["config"], "构造配置")
        checkpoint_path = _assert_snapshot(plan["checkpoint"], "checkpoint")
        result["steps"].append({"name": "asset_snapshots_verified", "status": "completed"})

        parent_added = str(model_path.parent)
        sys.path.insert(0, parent_added)
        module = _import_model_file(model_path, load_spec_id)
        class_name = plan["model"]["class_name"]
        if not SYMBOL_PATTERN.fullmatch(class_name):
            raise InstantiationFailure("MODEL_CLASS_INVALID", "模型类名格式无效。")
        model_class = getattr(module, class_name, None)
        if not isinstance(model_class, type):
            raise InstantiationFailure("MODEL_CLASS_NOT_FOUND", "声明模型类在运行时不存在。")
        try:
            model = model_class(**plan["model"]["constructor_kwargs"])
        except Exception as exc:
            raise InstantiationFailure("MODEL_CONSTRUCTION_FAILED", f"模型类构造失败：{type(exc).__name__}: {exc}") from exc
        if not isinstance(model, torch.nn.Module):
            raise InstantiationFailure("TARGET_NOT_TORCH_MODULE", "声明模型类没有产生 torch.nn.Module。")
        result["steps"].append({"name": "model_instantiated", "status": "completed", "type": type(model).__name__})

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        result["model"] = {
            "model_file": str(model_path),
            "class_name": type(model).__name__,
            "python_module": type(model).__module__,
            "target_is_torch_module": True,
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_count,
            "has_trainable_parameters": trainable_count > 0,
        }
        result["checkpoint"] = _load_checkpoint(model, plan["checkpoint"], checkpoint_path, torch)
        result["steps"].append({"name": "checkpoint_loaded_and_fully_matched", "status": "completed"})

        if analyze_structure:
            try:
                try:
                    from .structure import analyze_model_structure
                except ImportError:
                    from structure import analyze_model_structure
                result["structure_analysis"] = analyze_model_structure(
                    model,
                    load_spec_id=load_spec_id,
                    checkpoint=result["checkpoint"],
                )
            except Exception as exc:
                raise InstantiationFailure("STRUCTURE_ANALYSIS_FAILED", f"权重已加载，但结构解析失败：{type(exc).__name__}: {exc}") from exc
        result["status"] = "completed"
    except InstantiationFailure as exc:
        result["error"] = {"code": exc.code, "message": str(exc), "details": exc.details}
        if result["model"] is not None and result["checkpoint"]["status"] == "not_attempted":
            result["checkpoint"] = {"status": "failed", "reason_code": exc.code}
    except Exception as exc:
        result["error"] = {"code": "DELIVERED_MODEL_EXECUTION_FAILED", "message": f"{type(exc).__name__}: {exc}"}
    finally:
        if parent_added is not None and sys.path and sys.path[0] == parent_added:
            sys.path.pop(0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="加载已验证的模型交付物；可在同一进程解析结构。")
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
            "schema_version": "0.2",
            "load_spec_id": plan.get("load_spec_id") if isinstance(plan, dict) else None,
            "runtime": {"python_executable": sys.executable, "python_version": sys.version.split()[0], "torch_version": None},
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
