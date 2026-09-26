"""Inspect only model assets declared by TaskInput and authorized by preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .checkpoint_inspector import inspect_checkpoint
    from .config_inspector import inspect_config
except ImportError:  # direct script execution
    from checkpoint_inspector import inspect_checkpoint
    from config_inspector import inspect_config


REPORT_TYPE = "DeclaredModelAssetInspectionReport"
REPORT_SCHEMA_VERSION = "0.1"
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
ASSET_FIELDS = {
    "config": "assets.base_model.config",
    "checkpoint": "assets.base_model.checkpoint",
}


class BindingError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _source_metadata(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    if not path.is_file():
        return {
            "path": str(path),
            "size_bytes": metadata.st_size,
            "sha256": None,
        }
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "sha256": digest.hexdigest(),
    }


def _nonempty(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _asset_digest(task_id: str, version: int, assets: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"task_id": task_id, "task_input_version": version, "assets": assets},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _bind_declared_path(
    task_input: Any,
    preflight: Any,
    asset_kind: str,
) -> tuple[dict[str, Any], Path, str]:
    if asset_kind not in ASSET_FIELDS:
        raise BindingError("INVALID_ASSET_KIND", "asset_kind 只能是 config 或 checkpoint。")
    if not isinstance(task_input, dict) or not isinstance(preflight, dict):
        raise BindingError("INVALID_INPUT", "TaskInput 和 AssetPreflightReport 都必须是对象。")

    task_id = _nonempty(task_input.get("task_id"))
    version = task_input.get("task_input_version")
    assets = task_input.get("assets")
    if task_id is None or isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise BindingError("INVALID_TASK_BINDING", "TaskInput 缺少有效任务标识或版本。")
    if not isinstance(assets, dict):
        raise BindingError("INVALID_ASSETS", "TaskInput 缺少 assets 对象。")

    request = task_input.get("request")
    update_goal = request.get("update_goal") if isinstance(request, dict) else None
    intake = task_input.get("intake")
    confirmation = intake.get("human_confirmation") if isinstance(intake, dict) else None
    if (
        not isinstance(update_goal, dict)
        or update_goal.get("status") != "confirmed"
        or not isinstance(confirmation, dict)
        or confirmation.get("status") != "confirmed"
        or confirmation.get("confirmed_task_input_version") != version
    ):
        raise BindingError(
            "TASK_INPUT_NOT_CONFIRMED",
            "模型资产证据检查只接受已经确认且确认版本与当前版本一致的 TaskInput。",
        )

    digest = _asset_digest(task_id, version, assets)
    if preflight.get("report_type") != "AssetPreflightReport" or preflight.get("overall_status") != "passed":
        raise BindingError("PREFLIGHT_NOT_PASSED", "必须提供已通过的 AssetPreflightReport。")
    expected = {
        "task_id": task_id,
        "task_input_version": version,
        "asset_input_digest": digest,
    }
    for key, value in expected.items():
        if preflight.get(key) != value:
            raise BindingError("PREFLIGHT_BINDING_MISMATCH", f"AssetPreflightReport 的 {key} 与 TaskInput 不一致。")

    base_model = assets.get("base_model")
    base_model = base_model if isinstance(base_model, dict) else {}
    declared = _nonempty(base_model.get(asset_kind))
    if declared is None:
        raise BindingError("ASSET_NOT_DECLARED", f"TaskInput 没有人类声明的 {asset_kind} 路径。")

    field = ASSET_FIELDS[asset_kind]
    checks = preflight.get("checks")
    matching = [
        item
        for item in checks if isinstance(item, dict) and item.get("field") == field
    ] if isinstance(checks, list) else []
    if len(matching) != 1 or matching[0].get("status") != "passed":
        raise BindingError("ASSET_NOT_AUTHORIZED", f"资产预检没有唯一通过的 {field} 检查。")
    resolved_text = _nonempty(matching[0].get("resolved_path"))
    if resolved_text is None:
        raise BindingError("ASSET_PATH_MISSING", f"资产预检没有返回 {field} 的规范路径。")
    resolved = Path(resolved_text).resolve(strict=True)

    base_text = _nonempty(preflight.get("base_directory"))
    base = Path(base_text).resolve(strict=True) if base_text is not None else Path.cwd().resolve(strict=True)
    declared_path = Path(declared).expanduser()
    if not declared_path.is_absolute():
        declared_path = base / declared_path
    if declared_path.resolve(strict=True) != resolved:
        raise BindingError("DECLARED_PATH_MISMATCH", f"TaskInput 的 {field} 与资产预检规范路径不一致。")

    return expected, resolved, field


def inspect_declared_asset(
    task_input: Any,
    preflight: Any,
    asset_kind: str,
) -> dict[str, Any]:
    try:
        binding, path, field = _bind_declared_path(task_input, preflight, asset_kind)
        checkpoint_too_large = (
            asset_kind == "checkpoint"
            and path.is_file()
            and path.stat().st_size > MAX_CHECKPOINT_BYTES
        )
        source = (
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": None,
            }
            if checkpoint_too_large
            else _source_metadata(path)
        )
        if asset_kind == "config":
            inspection = inspect_config(path)
        else:
            if not path.is_file():
                inspection = {
                    "report_type": "CheckpointInspectionReport",
                    "schema_version": "0.1",
                    "status": "unavailable",
                    "safe_loading": {
                        "weights_only": True,
                        "map_location": "cpu",
                        "unsafe_fallback_used": False,
                    },
                    "error": {
                        "code": "CHECKPOINT_DIRECTORY_UNSUPPORTED",
                        "message": "当前检查器只支持单个 checkpoint 文件；目录式或分片格式仍待实现。",
                    },
                    "top_level_type": None,
                    "top_level_keys": [],
                    "state_dict_candidates": [],
                }
            elif checkpoint_too_large:
                inspection = {
                    "report_type": "CheckpointInspectionReport",
                    "schema_version": "0.1",
                    "status": "unavailable",
                    "safe_loading": {
                        "weights_only": True,
                        "map_location": "cpu",
                        "unsafe_fallback_used": False,
                    },
                    "error": {
                        "code": "CHECKPOINT_FILE_TOO_LARGE",
                        "message": f"checkpoint 超过 {MAX_CHECKPOINT_BYTES} 字节安全上限。",
                    },
                    "top_level_type": None,
                    "top_level_keys": [],
                    "state_dict_candidates": [],
                }
            else:
                inspection = inspect_checkpoint(path)
    except BindingError as exc:
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "asset_kind": asset_kind,
            "task_binding": None,
            "asset_field": ASSET_FIELDS.get(asset_kind),
            "source": None,
            "status": "binding_failed",
            "inspection": None,
            "error": {"code": exc.code, "message": str(exc)},
        }
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "asset_kind": asset_kind,
            "task_binding": None,
            "asset_field": ASSET_FIELDS.get(asset_kind),
            "source": None,
            "status": "binding_failed",
            "inspection": None,
            "error": {"code": "ASSET_BINDING_IO_ERROR", "message": str(exc)},
        }

    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "asset_kind": asset_kind,
        "task_binding": binding,
        "asset_field": field,
        "source": source,
        "status": inspection.get("status"),
        "inspection": inspection,
        "error": inspection.get("error"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 TaskInput 声明且经预检授权的模型资产。")
    parser.add_argument("--stdin", action="store_true", dest="read_stdin")
    parser.add_argument("--asset-kind", choices=("config", "checkpoint"), required=True)
    args = parser.parse_args()
    if not args.read_stdin:
        parser.error("必须通过 --stdin 提供包含 task_input 和 asset_preflight_report 的 JSON。")
    return args


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise BindingError("INVALID_INPUT", "工具输入必须是 JSON 对象。")
        report = inspect_declared_asset(
            payload.get("task_input"),
            payload.get("asset_preflight_report"),
            args.asset_kind,
        )
    except json.JSONDecodeError as exc:
        report = {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "asset_kind": args.asset_kind,
            "task_binding": None,
            "asset_field": ASSET_FIELDS[args.asset_kind],
            "source": None,
            "status": "execution_error",
            "inspection": None,
            "error": {"code": "INVALID_JSON", "message": str(exc)},
            "execution_error": True,
        }
    except BindingError as exc:
        report = {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "asset_kind": args.asset_kind,
            "task_binding": None,
            "asset_field": ASSET_FIELDS[args.asset_kind],
            "source": None,
            "status": "execution_error",
            "inspection": None,
            "error": {"code": exc.code, "message": str(exc)},
            "execution_error": True,
        }
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    if report.get("execution_error"):
        return 2
    return 0 if report.get("status") == "inspected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
