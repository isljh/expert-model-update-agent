"""Inspect checkpoint structure without importing or executing the target project.

This worker is intentionally invoked in a separate Python process.  It uses
``torch.load(weights_only=True, map_location="cpu")`` and never falls back to
ordinary pickle loading when the safe mode is unavailable or rejects a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPORT_TYPE = "CheckpointInspectionReport"
REPORT_SCHEMA_VERSION = "0.1"
MAX_REPORTED_KEYS = 20_000
MAX_TRAVERSAL_DEPTH = 4


def _selector_child(path: str, key: Any) -> str | None:
    if not isinstance(key, str) or not key.isidentifier():
        return None
    return f"{path}.{key}"


def _tensor_metadata(torch: Any, value: Any) -> dict[str, Any] | None:
    if not torch.is_tensor(value):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "layout": str(value.layout),
        "requires_grad": bool(getattr(value, "requires_grad", False)),
    }


def _state_dict_summary(torch: Any, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not value:
        return None

    tensor_items: list[tuple[str, Any]] = []
    non_tensor_keys: list[str] = []
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        if torch.is_tensor(item):
            tensor_items.append((key, item))
        else:
            non_tensor_keys.append(key)

    if not tensor_items or non_tensor_keys:
        return None

    truncated = len(tensor_items) > MAX_REPORTED_KEYS
    selected = tensor_items[:MAX_REPORTED_KEYS]
    keys = [key for key, _ in selected]
    uniform_prefix = None
    for candidate in ("module.", "model.", "network.", "_orig_mod."):
        if keys and all(key.startswith(candidate) for key in keys):
            uniform_prefix = candidate
            break

    return {
        "tensor_count": len(tensor_items),
        "reported_tensor_count": len(selected),
        "truncated": truncated,
        "uniform_prefix": uniform_prefix,
        "tensors": {
            key: _tensor_metadata(torch, item)
            for key, item in selected
        },
    }


def _find_state_dict_candidates(
    torch: Any,
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
) -> list[dict[str, Any]]:
    summary = _state_dict_summary(torch, value)
    if summary is not None:
        return [{"selector_path": path, **summary}]
    if depth >= MAX_TRAVERSAL_DEPTH or not isinstance(value, Mapping):
        return []

    candidates: list[dict[str, Any]] = []
    for key, item in value.items():
        child = _selector_child(path, key)
        if child is None:
            continue
        candidates.extend(
            _find_state_dict_candidates(
                torch,
                item,
                path=child,
                depth=depth + 1,
            )
        )
    return candidates


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime image
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "unavailable",
            "safe_loading": {
                "weights_only": True,
                "map_location": "cpu",
                "unsafe_fallback_used": False,
            },
            "error": {
                "code": "TORCH_UNAVAILABLE",
                "message": f"无法导入 PyTorch：{exc}",
            },
            "top_level_type": None,
            "top_level_keys": [],
            "state_dict_candidates": [],
        }

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "unavailable",
            "safe_loading": {
                "weights_only": True,
                "map_location": "cpu",
                "unsafe_fallback_used": False,
            },
            "error": {
                "code": "WEIGHTS_ONLY_UNSUPPORTED",
                "message": f"当前 PyTorch 不支持安全加载模式：{exc}",
            },
            "top_level_type": None,
            "top_level_keys": [],
            "state_dict_candidates": [],
        }
    except Exception as exc:
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "failed",
            "safe_loading": {
                "weights_only": True,
                "map_location": "cpu",
                "unsafe_fallback_used": False,
            },
            "error": {
                "code": "SAFE_LOAD_FAILED",
                "message": f"checkpoint 安全加载失败：{type(exc).__name__}: {exc}",
            },
            "top_level_type": None,
            "top_level_keys": [],
            "state_dict_candidates": [],
        }

    top_level_keys = []
    if isinstance(loaded, Mapping):
        top_level_keys = [str(key) for key in list(loaded.keys())[:MAX_REPORTED_KEYS]]
    candidates = _find_state_dict_candidates(torch, loaded)
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "inspected",
        "safe_loading": {
            "weights_only": True,
            "map_location": "cpu",
            "unsafe_fallback_used": False,
        },
        "error": None,
        "top_level_type": f"{type(loaded).__module__}.{type(loaded).__name__}",
        "top_level_keys": top_level_keys,
        "state_dict_candidates": candidates,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="安全检查 PyTorch checkpoint 的结构和张量元数据。")
    parser.add_argument("checkpoint", type=Path)
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    report = inspect_checkpoint(args.checkpoint)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "inspected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
