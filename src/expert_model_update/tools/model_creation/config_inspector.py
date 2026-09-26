"""Safely parse a human-declared model configuration file as data.

The inspector never imports the target project and never executes configuration
code.  Executable Python configs and unsupported formats are rejected.
"""

from __future__ import annotations

import argparse
import configparser
import datetime as datetime_types
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPORT_TYPE = "ConfigInspectionReport"
REPORT_SCHEMA_VERSION = "0.1"
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_NODES = 50_000
MAX_DEPTH = 32


class ConfigInspectionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _reject_json_constant(value: str) -> None:
    raise ConfigInspectionError(
        "NON_FINITE_NUMBER",
        f"配置包含 JSON 标准不允许的数值 {value}。",
    )


def _parse_text(path: Path, text: str) -> tuple[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json", json.loads(text, parse_constant=_reject_json_constant)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise ConfigInspectionError(
                "YAML_PARSER_UNAVAILABLE",
                f"当前 Python 环境没有可用的安全 YAML 解析器：{exc}",
            ) from exc
        return "yaml", yaml.safe_load(text)
    if suffix == ".toml":
        try:
            try:
                import tomllib  # type: ignore[import-not-found]
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found,no-redef]
        except Exception as exc:
            raise ConfigInspectionError(
                "TOML_PARSER_UNAVAILABLE",
                f"当前 Python 环境没有可用的 TOML 解析器：{exc}",
            ) from exc
        return "toml", tomllib.loads(text)
    if suffix in {".ini", ".cfg"}:
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser.read_string(text)
        values: dict[str, Any] = {}
        if parser.defaults():
            values["DEFAULT"] = dict(parser.defaults())
        for section in parser.sections():
            values[section] = dict(parser.items(section, raw=True))
        return "ini", values
    if suffix == ".py":
        raise ConfigInspectionError(
            "EXECUTABLE_CONFIG_UNSUPPORTED",
            "Python 配置文件可能执行代码；只允许把它作为源码证据静态阅读，不能由配置检查器运行。",
        )
    raise ConfigInspectionError(
        "UNSUPPORTED_CONFIG_FORMAT",
        "只支持 JSON、YAML、TOML、INI 和 CFG 数据配置文件。",
    )


def _normalize(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> Any:
    if depth > MAX_DEPTH:
        raise ConfigInspectionError("CONFIG_TOO_DEEP", f"配置嵌套深度超过 {MAX_DEPTH}。")
    active_counter = counter if counter is not None else [0]
    active_counter[0] += 1
    if active_counter[0] > MAX_NODES:
        raise ConfigInspectionError("CONFIG_TOO_LARGE", f"配置值节点超过 {MAX_NODES}。")

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigInspectionError("NON_FINITE_NUMBER", "配置包含非有限浮点数。")
        return value
    if isinstance(value, (datetime_types.datetime, datetime_types.date, datetime_types.time)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [
            _normalize(item, depth=depth + 1, counter=active_counter)
            for item in value
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigInspectionError(
                    "NON_STRING_KEY",
                    "配置映射包含非字符串键，无法形成稳定的机器证据。",
                )
            normalized[key] = _normalize(
                item,
                depth=depth + 1,
                counter=active_counter,
            )
        return normalized
    raise ConfigInspectionError(
        "UNSUPPORTED_CONFIG_VALUE",
        f"配置包含不支持的值类型：{type(value).__module__}.{type(value).__name__}。",
    )


def _error_report(
    path: Path,
    status: str,
    code: str,
    message: str,
    *,
    size_bytes: int | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "source": {
            "path": str(path),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "format": None,
        },
        "values": None,
        "error": {"code": code, "message": message},
    }


def inspect_config(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            return _error_report(
                resolved,
                "failed",
                "CONFIG_NOT_FILE",
                "声明的模型配置路径不是普通文件。",
            )
        size_bytes = resolved.stat().st_size
        if size_bytes > MAX_FILE_BYTES:
            return _error_report(
                resolved,
                "failed",
                "CONFIG_FILE_TOO_LARGE",
                f"配置文件超过 {MAX_FILE_BYTES} 字节安全上限。",
                size_bytes=size_bytes,
            )
        raw = resolved.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig")
        config_format, parsed = _parse_text(resolved, text)
        values = _normalize(parsed)
    except FileNotFoundError:
        return _error_report(path, "failed", "CONFIG_NOT_FOUND", "配置文件不存在。")
    except UnicodeDecodeError as exc:
        return _error_report(
            path,
            "failed",
            "CONFIG_NOT_UTF8",
            f"配置文件不是有效 UTF-8 文本：{exc}",
        )
    except ConfigInspectionError as exc:
        return _error_report(
            path,
            "unavailable" if exc.code.endswith("_UNAVAILABLE") else "failed",
            exc.code,
            str(exc),
        )
    except (json.JSONDecodeError, configparser.Error, ValueError, OSError) as exc:
        return _error_report(
            path,
            "failed",
            "CONFIG_PARSE_FAILED",
            f"配置解析失败：{type(exc).__name__}: {exc}",
        )

    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "inspected",
        "source": {
            "path": str(resolved),
            "size_bytes": size_bytes,
            "sha256": digest,
            "format": config_format,
        },
        "values": values,
        "error": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="安全解析人类声明的模型配置文件。")
    parser.add_argument("config", type=Path)
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    report = inspect_config(parse_args().config)
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["status"] == "inspected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
