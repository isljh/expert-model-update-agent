"""Validate Agent-proposed model creation specifications without executing project code.

The validator is deliberately proposal-scoped.  It does not discover models or
rank model roles.  It checks the ordered callable graph, argument values and
object references, runtime/target objects, declared lifecycle operations, and
checkpoint recipe supplied by the upstream project-understanding Agent.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence


REPORT_TYPE = "ModelCreationValidationReport"
REPORT_SCHEMA_VERSION = "0.2"
PROPOSAL_TYPE = "ModelCreationProposalSet"
PROPOSAL_SCHEMA_VERSION = "0.2"
OBJECT_PATH_PATTERN = re.compile(r"^\$(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
GIT_REVISION_PATTERN = re.compile(r"^[a-fA-F0-9]{40}$")
APPROVED_PYTORCH_CALLABLES = {
    ("torch.nn", "DataParallel"),
    ("torch.nn.parallel", "DistributedDataParallel"),
}
KNOWN_RESULT_PATHS = {
    ("torch.nn", "DataParallel"): {"$", "$.module"},
    ("torch.nn.parallel", "DistributedDataParallel"): {"$", "$.module"},
}
PYTORCH_WRAPPER_MODULE_PARAMETER = {
    ("torch.nn", "DataParallel"): "module",
    ("torch.nn.parallel", "DistributedDataParallel"): "module",
}

CheckStatus = Literal["passed", "failed", "unresolved", "warning"]


@dataclass(frozen=True)
class Finding:
    code: str
    field: str
    status: CheckStatus
    message: str
    evidence: dict[str, Any] | None = None


class InputContractError(ValueError):
    def __init__(self, findings: list[Finding]) -> None:
        super().__init__("模型创建方案不符合输入契约。")
        self.findings = findings


@dataclass
class ParsedModule:
    module: str
    path: Path
    source: str
    tree: ast.Module
    sha256: str


@dataclass
class CallableDefinition:
    module: ParsedModule
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    signature_node: ast.FunctionDef | ast.AsyncFunctionDef
    kind: Literal["function", "class"]


class SourceRepository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._module_cache: dict[str, ParsedModule] = {}
        self._path_cache: dict[Path, ParsedModule] = {}

    def confined_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("源码锚点必须使用项目根目录下的相对路径。")
        resolved = (self.root / candidate).resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("源码锚点越过了项目根目录。") from exc
        if not resolved.is_file():
            raise ValueError("源码锚点不是普通文件。")
        return resolved

    def module_path(self, module: str) -> Path:
        parts = module.split(".")
        file_candidate = self.root.joinpath(*parts).with_suffix(".py")
        package_candidate = self.root.joinpath(*parts, "__init__.py")
        selected = None
        if file_candidate.is_file():
            selected = file_candidate
        elif package_candidate.is_file():
            selected = package_candidate
        if selected is not None:
            resolved = selected.resolve(strict=True)
            try:
                resolved.relative_to(self.root)
            except ValueError as exc:
                raise PermissionError(
                    f"模块 {module!r} 通过符号链接越过了项目根目录。"
                ) from exc
            return resolved
        raise FileNotFoundError(f"找不到模块 {module!r} 对应的 Python 文件。")

    def parse_module(self, module: str) -> ParsedModule:
        cached = self._module_cache.get(module)
        if cached is not None:
            return cached
        path = self.module_path(module)
        parsed = self.parse_path(path, module=module)
        self._module_cache[module] = parsed
        return parsed

    def parse_relative_path(self, relative_path: str) -> ParsedModule:
        path = self.confined_path(relative_path)
        return self.parse_path(path, module=self.module_name_for_path(path))

    def parse_path(self, path: Path, *, module: str) -> ParsedModule:
        cached = self._path_cache.get(path)
        if cached is not None:
            return cached
        raw_source = path.read_bytes()
        source = raw_source.decode("utf-8")
        tree = ast.parse(source, filename=str(path))
        digest = hashlib.sha256(raw_source).hexdigest()
        parsed = ParsedModule(module, path, source, tree, digest)
        self._path_cache[path] = parsed
        return parsed

    def module_name_for_path(self, path: Path) -> str:
        relative = path.relative_to(self.root)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem
        return ".".join(parts)

    def find_callable(self, module: str, symbol: str) -> CallableDefinition:
        parsed = self.parse_module(module)
        for node in parsed.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol:
                return CallableDefinition(parsed, node, node, "function")
            if isinstance(node, ast.ClassDef) and node.name == symbol:
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__":
                        return CallableDefinition(parsed, node, child, "class")
                synthetic = ast.FunctionDef(
                    name="__init__",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[ast.arg(arg="self")],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[],
                    decorator_list=[],
                )
                return CallableDefinition(parsed, node, synthetic, "class")
        raise LookupError(f"模块 {module!r} 中不存在顶层类或函数 {symbol!r}。")


def _finding(
    code: str,
    field: str,
    status: CheckStatus,
    message: str,
    evidence: dict[str, Any] | None = None,
) -> Finding:
    return Finding(code, field, status, message, evidence)


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _check_keys(
    value: Any,
    path: str,
    required: Iterable[str],
    allowed: Iterable[str],
    findings: list[Finding],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        findings.append(_finding("TYPE", path, "failed", "应为 JSON 对象。"))
        return None
    required_set = set(required)
    allowed_set = set(allowed)
    for key in sorted(required_set - value.keys()):
        findings.append(_finding("REQUIRED", f"{path}.{key}", "failed", "缺少必填字段。"))
    for key in sorted(value.keys() - allowed_set):
        findings.append(_finding("UNKNOWN", f"{path}.{key}", "failed", "存在未定义字段。"))
    return value


def _check_string(
    value: Any,
    path: str,
    findings: list[Finding],
    *,
    pattern: re.Pattern[str] | None = None,
    nullable: bool = False,
) -> None:
    if value is None and nullable:
        return
    text = _nonempty_string(value)
    if text is None:
        findings.append(_finding("TYPE", path, "failed", "应为非空字符串。"))
        return
    if pattern is not None and pattern.fullmatch(text) is None:
        findings.append(_finding("FORMAT", path, "failed", "字符串格式不符合约定。"))


def _validate_object_ref(value: Any, path: str, findings: list[Finding]) -> None:
    obj = _check_keys(value, path, ("step_id", "path"), ("step_id", "path"), findings)
    if obj is None:
        return
    _check_string(obj.get("step_id"), f"{path}.step_id", findings, pattern=SYMBOL_PATTERN)
    _check_string(obj.get("path"), f"{path}.path", findings, pattern=OBJECT_PATH_PATTERN)


def _validate_callable_ref(value: Any, path: str, findings: list[Finding]) -> None:
    obj = _check_keys(value, path, ("origin", "module", "symbol"), ("origin", "module", "symbol"), findings)
    if obj is None:
        return
    if obj.get("origin") not in {"project", "pytorch"}:
        findings.append(_finding("ENUM", f"{path}.origin", "failed", "origin 只能是 project 或 pytorch。"))
    _check_string(obj.get("module"), f"{path}.module", findings, pattern=MODULE_PATTERN)
    _check_string(obj.get("symbol"), f"{path}.symbol", findings, pattern=SYMBOL_PATTERN)


def _validate_argument_value(value: Any, path: str, findings: list[Finding]) -> None:
    obj = _check_keys(value, path, ("kind",), ("kind", "value", "module", "symbol", "object"), findings)
    if obj is None:
        return
    kind = obj.get("kind")
    if kind == "literal":
        if "value" not in obj:
            findings.append(_finding("REQUIRED", f"{path}.value", "failed", "literal 参数缺少 value。"))
        for key in ("module", "symbol", "object"):
            if key in obj:
                findings.append(_finding("UNKNOWN", f"{path}.{key}", "failed", "literal 参数包含不允许的字段。"))
    elif kind == "symbol":
        _check_string(obj.get("module"), f"{path}.module", findings, pattern=MODULE_PATTERN)
        _check_string(obj.get("symbol"), f"{path}.symbol", findings, pattern=SYMBOL_PATTERN)
        for key in ("value", "object"):
            if key in obj:
                findings.append(_finding("UNKNOWN", f"{path}.{key}", "failed", "symbol 参数包含不允许的字段。"))
    elif kind == "object_ref":
        _validate_object_ref(obj.get("object"), f"{path}.object", findings)
        for key in ("value", "module", "symbol"):
            if key in obj:
                findings.append(_finding("UNKNOWN", f"{path}.{key}", "failed", "object_ref 参数包含不允许的字段。"))
    else:
        findings.append(_finding("ENUM", f"{path}.kind", "failed", "kind 只能是 literal、symbol 或 object_ref。"))


def _validate_arguments(value: Any, path: str, findings: list[Finding], *, keyword: bool) -> None:
    if not isinstance(value, list):
        findings.append(_finding("TYPE", path, "failed", "应为数组。"))
        return
    seen: set[str] = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        required = ("parameter", "value", "source_expression") if keyword else ("value", "source_expression")
        obj = _check_keys(item, item_path, required, required, findings)
        if obj is None:
            continue
        if keyword:
            parameter = obj.get("parameter")
            _check_string(parameter, f"{item_path}.parameter", findings, pattern=SYMBOL_PATTERN)
            if isinstance(parameter, str):
                if parameter in seen:
                    findings.append(_finding("DUPLICATE", f"{item_path}.parameter", "failed", "关键字参数重复。"))
                seen.add(parameter)
        _validate_argument_value(obj.get("value"), f"{item_path}.value", findings)
        if obj.get("source_expression") is not None and not isinstance(obj.get("source_expression"), str):
            findings.append(_finding("TYPE", f"{item_path}.source_expression", "failed", "应为字符串或 null。"))


def _validate_anchor(value: Any, path: str, findings: list[Finding]) -> None:
    obj = _check_keys(value, path, ("path", "scope", "line", "call_expression"), ("path", "scope", "line", "call_expression"), findings)
    if obj is None:
        return
    _check_string(obj.get("path"), f"{path}.path", findings)
    for key in ("scope", "call_expression"):
        if obj.get(key) is not None and not isinstance(obj.get(key), str):
            findings.append(_finding("TYPE", f"{path}.{key}", "failed", "应为字符串或 null。"))
    line = obj.get("line")
    if line is not None and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
        findings.append(_finding("TYPE", f"{path}.line", "failed", "应为不小于 1 的整数或 null。"))


def _validate_evidence_refs(value: Any, path: str, findings: list[Finding]) -> None:
    if not isinstance(value, list) or any(_nonempty_string(item) is None for item in value):
        findings.append(_finding("TYPE", path, "failed", "应为字符串数组。"))


def _validate_operation(value: Any, path: str, findings: list[Finding]) -> None:
    obj = _check_keys(value, path, ("target", "method", "positional_arguments", "keyword_arguments", "source_anchor"), ("target", "method", "positional_arguments", "keyword_arguments", "source_anchor"), findings)
    if obj is None:
        return
    _validate_object_ref(obj.get("target"), f"{path}.target", findings)
    _check_string(obj.get("method"), f"{path}.method", findings, pattern=SYMBOL_PATTERN)
    _validate_arguments(obj.get("positional_arguments"), f"{path}.positional_arguments", findings, keyword=False)
    _validate_arguments(obj.get("keyword_arguments"), f"{path}.keyword_arguments", findings, keyword=True)
    _validate_anchor(obj.get("source_anchor"), f"{path}.source_anchor", findings)


def _validate_checkpoint(value: Any, path: str, findings: list[Finding]) -> None:
    if value is None:
        return
    obj = _check_keys(value, path, ("path", "selector_path", "key_transforms", "load_target", "strict", "source_anchor"), ("path", "selector_path", "key_transforms", "load_target", "strict", "source_anchor"), findings)
    if obj is None:
        return
    _check_string(obj.get("path"), f"{path}.path", findings)
    _check_string(obj.get("selector_path"), f"{path}.selector_path", findings, pattern=OBJECT_PATH_PATTERN, nullable=True)
    _validate_object_ref(obj.get("load_target"), f"{path}.load_target", findings)
    if not isinstance(obj.get("strict"), bool):
        findings.append(_finding("TYPE", f"{path}.strict", "failed", "应为布尔值。"))
    transforms = obj.get("key_transforms")
    if not isinstance(transforms, list):
        findings.append(_finding("TYPE", f"{path}.key_transforms", "failed", "应为数组。"))
    else:
        for index, transform in enumerate(transforms):
            transform_path = f"{path}.key_transforms[{index}]"
            item = _check_keys(transform, transform_path, ("operation", "value"), ("operation", "value"), findings)
            if item is None:
                continue
            if item.get("operation") not in {"strip_prefix", "add_prefix"}:
                findings.append(_finding("ENUM", f"{transform_path}.operation", "failed", "只支持 strip_prefix 或 add_prefix。"))
            _check_string(item.get("value"), f"{transform_path}.value", findings)
    _validate_anchor(obj.get("source_anchor"), f"{path}.source_anchor", findings)


def _iter_argument_object_refs(arguments: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(arguments, list):
        return
    for argument in arguments:
        if not isinstance(argument, dict):
            continue
        value = argument.get("value")
        if isinstance(value, dict) and value.get("kind") == "object_ref" and isinstance(value.get("object"), dict):
            yield value["object"]


def _validate_reference_semantics(proposal: dict[str, Any], path: str, findings: list[Finding]) -> None:
    steps = proposal.get("construction_steps")
    if not isinstance(steps, list):
        return
    positions: dict[str, int] = {}
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or not isinstance(step.get("step_id"), str):
            continue
        step_id = step["step_id"]
        if step_id in positions:
            findings.append(_finding("DUPLICATE", f"{path}.construction_steps[{index}].step_id", "failed", "step_id 重复。"))
        else:
            positions[step_id] = index

    def check(ref: Any, ref_path: str, *, before: int | None = None) -> None:
        if not isinstance(ref, dict) or not isinstance(ref.get("step_id"), str):
            return
        target = positions.get(ref["step_id"])
        if target is None:
            findings.append(_finding("REFERENCE", f"{ref_path}.step_id", "failed", "引用了不存在的 construction step。"))
        elif before is not None and target >= before:
            findings.append(_finding("FORWARD_REFERENCE", f"{ref_path}.step_id", "failed", "创建步骤只能引用已经完成的前序步骤。"))

    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        for name in ("positional_arguments", "keyword_arguments"):
            for ref_index, ref in enumerate(_iter_argument_object_refs(step.get(name))):
                check(ref, f"{path}.construction_steps[{index}].{name}[object_ref:{ref_index}].value.object", before=index)
    result = proposal.get("result")
    if isinstance(result, dict):
        check(result.get("runtime_object"), f"{path}.result.runtime_object")
        check(result.get("target_model"), f"{path}.result.target_model")
    for operations_name in ("pre_checkpoint_load_operations", "post_checkpoint_load_operations"):
        operations = proposal.get(operations_name)
        if isinstance(operations, list):
            for index, operation in enumerate(operations):
                if isinstance(operation, dict):
                    check(operation.get("target"), f"{path}.{operations_name}[{index}].target")
                    for arguments_name in ("positional_arguments", "keyword_arguments"):
                        for ref_index, ref in enumerate(_iter_argument_object_refs(operation.get(arguments_name))):
                            check(ref, f"{path}.{operations_name}[{index}].{arguments_name}[object_ref:{ref_index}].value.object")
    checkpoint = proposal.get("checkpoint")
    if isinstance(checkpoint, dict):
        check(checkpoint.get("load_target"), f"{path}.checkpoint.load_target")


def _validate_input_contract(payload: Any) -> dict[str, Any]:
    findings: list[Finding] = []
    root_fields = ("report_type", "schema_version", "task_binding", "asset_preflight_report", "project", "selected_proposal_id", "proposals")
    root = _check_keys(payload, "$", root_fields, root_fields, findings)
    if root is None:
        raise InputContractError(findings)
    if root.get("report_type") != PROPOSAL_TYPE:
        findings.append(_finding("CONST", "$.report_type", "failed", f"必须为 {PROPOSAL_TYPE}。"))
    if root.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        findings.append(_finding("CONST", "$.schema_version", "failed", f"必须为 {PROPOSAL_SCHEMA_VERSION}。"))

    task = _check_keys(root.get("task_binding"), "$.task_binding", ("task_id", "task_input_version", "asset_input_digest"), ("task_id", "task_input_version", "asset_input_digest"), findings)
    if task is not None:
        _check_string(task.get("task_id"), "$.task_binding.task_id", findings)
        version = task.get("task_input_version")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            findings.append(_finding("TYPE", "$.task_binding.task_input_version", "failed", "应为不小于 1 的整数。"))
        _check_string(task.get("asset_input_digest"), "$.task_binding.asset_input_digest", findings, pattern=SHA256_PATTERN)
    if not isinstance(root.get("asset_preflight_report"), dict):
        findings.append(_finding("TYPE", "$.asset_preflight_report", "failed", "应为 AssetPreflightReport 对象。"))
    project = _check_keys(root.get("project"), "$.project", ("source_project_path", "source_revision"), ("source_project_path", "source_revision"), findings)
    if project is not None:
        _check_string(project.get("source_project_path"), "$.project.source_project_path", findings)
        _check_string(project.get("source_revision"), "$.project.source_revision", findings, pattern=GIT_REVISION_PATTERN, nullable=True)

    selected = root.get("selected_proposal_id")
    if selected is not None:
        _check_string(selected, "$.selected_proposal_id", findings)
    proposals = root.get("proposals")
    ids: set[str] = set()
    if not isinstance(proposals, list) or not proposals:
        findings.append(_finding("TYPE", "$.proposals", "failed", "应为至少包含一个方案的数组。"))
    else:
        proposal_fields = ("proposal_id", "construction_steps", "result", "pre_checkpoint_load_operations", "post_checkpoint_load_operations", "checkpoint", "evidence_refs")
        step_fields = ("step_id", "callable", "positional_arguments", "keyword_arguments", "source_anchor", "evidence_refs")
        for index, proposal in enumerate(proposals):
            path = f"$.proposals[{index}]"
            obj = _check_keys(proposal, path, proposal_fields, proposal_fields, findings)
            if obj is None:
                continue
            proposal_id = obj.get("proposal_id")
            _check_string(proposal_id, f"{path}.proposal_id", findings)
            if isinstance(proposal_id, str):
                if proposal_id in ids:
                    findings.append(_finding("DUPLICATE", f"{path}.proposal_id", "failed", "proposal_id 重复。"))
                ids.add(proposal_id)
            steps = obj.get("construction_steps")
            if not isinstance(steps, list) or not steps:
                findings.append(_finding("TYPE", f"{path}.construction_steps", "failed", "应为至少包含一个步骤的数组。"))
            else:
                for step_index, step in enumerate(steps):
                    step_path = f"{path}.construction_steps[{step_index}]"
                    step_obj = _check_keys(step, step_path, step_fields, step_fields, findings)
                    if step_obj is None:
                        continue
                    _check_string(step_obj.get("step_id"), f"{step_path}.step_id", findings, pattern=SYMBOL_PATTERN)
                    _validate_callable_ref(step_obj.get("callable"), f"{step_path}.callable", findings)
                    _validate_arguments(step_obj.get("positional_arguments"), f"{step_path}.positional_arguments", findings, keyword=False)
                    _validate_arguments(step_obj.get("keyword_arguments"), f"{step_path}.keyword_arguments", findings, keyword=True)
                    _validate_anchor(step_obj.get("source_anchor"), f"{step_path}.source_anchor", findings)
                    _validate_evidence_refs(step_obj.get("evidence_refs"), f"{step_path}.evidence_refs", findings)
            result = _check_keys(obj.get("result"), f"{path}.result", ("runtime_object", "target_model"), ("runtime_object", "target_model"), findings)
            if result is not None:
                _validate_object_ref(result.get("runtime_object"), f"{path}.result.runtime_object", findings)
                _validate_object_ref(result.get("target_model"), f"{path}.result.target_model", findings)
            for operations_name in ("pre_checkpoint_load_operations", "post_checkpoint_load_operations"):
                operations = obj.get(operations_name)
                if not isinstance(operations, list):
                    findings.append(_finding("TYPE", f"{path}.{operations_name}", "failed", "应为数组。"))
                else:
                    for operation_index, operation in enumerate(operations):
                        _validate_operation(operation, f"{path}.{operations_name}[{operation_index}]", findings)
            _validate_checkpoint(obj.get("checkpoint"), f"{path}.checkpoint", findings)
            _validate_evidence_refs(obj.get("evidence_refs"), f"{path}.evidence_refs", findings)
            _validate_reference_semantics(obj, path, findings)
    if selected is not None and selected not in ids:
        findings.append(_finding("REFERENCE", "$.selected_proposal_id", "failed", "没有对应的 proposal_id。"))
    if any(item.status == "failed" for item in findings):
        raise InputContractError(findings)
    return root


def _resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _normalize_path_without_io(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _bind_preflight(payload: dict[str, Any], base_directory: Path) -> tuple[Path, Path | None, list[Finding]]:
    findings: list[Finding] = []
    task = payload["task_binding"]
    report = payload["asset_preflight_report"]
    if report.get("report_type") != "AssetPreflightReport" or report.get("overall_status") != "passed":
        findings.append(_finding("PREFLIGHT_NOT_PASSED", "$.asset_preflight_report", "failed", "必须提供通过的 AssetPreflightReport。"))
    for key in ("task_id", "task_input_version", "asset_input_digest"):
        if report.get(key) != task.get(key):
            findings.append(_finding("PREFLIGHT_BINDING_MISMATCH", f"$.asset_preflight_report.{key}", "failed", "与 task_binding 不一致。"))

    checks = report.get("checks")
    target_check = None
    checkpoint_check = None
    if isinstance(checks, list):
        for item in checks:
            if not isinstance(item, dict):
                continue
            if item.get("field") == "assets.target_project.location":
                target_check = item
            elif item.get("field") == "assets.base_model.checkpoint":
                checkpoint_check = item
    if not isinstance(target_check, dict) or target_check.get("status") != "passed":
        findings.append(_finding("PROJECT_NOT_AUTHORIZED", "$.asset_preflight_report.checks", "failed", "资产预检没有通过的目标项目检查。"))
        authorized_project = Path(".")
    else:
        try:
            authorized_project = Path(str(target_check["resolved_path"])).resolve(strict=True)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            findings.append(_finding("PROJECT_PATH_INVALID", "$.asset_preflight_report.checks", "failed", "目标项目规范路径无效。"))
            authorized_project = Path(".")

    try:
        proposed_project = _resolve_path(payload["project"]["source_project_path"], base_directory)
        if proposed_project != authorized_project:
            findings.append(_finding("PROJECT_PATH_MISMATCH", "$.project.source_project_path", "failed", "与资产预检授权的目标项目路径不一致。", {"authorized": str(authorized_project), "proposed": str(proposed_project)}))
        if not proposed_project.is_dir():
            findings.append(_finding("PROJECT_NOT_DIRECTORY", "$.project.source_project_path", "failed", "目标项目不是目录。"))
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        findings.append(_finding("PROJECT_PATH_ERROR", "$.project.source_project_path", "failed", f"无法解析目标项目：{exc}"))
        proposed_project = authorized_project

    authorized_checkpoint: Path | None = None
    if isinstance(checkpoint_check, dict) and checkpoint_check.get("status") == "passed":
        try:
            authorized_checkpoint = Path(str(checkpoint_check["resolved_path"])).resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            findings.append(_finding("CHECKPOINT_AUTHORIZATION_INVALID", "$.asset_preflight_report.checks", "failed", "checkpoint 授权路径无效。"))
    return proposed_project, authorized_checkpoint, findings


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and GIT_REVISION_PATTERN.fullmatch(value) else None


def _find_scope(tree: ast.Module, scope: str | None) -> ast.AST:
    if scope in (None, "", "<module>"):
        return tree
    parts = scope.split(".")
    current: ast.AST = tree
    for part in parts:
        body = getattr(current, "body", [])
        next_node = next(
            (
                node
                for node in body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == part
            ),
            None,
        )
        if next_node is None:
            raise LookupError(f"找不到源码作用域 {scope!r}。")
        current = next_node
    return current


def _imports(tree: ast.Module, current_module: str) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    modules: dict[str, str] = {}
    symbols: dict[str, tuple[str, str]] = {}
    current_package = current_module.rsplit(".", 1)[0] if "." in current_module else ""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    modules[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    modules[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            module = node.module
            if node.level and current_package:
                prefix_parts = current_package.split(".")
                keep = max(0, len(prefix_parts) - node.level + 1)
                module = ".".join([*prefix_parts[:keep], node.module]).strip(".")
            for alias in node.names:
                if alias.name == "*":
                    continue
                symbols[alias.asname or alias.name] = (module, alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.setdefault(node.name, (current_module, node.name))
    return modules, symbols


def _attribute_parts(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return list(reversed(parts))


def _call_match_mode(
    call: ast.Call,
    callable_module: str,
    callable_symbol: str,
    module_imports: dict[str, str],
    symbol_imports: dict[str, tuple[str, str]],
    anchor_module: str,
) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        origin = symbol_imports.get(func.id)
        if origin == (callable_module, callable_symbol):
            return "direct_imported_symbol"
        if func.id == callable_symbol and anchor_module == callable_module:
            return "direct_local_symbol"

    parts = _attribute_parts(func)
    if parts and len(parts) >= 2:
        imported_root = module_imports.get(parts[0]) or parts[0]
        resolved_module = ".".join([imported_root, *parts[1:-1]])
        resolved_symbol = parts[-1]
        if resolved_module == callable_module and resolved_symbol == callable_symbol:
            return "direct_module_attribute"

    if isinstance(func, ast.Subscript) and isinstance(func.value, ast.Attribute) and func.value.attr == "__dict__":
        owner = _attribute_parts(func.value.value)
        if owner:
            resolved_module = module_imports.get(owner[0]) or ".".join(owner)
            if resolved_module == callable_module:
                return "dynamic_module_dict_candidate"

    if isinstance(func, ast.Call) and isinstance(func.func, ast.Name) and func.func.id == "getattr" and len(func.args) >= 2:
        owner = _attribute_parts(func.args[0])
        if owner:
            resolved_module = module_imports.get(owner[0]) or ".".join(owner)
            if resolved_module == callable_module:
                return "dynamic_getattr_candidate"
    return None


def _source_text(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    return segment.strip() if isinstance(segment, str) else ast.unparse(node)


def _expressions_equivalent(left: str, right: str) -> bool:
    try:
        left_tree = ast.parse(left.strip(), mode="eval")
        right_tree = ast.parse(right.strip(), mode="eval")
    except SyntaxError:
        return "".join(left.split()) == "".join(right.split())
    return ast.dump(left_tree.body, include_attributes=False) == ast.dump(
        right_tree.body,
        include_attributes=False,
    )


def _walk_lexical_scope(scope: ast.AST) -> Iterable[ast.AST]:
    """Walk one lexical scope without leaking into nested functions/classes."""
    stack = list(reversed(list(ast.iter_child_nodes(scope))))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(node))))


def _calls_at_anchor(parsed: ParsedModule, scope: ast.AST, anchor: dict[str, Any]) -> list[ast.Call]:
    calls = [node for node in _walk_lexical_scope(scope) if isinstance(node, ast.Call)]
    line = anchor.get("line")
    if isinstance(line, int):
        calls = [
            call
            for call in calls
            if getattr(call, "lineno", -1) <= line <= getattr(call, "end_lineno", getattr(call, "lineno", -1))
        ]
    return calls


def _find_assignments(scope: ast.AST, name: str, before_line: int) -> list[ast.AST]:
    candidates: list[ast.AST] = []
    for node in _walk_lexical_scope(scope):
        line = getattr(node, "lineno", before_line)
        if line >= before_line:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: Sequence[ast.AST]
            value: ast.AST | None
            if isinstance(node, ast.Assign):
                targets = node.targets
                value = node.value
            else:
                targets = [node.target]
                value = node.value
            if value is not None and any(isinstance(target, ast.Name) and target.id == name for target in targets):
                candidates.append(node)
    return sorted(candidates, key=lambda item: getattr(item, "lineno", 0), reverse=True)


def _resolve_expression(
    node: ast.AST,
    scope: ast.AST,
    before_line: int,
    module_imports: dict[str, str],
    symbol_imports: dict[str, tuple[str, str]],
    *,
    seen: set[str] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    provenance: list[dict[str, Any]] = []
    try:
        literal = ast.literal_eval(node)
        return {"kind": "literal", "value": literal}, provenance
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        pass

    if isinstance(node, ast.Name):
        if node.id in symbol_imports:
            module, symbol = symbol_imports[node.id]
            return {"kind": "symbol", "module": module, "symbol": symbol}, provenance
        active_seen = set() if seen is None else set(seen)
        if node.id in active_seen:
            return None, provenance
        active_seen.add(node.id)
        assignments = _find_assignments(scope, node.id, before_line)
        if assignments:
            assignment = assignments[0]
            value = assignment.value  # type: ignore[attr-defined]
            provenance.append({"line": getattr(assignment, "lineno", None), "expression": ast.unparse(value)})
            resolved, nested = _resolve_expression(
                value,
                scope,
                getattr(assignment, "lineno", before_line),
                module_imports,
                symbol_imports,
                seen=active_seen,
            )
            return resolved, [*provenance, *nested]

    parts = _attribute_parts(node)
    if parts:
        if len(parts) == 2 and parts[0] in module_imports:
            return {
                "kind": "symbol",
                "module": module_imports[parts[0]],
                "symbol": parts[1],
            }, provenance
    return None, provenance


def _argument_value_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right


def _direct_assignment_target(scope: ast.AST, call: ast.Call) -> str | None:
    """Return a name only when this exact call is the assignment value.

    This deliberately does not bind an inner call to the assignment that owns an
    outer wrapper call, e.g. ``model = DataParallel(resnet20())``.
    """
    for node in _walk_lexical_scope(scope):
        value: ast.AST | None = None
        targets: Sequence[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if value is not call:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        return names[0] if len(names) == 1 else None
    return None


def _direct_result_site(scope: ast.AST, call: ast.Call) -> dict[str, Any] | None:
    binding = _direct_assignment_target(scope, call)
    if binding is not None:
        return {"kind": "assignment", "binding": binding}
    for node in _walk_lexical_scope(scope):
        if isinstance(node, ast.Return) and node.value is call:
            return {"kind": "return", "binding": None}
    return None


def _source_object_ref(
    node: ast.AST,
    parsed: ParsedModule,
    scope: ast.AST,
    step_contexts: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    matches: list[dict[str, str]] = []
    for step_id, context in step_contexts.items():
        if context.get("parsed") is not parsed or context.get("scope") is not scope:
            continue
        if node is context.get("call"):
            matches.append({"step_id": step_id, "path": "$"})
            continue
        path = _object_path_from_expression(node, context.get("result_binding"))
        if path is not None:
            matches.append({"step_id": step_id, "path": path})
    unique = {(item["step_id"], item["path"]) for item in matches}
    if len(unique) != 1:
        return None
    step_id, path = next(iter(unique))
    return {"step_id": step_id, "path": path}


def _compare_argument(
    source_arg: ast.AST,
    proposed_arg: dict[str, Any],
    parsed: ParsedModule,
    scope: ast.AST,
    module_imports: dict[str, str],
    symbol_imports: dict[str, tuple[str, str]],
    step_contexts: dict[str, dict[str, Any]],
    before_line: int,
    path: str,
) -> list[Finding]:
    expression = _source_text(parsed.source, source_arg)
    declared_expression = proposed_arg.get("source_expression")
    if declared_expression is not None and not _expressions_equivalent(declared_expression, expression):
        return [_finding("SOURCE_EXPRESSION_MISMATCH", f"{path}.source_expression", "failed", "方案记录的表达式与源码调用不一致。", {"source": expression, "proposal": declared_expression})]

    proposed_value = proposed_arg["value"]
    if proposed_value.get("kind") == "object_ref":
        resolved_ref = _source_object_ref(source_arg, parsed, scope, step_contexts)
        if resolved_ref is None:
            return [_finding("OBJECT_REFERENCE_UNRESOLVED", f"{path}.value.object", "unresolved", "无法把源码实参唯一绑定到某个前序创建步骤。", {"source_expression": expression})]
        if resolved_ref != proposed_value["object"]:
            return [_finding("OBJECT_REFERENCE_MISMATCH", f"{path}.value.object", "failed", "源码实参引用的对象与方案声明不一致。", {"source": resolved_ref, "proposal": proposed_value["object"]})]
        return [_finding("OBJECT_REFERENCE_MATCH", f"{path}.value.object", "passed", "源码实参与前序创建步骤的对象引用一致。", {"source_expression": expression})]

    resolved, provenance = _resolve_expression(
        source_arg,
        scope,
        before_line,
        module_imports,
        symbol_imports,
    )
    if resolved is None:
        return [_finding("ARGUMENT_VALUE_UNRESOLVED", f"{path}.value", "unresolved", "无法通过受支持的静态追溯确定源码实参。", {"source_expression": expression, "provenance": provenance})]
    if not _argument_value_equal(resolved, proposed_value):
        return [_finding("ARGUMENT_VALUE_MISMATCH", f"{path}.value", "failed", "方案参数值与静态追溯结果不一致。", {"source": resolved, "proposal": proposed_value, "provenance": provenance})]
    return [_finding("ARGUMENT_VALUE_MATCH", f"{path}.value", "passed", "参数值与静态追溯结果一致。", {"source_expression": expression, "provenance": provenance})]


def _compare_call_arguments(
    call: ast.Call,
    specification: dict[str, Any],
    parsed: ParsedModule,
    scope: ast.AST,
    module_imports: dict[str, str],
    symbol_imports: dict[str, tuple[str, str]],
    step_contexts: dict[str, dict[str, Any]],
    field_prefix: str,
) -> list[Finding]:
    findings: list[Finding] = []
    positional = specification["positional_arguments"]
    keywords = specification["keyword_arguments"]
    if len(call.args) != len(positional):
        findings.append(_finding("CALL_POSITIONAL_COUNT_MISMATCH", f"{field_prefix}.positional_arguments", "failed", "方案的位置参数数量与源码调用不一致。", {"source_count": len(call.args), "proposal_count": len(positional)}))
    else:
        for index, (source_arg, proposed_arg) in enumerate(zip(call.args, positional)):
            findings.extend(_compare_argument(source_arg, proposed_arg, parsed, scope, module_imports, symbol_imports, step_contexts, getattr(call, "lineno", 0), f"{field_prefix}.positional_arguments[{index}]"))

    unpacked_keywords = [item for item in call.keywords if item.arg is None]
    if unpacked_keywords:
        findings.append(_finding("UNPACKED_KEYWORDS_UNSUPPORTED", f"{field_prefix}.keyword_arguments", "unresolved", "源码调用包含 **kwargs，静态工具不能证明完整关键字参数集合。"))
        return findings
    source_keywords = {item.arg: item for item in call.keywords if item.arg is not None}
    proposal_keywords = {item["parameter"]: item for item in keywords}
    if set(source_keywords) != set(proposal_keywords):
        findings.append(_finding("CALL_KEYWORD_SET_MISMATCH", f"{field_prefix}.keyword_arguments", "failed", "方案的关键字参数集合与源码调用不一致。", {"source": sorted(source_keywords), "proposal": sorted(proposal_keywords)}))
    else:
        for parameter, source_keyword in source_keywords.items():
            findings.extend(_compare_argument(source_keyword.value, proposal_keywords[parameter], parsed, scope, module_imports, symbol_imports, step_contexts, getattr(call, "lineno", 0), f"{field_prefix}.keyword_arguments[{parameter}]"))
    return findings


def _signature_parameters(definition: CallableDefinition) -> dict[str, Any]:
    args = definition.signature_node.args
    positional_nodes = [*args.posonlyargs, *args.args]
    if definition.kind == "class" and positional_nodes and positional_nodes[0].arg in {"self", "cls"}:
        positional_nodes = positional_nodes[1:]
    defaults: list[ast.AST | None] = [None] * (len(positional_nodes) - len(args.defaults)) + list(args.defaults)
    positional = [
        {
            "name": parameter.arg,
            "positional_only": index < len(args.posonlyargs),
            "default": ast.unparse(default) if default is not None else None,
            "required": default is None,
        }
        for index, (parameter, default) in enumerate(zip(positional_nodes, defaults))
    ]
    keyword_only = [
        {
            "name": parameter.arg,
            "default": ast.unparse(default) if default is not None else None,
            "required": default is None,
        }
        for parameter, default in zip(args.kwonlyargs, args.kw_defaults)
    ]
    return {
        "positional": positional,
        "keyword_only": keyword_only,
        "vararg": args.vararg.arg if args.vararg else None,
        "kwarg": args.kwarg.arg if args.kwarg else None,
    }


def _bind_signature(definition: CallableDefinition, specification: dict[str, Any], field_prefix: str) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    signature = _signature_parameters(definition)
    positional_values = specification["positional_arguments"]
    keyword_values = {item["parameter"]: item for item in specification["keyword_arguments"]}
    positional_parameters = signature["positional"]
    if signature["vararg"] is None and len(positional_values) > len(positional_parameters):
        findings.append(_finding("TOO_MANY_POSITIONAL_ARGUMENTS", f"{field_prefix}.positional_arguments", "failed", "位置参数数量超过 callable 签名。"))

    bound_names = {item["name"] for item in positional_parameters[: len(positional_values)]}
    known_keyword_names = {item["name"] for item in positional_parameters if not item["positional_only"]} | {item["name"] for item in signature["keyword_only"]}
    for name in keyword_values:
        if name in bound_names:
            findings.append(_finding("DUPLICATE_ARGUMENT_BINDING", f"{field_prefix}.keyword_arguments", "failed", f"参数 {name} 同时由位置和关键字提供。"))
        elif name not in known_keyword_names and signature["kwarg"] is None:
            findings.append(_finding("UNKNOWN_KEYWORD_ARGUMENT", f"{field_prefix}.keyword_arguments", "failed", f"callable 签名不接受关键字参数 {name}。"))

    supplied = bound_names | set(keyword_values)
    required = {item["name"] for item in [*positional_parameters, *signature["keyword_only"]] if item["required"]}
    missing = sorted(required - supplied)
    if missing:
        findings.append(_finding("MISSING_REQUIRED_ARGUMENTS", field_prefix, "failed", "缺少 callable 的必填参数。", {"missing": missing}))
    if not any(item.status == "failed" for item in findings):
        findings.append(_finding("CALLABLE_SIGNATURE_BOUND", field_prefix, "passed", "方案参数可以绑定到 callable 签名。", signature))
    return findings, signature


def _validate_symbol_arguments(repo: SourceRepository, specification: dict[str, Any], field_prefix: str) -> list[Finding]:
    findings: list[Finding] = []
    arguments = [
        *specification["positional_arguments"],
        *specification["keyword_arguments"],
    ]
    for index, argument in enumerate(arguments):
        value = argument["value"]
        if value.get("kind") != "symbol":
            continue
        try:
            repo.find_callable(value["module"], value["symbol"])
        except (FileNotFoundError, LookupError, OSError, UnicodeError, SyntaxError) as exc:
            findings.append(_finding("ARGUMENT_SYMBOL_NOT_FOUND", f"{field_prefix}.arguments[{index}]", "failed", f"符号参数无法定位：{exc}"))
        else:
            findings.append(_finding("ARGUMENT_SYMBOL_FOUND", f"{field_prefix}.arguments[{index}]", "passed", "符号参数可以在项目源码中定位。", {"module": value["module"], "symbol": value["symbol"]}))
    return findings


def _validate_construction_step(
    repo: SourceRepository,
    step: dict[str, Any],
    step_contexts: dict[str, dict[str, Any]],
    field_prefix: str,
) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    callable_ref = step["callable"]
    anchor = step["source_anchor"]
    definition: CallableDefinition | None = None
    if callable_ref["origin"] == "project":
        try:
            definition = repo.find_callable(callable_ref["module"], callable_ref["symbol"])
        except (FileNotFoundError, LookupError, OSError, UnicodeError, SyntaxError) as exc:
            findings.append(_finding("CALLABLE_NOT_FOUND", f"{field_prefix}.callable", "failed", f"无法定位项目 callable：{exc}"))
        else:
            if isinstance(definition.node, ast.AsyncFunctionDef):
                findings.append(_finding("ASYNC_CALLABLE_UNSUPPORTED", f"{field_prefix}.callable", "failed", "不支持异步模型工厂。"))
            else:
                findings.append(_finding("CALLABLE_FOUND", f"{field_prefix}.callable", "passed", "项目 callable 可以在源码中定位。", {"kind": definition.kind, "path": str(definition.module.path), "line": getattr(definition.node, "lineno", None), "sha256": definition.module.sha256}))
                signature_findings, _ = _bind_signature(definition, step, field_prefix)
                findings.extend(signature_findings)
                findings.extend(_validate_symbol_arguments(repo, step, field_prefix))
    elif (callable_ref["module"], callable_ref["symbol"]) not in APPROVED_PYTORCH_CALLABLES:
        findings.append(_finding("PYTORCH_CALLABLE_NOT_APPROVED", f"{field_prefix}.callable", "failed", "该 PyTorch callable 尚未列入安全实例化允许列表。"))
    else:
        findings.append(_finding("PYTORCH_CALLABLE_APPROVED", f"{field_prefix}.callable", "passed", "PyTorch 包装 callable 位于明确允许列表中。"))
        wrapper_key = (callable_ref["module"], callable_ref["symbol"])
        module_parameter = PYTORCH_WRAPPER_MODULE_PARAMETER[wrapper_key]
        module_argument = None
        if step["positional_arguments"]:
            module_argument = step["positional_arguments"][0]
        else:
            module_argument = next(
                (item for item in step["keyword_arguments"] if item["parameter"] == module_parameter),
                None,
            )
        if module_argument is None:
            findings.append(_finding("WRAPPER_MODULE_ARGUMENT_MISSING", field_prefix, "failed", "PyTorch 包装步骤没有提供被包装的 module 对象。"))
        elif module_argument["value"].get("kind") != "object_ref":
            findings.append(_finding("WRAPPER_MODULE_ARGUMENT_NOT_OBJECT_REFERENCE", field_prefix, "failed", "PyTorch 包装步骤的 module 参数必须引用前序创建步骤。"))
        else:
            findings.append(_finding("WRAPPER_MODULE_ARGUMENT_BOUND", field_prefix, "passed", "PyTorch 包装步骤显式引用前序创建对象。"))

    try:
        parsed = repo.parse_relative_path(anchor["path"])
        scope = _find_scope(parsed.tree, anchor.get("scope"))
    except (FileNotFoundError, OSError, UnicodeError, SyntaxError, ValueError, LookupError) as exc:
        findings.append(_finding("CONSTRUCTION_ANCHOR_INVALID", f"{field_prefix}.source_anchor", "failed", f"无法读取创建调用锚点：{exc}"))
        return findings, {}

    calls = _calls_at_anchor(parsed, scope, anchor)
    module_imports, symbol_imports = _imports(parsed.tree, parsed.module)
    matches: list[tuple[ast.Call, str]] = []
    for call in calls:
        mode = _call_match_mode(
            call,
            callable_ref["module"],
            callable_ref["symbol"],
            module_imports,
            symbol_imports,
            parsed.module,
        )
        if mode is not None:
            matches.append((call, mode))
    declared_expression = anchor.get("call_expression")
    if declared_expression is not None:
        matches = [item for item in matches if _expressions_equivalent(declared_expression, _source_text(parsed.source, item[0]))]
    if not matches:
        findings.append(_finding("CALLABLE_NOT_BOUND_AT_ANCHOR", f"{field_prefix}.source_anchor", "failed", "创建调用锚点不能唯一绑定到方案声明的 callable 和调用表达式。", {"candidate_calls": [_source_text(parsed.source, call) for call in calls]}))
        return findings, {"path": str(parsed.path), "sha256": parsed.sha256}
    if len(matches) > 1:
        findings.append(_finding("MULTIPLE_CALLABLE_MATCHES_AT_ANCHOR", f"{field_prefix}.source_anchor", "unresolved", "创建锚点存在多个与 callable 相符的调用，无法唯一绑定。", {"matches": [{"line": call.lineno, "mode": mode, "expression": _source_text(parsed.source, call)} for call, mode in matches]}))
        return findings, {"path": str(parsed.path), "sha256": parsed.sha256}

    call, mode = matches[0]
    findings.append(_finding("CALLABLE_BOUND_AT_ANCHOR", f"{field_prefix}.source_anchor", "passed", "创建锚点与方案 callable 绑定成功。", {"line": call.lineno, "mode": mode, "expression": _source_text(parsed.source, call)}))
    findings.extend(_compare_call_arguments(call, step, parsed, scope, module_imports, symbol_imports, step_contexts, field_prefix))
    result_binding = _direct_assignment_target(scope, call)
    result_site = _direct_result_site(scope, call)
    context = {
        "step": step,
        "parsed": parsed,
        "scope": scope,
        "call": call,
        "result_binding": result_binding,
        "result_site": result_site,
        "definition": definition,
        "anchor_details": {
        "path": str(parsed.path),
        "relative_path": anchor["path"],
        "sha256": parsed.sha256,
        "scope": anchor.get("scope"),
        "line": call.lineno,
        "match_mode": mode,
        "call_expression": _source_text(parsed.source, call),
        "result_binding": result_binding,
        "result_site": result_site,
        },
    }
    return findings, context


def _object_path_from_expression(node: ast.AST, root_name: str | None) -> str | None:
    parts = _attribute_parts(node)
    if not parts or root_name is None or parts[0] != root_name:
        return None
    return "$" + "".join(f".{part}" for part in parts[1:])


def _validate_operation_anchor(
    repo: SourceRepository,
    operation: dict[str, Any],
    step_contexts: dict[str, dict[str, Any]],
    field: str,
) -> list[Finding]:
    findings: list[Finding] = []
    anchor = operation["source_anchor"]
    try:
        parsed = repo.parse_relative_path(anchor["path"])
        scope = _find_scope(parsed.tree, anchor.get("scope"))
    except (FileNotFoundError, OSError, UnicodeError, SyntaxError, ValueError, LookupError) as exc:
        return [_finding("OPERATION_ANCHOR_INVALID", field, "failed", f"无法读取生命周期操作锚点：{exc}")]
    calls = _calls_at_anchor(parsed, scope, anchor)
    matches = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == operation["method"]
    ]
    declared_expression = anchor.get("call_expression")
    if declared_expression is not None:
        matches = [call for call in matches if _expressions_equivalent(declared_expression, _source_text(parsed.source, call))]
    if len(matches) != 1:
        status: CheckStatus = "failed" if not matches else "unresolved"
        return [_finding("OPERATION_CALL_NOT_UNIQUE", field, status, "生命周期操作锚点没有唯一匹配的方法调用。", {"match_count": len(matches)})]
    call = matches[0]
    actual_target = _source_object_ref(call.func.value, parsed, scope, step_contexts)
    if actual_target is not None and actual_target != operation["target"]:
        findings.append(_finding("OPERATION_TARGET_MISMATCH", f"{field}.target", "failed", "操作接收对象与方案声明不一致。", {"source": actual_target, "proposal": operation["target"]}))
    elif actual_target is None:
        findings.append(_finding("OPERATION_TARGET_UNRESOLVED", f"{field}.target", "unresolved", "无法把操作接收对象唯一绑定到创建步骤。"))
    else:
        findings.append(_finding("OPERATION_TARGET_MATCH", f"{field}.target", "passed", "操作接收对象与创建步骤引用一致。"))
    module_imports, symbol_imports = _imports(parsed.tree, parsed.module)
    findings.extend(_compare_call_arguments(call, operation, parsed, scope, module_imports, symbol_imports, step_contexts, field))
    return findings


def _selector_path_from_expression(node: ast.AST, root_name: str | None = None) -> str | None:
    parts: list[str] = []
    current = node
    while True:
        if isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
            continue
        if isinstance(current, ast.Subscript):
            slice_node = current.slice
            try:
                key = ast.literal_eval(slice_node)
            except (ValueError, TypeError, SyntaxError):
                return None
            if not isinstance(key, str) or not key.isidentifier():
                return None
            parts.append(key)
            current = current.value
            continue
        break
    if not isinstance(current, ast.Name):
        return None
    if root_name is not None and current.id != root_name:
        return None
    return "$" + "".join(f".{part}" for part in reversed(parts))


def _validate_checkpoint_source_anchor(
    repo: SourceRepository,
    checkpoint: dict[str, Any],
    step_contexts: dict[str, dict[str, Any]],
    field: str,
) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    anchor = checkpoint["source_anchor"]
    try:
        parsed = repo.parse_relative_path(anchor["path"])
        scope = _find_scope(parsed.tree, anchor.get("scope"))
    except (FileNotFoundError, OSError, UnicodeError, SyntaxError, ValueError, LookupError) as exc:
        return [_finding("CHECKPOINT_ANCHOR_INVALID", field, "failed", f"无法读取 checkpoint 加载锚点：{exc}")], {}
    calls = _calls_at_anchor(parsed, scope, anchor)
    matches = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "load_state_dict"
    ]
    declared_expression = anchor.get("call_expression")
    if declared_expression is not None:
        matches = [call for call in matches if _expressions_equivalent(declared_expression, _source_text(parsed.source, call))]
    if len(matches) != 1:
        status: CheckStatus = "failed" if not matches else "unresolved"
        return [_finding("LOAD_STATE_DICT_NOT_UNIQUE", field, status, "checkpoint 锚点没有唯一匹配的 load_state_dict 调用。", {"match_count": len(matches)})], {"path": str(parsed.path), "sha256": parsed.sha256}
    call = matches[0]
    target_ref = _source_object_ref(call.func.value, parsed, scope, step_contexts)
    if target_ref is None:
        findings.append(_finding("CHECKPOINT_LOAD_TARGET_UNRESOLVED", f"{field}.load_target", "unresolved", "无法把 load_state_dict 接收对象唯一绑定到创建步骤。"))
    elif target_ref != checkpoint["load_target"]:
        findings.append(_finding("CHECKPOINT_LOAD_TARGET_MISMATCH", f"{field}.load_target", "failed", "load_state_dict 接收对象与方案不一致。", {"source": target_ref, "proposal": checkpoint["load_target"]}))
    else:
        findings.append(_finding("CHECKPOINT_LOAD_TARGET_MATCH", f"{field}.load_target", "passed", "load_state_dict 接收对象与方案一致。"))

    source_selector = None
    if call.args:
        source_selector = _selector_path_from_expression(call.args[0])
    proposed_selector = checkpoint.get("selector_path")
    if source_selector is not None and source_selector != proposed_selector:
        findings.append(_finding("CHECKPOINT_SELECTOR_MISMATCH", f"{field}.selector_path", "failed", "checkpoint 选择路径与源码调用不一致。", {"source": source_selector, "proposal": proposed_selector}))
    elif source_selector is not None:
        findings.append(_finding("CHECKPOINT_SELECTOR_SOURCE_FOUND", f"{field}.selector_path", "passed", "proposal 的 checkpoint 选择路径与源码加载调用一致。", {"source_selector_path": source_selector}))
    else:
        findings.append(_finding("CHECKPOINT_SELECTOR_NOT_REDERIVED", f"{field}.selector_path", "warning", "无法从源码调用机械提取 selector；静态验证器不重新分析 checkpoint 或替 Agent 推导该值。"))
    return findings, {
        "path": str(parsed.path),
        "sha256": parsed.sha256,
        "line": call.lineno,
        "source_selector_path": source_selector,
    }


def _path_is_authorized_checkpoint(path: Path, authorized: Path | None, project_root: Path) -> bool:
    try:
        path.relative_to(authorized or project_root)
        return True
    except ValueError:
        return False


def _bind_checkpoint_recipe(
    checkpoint: dict[str, Any],
    project_root: Path,
    authorized_checkpoint: Path | None,
    base_directory: Path,
    field: str,
) -> tuple[list[Finding], dict[str, Any] | None]:
    try:
        path = _normalize_path_without_io(checkpoint["path"], base_directory)
    except (OSError, RuntimeError) as exc:
        return [_finding("CHECKPOINT_PATH_ERROR", f"{field}.path", "failed", f"无法规范化 checkpoint 路径：{exc}")], None
    if not _path_is_authorized_checkpoint(path, authorized_checkpoint, project_root):
        return [_finding("CHECKPOINT_NOT_AUTHORIZED", f"{field}.path", "failed", "checkpoint 加载指令不在资产预检授权范围内。")], None
    recipe = {
        "path": str(path),
        "selector_path": checkpoint.get("selector_path"),
        "key_transforms": checkpoint["key_transforms"],
        "load_target": checkpoint["load_target"],
        "strict": checkpoint["strict"],
    }
    return [
        _finding(
            "CHECKPOINT_RECIPE_BOUND",
            field,
            "passed",
            "checkpoint 加载指令已绑定到资产授权范围；静态验证器未打开或解析 checkpoint。",
            {"normalized_path": str(path)},
        )
    ], recipe


def _status_from_findings(findings: Sequence[Finding]) -> str:
    if any(item.status == "failed" for item in findings):
        return "rejected"
    if any(item.status == "unresolved" for item in findings):
        return "unresolved"
    return "validated"


def _validate_result_reference(
    reference: dict[str, str],
    step_contexts: dict[str, dict[str, Any]],
    field: str,
) -> list[Finding]:
    context = step_contexts.get(reference["step_id"])
    if context is None:
        return [_finding("RESULT_STEP_NOT_VALIDATED", f"{field}.step_id", "failed", "结果引用的创建步骤未通过锚点绑定。")]
    path = reference["path"]
    if path == "$":
        return [_finding("RESULT_OBJECT_ROOT", field, "passed", "结果引用创建步骤的直接返回对象。")]
    callable_ref = context["step"]["callable"]
    known_paths = KNOWN_RESULT_PATHS.get((callable_ref["module"], callable_ref["symbol"]), set())
    if callable_ref["origin"] == "pytorch" and path in known_paths:
        return [_finding("RESULT_OBJECT_PATH_KNOWN", field, "passed", "结果路径是该受支持 PyTorch 包装器的已知公开对象路径。", {"callable": callable_ref, "path": path})]
    return [_finding("RESULT_OBJECT_PATH_UNRESOLVED", f"{field}.path", "unresolved", "仅凭静态 proposal 不能证明该创建步骤的非根结果路径存在。", {"callable": callable_ref, "path": path})]


def _validate_runtime_object_exposure(
    reference: dict[str, str],
    step_contexts: dict[str, dict[str, Any]],
    field: str,
) -> list[Finding]:
    context = step_contexts.get(reference["step_id"])
    if context is None:
        return []
    result_site = context.get("result_site")
    if result_site is None:
        return [_finding("RUNTIME_OBJECT_NOT_DIRECTLY_EXPOSED", field, "unresolved", "声明的运行时对象只出现在另一个未纳入方案的外层表达式中；创建链可能不完整。")]
    return [_finding("RUNTIME_OBJECT_DIRECTLY_EXPOSED", field, "passed", "运行时对象所属步骤的根结果由源码直接赋值或返回。", result_site)]


def _validate_construction_graph_completeness(
    proposal: dict[str, Any],
    field: str,
) -> list[Finding]:
    dependencies: dict[str, set[str]] = {}
    for step in proposal["construction_steps"]:
        refs: set[str] = set()
        for arguments_name in ("positional_arguments", "keyword_arguments"):
            refs.update(ref["step_id"] for ref in _iter_argument_object_refs(step[arguments_name]))
        dependencies[step["step_id"]] = refs
    runtime_step = proposal["result"]["runtime_object"]["step_id"]
    reachable: set[str] = set()
    pending = [runtime_step]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(dependencies.get(current, set()))
    disconnected = sorted(set(dependencies) - reachable)
    if disconnected:
        return [_finding("CONSTRUCTION_STEPS_DISCONNECTED_FROM_RUNTIME", field, "failed", "存在未参与最终运行时对象创建的步骤；方案不是一条完整且单一的对象创建链。", {"disconnected_step_ids": disconnected, "runtime_step_id": runtime_step})]
    return [_finding("CONSTRUCTION_GRAPH_CONNECTED", field, "passed", "所有创建步骤都参与最终运行时对象的依赖链。", {"runtime_step_id": runtime_step})]


def _validate_proposal(
    repo: SourceRepository,
    proposal: dict[str, Any],
    index: int,
    authorized_checkpoint: Path | None,
    base_directory: Path,
) -> dict[str, Any]:
    prefix = f"$.proposals[{index}]"
    construction_findings: list[Finding] = []
    checkpoint_findings: list[Finding] = []
    step_contexts: dict[str, dict[str, Any]] = {}
    for step_index, step in enumerate(proposal["construction_steps"]):
        step_prefix = f"{prefix}.construction_steps[{step_index}]"
        step_findings, context = _validate_construction_step(repo, step, step_contexts, step_prefix)
        construction_findings.extend(step_findings)
        if context.get("step") is not None:
            step_contexts[step["step_id"]] = context

    construction_findings.extend(_validate_result_reference(proposal["result"]["runtime_object"], step_contexts, f"{prefix}.result.runtime_object"))
    construction_findings.extend(_validate_runtime_object_exposure(proposal["result"]["runtime_object"], step_contexts, f"{prefix}.result.runtime_object"))
    construction_findings.extend(_validate_result_reference(proposal["result"]["target_model"], step_contexts, f"{prefix}.result.target_model"))
    construction_findings.extend(_validate_construction_graph_completeness(proposal, f"{prefix}.construction_steps"))

    for name in ("pre_checkpoint_load_operations", "post_checkpoint_load_operations"):
        for operation_index, operation in enumerate(proposal[name]):
            construction_findings.extend(
                _validate_operation_anchor(
                    repo,
                    operation,
                    step_contexts,
                    f"{prefix}.{name}[{operation_index}]",
                )
            )

    construction_status = _status_from_findings(construction_findings)
    checkpoint_recipe_status = "not_provided"
    resolved_checkpoint_spec = None
    if proposal["checkpoint"] is not None:
        source_findings, source_details = _validate_checkpoint_source_anchor(
            repo,
            proposal["checkpoint"],
            step_contexts,
            f"{prefix}.checkpoint",
        )
        checkpoint_findings.extend(source_findings)
        binding_findings, resolved_checkpoint_spec = _bind_checkpoint_recipe(
            proposal["checkpoint"],
            repo.root,
            authorized_checkpoint,
            base_directory,
            f"{prefix}.checkpoint",
        )
        checkpoint_findings.extend(binding_findings)
        checkpoint_recipe_status = _status_from_findings(checkpoint_findings)
        if checkpoint_recipe_status != "validated":
            resolved_checkpoint_spec = None
        elif resolved_checkpoint_spec is not None:
            resolved_checkpoint_spec["source_anchor_snapshot"] = source_details

    validated_spec = None
    if construction_status == "validated":
        step_sources = []
        for step in proposal["construction_steps"]:
            context = step_contexts[step["step_id"]]
            definition = context.get("definition")
            callable_source = None
            if definition is not None:
                callable_source = {
                    "path": str(definition.module.path),
                    "sha256": definition.module.sha256,
                    "line": getattr(definition.node, "lineno", None),
                }
            step_sources.append({
                "step_id": step["step_id"],
                "callable_source": callable_source,
                "construction_anchor_source": context["anchor_details"],
            })
        validated_spec = {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "proposal_id": proposal["proposal_id"],
            "project": {
                "source_project_path": str(repo.root),
                "source_revision": None,
                "step_sources": step_sources,
            },
            "construction_steps": proposal["construction_steps"],
            "result": proposal["result"],
            "pre_checkpoint_load_operations": proposal["pre_checkpoint_load_operations"],
            "post_checkpoint_load_operations": proposal["post_checkpoint_load_operations"],
            "checkpoint_load_attempt": resolved_checkpoint_spec,
        }
    return {
        "proposal_id": proposal["proposal_id"],
        "construction_status": construction_status,
        "checkpoint_recipe_status": checkpoint_recipe_status,
        "findings": [asdict(item) for item in [*construction_findings, *checkpoint_findings]],
        "validated_creation_spec": validated_spec,
    }


def _selection(payload: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [item for item in results if item["construction_status"] == "validated"]
    requested = payload.get("selected_proposal_id")
    if requested is not None:
        selected = next((item for item in ready if item["proposal_id"] == requested), None)
        if selected is not None:
            return {"status": "selected", "selected_proposal_id": requested, "viable_proposal_ids": [item["proposal_id"] for item in ready]}
        return {"status": "selected_proposal_not_validated", "selected_proposal_id": None, "requested_proposal_id": requested, "viable_proposal_ids": [item["proposal_id"] for item in ready]}
    if len(ready) == 1:
        return {"status": "unique", "selected_proposal_id": ready[0]["proposal_id"], "viable_proposal_ids": [ready[0]["proposal_id"]]}
    if len(ready) > 1:
        return {"status": "ambiguous", "selected_proposal_id": None, "viable_proposal_ids": [item["proposal_id"] for item in ready]}
    return {"status": "none", "selected_proposal_id": None, "viable_proposal_ids": []}


def validate_model_creation_proposals(
    payload: Any,
    *,
    base_directory: Path | None = None,
) -> dict[str, Any]:
    base = (base_directory or Path.cwd()).resolve(strict=True)
    try:
        contract = _validate_input_contract(payload)
    except InputContractError as exc:
        return {
            "report_type": REPORT_TYPE,
            "schema_version": REPORT_SCHEMA_VERSION,
            "task_binding": None,
            "project_snapshot": None,
            "input_findings": [asdict(item) for item in exc.findings],
            "proposal_results": [],
            "selection": {"status": "none", "selected_proposal_id": None, "viable_proposal_ids": []},
            "validated_instantiation_plan": None,
            "readiness": {
                "ready_for_model_construction": False,
                "ready_for_checkpoint_load_attempt": False,
                "blockers": ["input_contract_invalid"],
            },
        }

    project_root, authorized_checkpoint, binding_findings = _bind_preflight(contract, base)
    expected_revision = contract["project"].get("source_revision")
    actual_revision = _git_revision(project_root) if project_root.is_dir() else None
    if expected_revision is not None and actual_revision != expected_revision:
        binding_findings.append(_finding("SOURCE_REVISION_MISMATCH", "$.project.source_revision", "failed", "当前项目 revision 与方案声明不一致。", {"expected": expected_revision, "actual": actual_revision}))

    if any(item.status == "failed" for item in binding_findings):
        results: list[dict[str, Any]] = []
    else:
        repo = SourceRepository(project_root)
        results = [
            _validate_proposal(repo, proposal, index, authorized_checkpoint, base)
            for index, proposal in enumerate(contract["proposals"])
        ]
        for result in results:
            spec = result.get("validated_creation_spec")
            if isinstance(spec, dict):
                spec["project"]["source_revision"] = actual_revision

    selection = _selection(contract, results)
    selected_result = next(
        (item for item in results if item["proposal_id"] == selection.get("selected_proposal_id")),
        None,
    )
    ready_for_construction = selected_result is not None and selected_result["construction_status"] == "validated"
    ready_for_checkpoint = ready_for_construction and selected_result["checkpoint_recipe_status"] in {"validated", "not_provided"}
    if selected_result is not None and selected_result["checkpoint_recipe_status"] == "not_provided":
        ready_for_checkpoint = False

    blockers: list[str] = []
    if any(item.status == "failed" for item in binding_findings):
        blockers.append("upstream_binding_failed")
    if selection["status"] == "ambiguous":
        blockers.append("multiple_validated_proposals_require_selection")
    elif selection["status"] in {"none", "selected_proposal_not_validated"}:
        blockers.append("no_selected_validated_proposal")
    if ready_for_construction and not ready_for_checkpoint and selected_result is not None and selected_result["checkpoint_recipe_status"] != "not_provided":
        blockers.append("checkpoint_recipe_not_validated")

    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_binding": contract["task_binding"],
        "project_snapshot": {
            "source_project_path": str(project_root),
            "declared_revision": expected_revision,
            "actual_revision": actual_revision,
            "binding_findings": [asdict(item) for item in binding_findings],
        },
        "input_findings": [],
        "proposal_results": results,
        "selection": selection,
        "validated_instantiation_plan": selected_result.get("validated_creation_spec") if ready_for_construction and selected_result is not None else None,
        "readiness": {
            "ready_for_model_construction": ready_for_construction,
            "ready_for_checkpoint_load_attempt": ready_for_checkpoint,
            "blockers": blockers,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="静态验证 Agent 提出的模型创建方案，不执行目标项目代码。")
    parser.add_argument("input", nargs="?", type=Path, help="ModelCreationProposalSet JSON 文件")
    parser.add_argument("--stdin", action="store_true", dest="read_stdin", help="从标准输入读取 JSON。")
    parser.add_argument("--base-directory", type=Path, default=None, help="解析相对路径的基准目录。")
    parser.add_argument("--output", choices=("text", "json"), default="text")
    args = parser.parse_args()
    if args.read_stdin == (args.input is not None):
        parser.error("必须且只能选择一个输入来源：文件路径或 --stdin。")
    return args


def _execution_error(code: str, message: str) -> dict[str, Any]:
    return {
        "report_type": REPORT_TYPE,
        "schema_version": REPORT_SCHEMA_VERSION,
        "task_binding": None,
        "project_snapshot": None,
        "input_findings": [],
        "proposal_results": [],
        "selection": {"status": "none", "selected_proposal_id": None, "viable_proposal_ids": []},
        "validated_instantiation_plan": None,
        "readiness": {
            "ready_for_model_construction": False,
            "ready_for_checkpoint_load_attempt": False,
            "blockers": ["execution_error"],
        },
        "execution_error": True,
        "error": {"code": code, "message": message},
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    args = parse_args()
    try:
        if args.read_stdin:
            payload = json.load(sys.stdin)
        else:
            with args.input.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        report = validate_model_creation_proposals(payload, base_directory=args.base_directory)
    except FileNotFoundError as exc:
        report = _execution_error("INPUT_NOT_FOUND", str(exc))
    except json.JSONDecodeError as exc:
        report = _execution_error("INVALID_JSON", str(exc))
    except OSError as exc:
        report = _execution_error("VALIDATION_IO_ERROR", str(exc))
    except Exception as exc:  # defensive Harness boundary
        report = _execution_error("UNEXPECTED_VALIDATION_ERROR", f"{type(exc).__name__}: {exc}")

    if args.output == "json":
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    else:
        readiness = report["readiness"]
        print(
            "模型创建方案静态验证："
            f"construction={readiness['ready_for_model_construction']}, "
            f"checkpoint={readiness['ready_for_checkpoint_load_attempt']}"
        )
        for blocker in readiness["blockers"]:
            print(f"- blocker: {blocker}")
    if report.get("execution_error"):
        return 2
    return 0 if report["readiness"]["ready_for_model_construction"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
