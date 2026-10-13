"""Bounded, project-agnostic incremental ridge-regression calculations.

The method accumulates unregularized sufficient statistics and solves the
fixed objective ``sum_squared_error + lambda * l2_penalty``.  It never reads a
project, discovers a model, mutates a checkpoint, or changes the objective.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "0.3"
METHOD_ID = "incremental_ridge_closed_form"
METHOD_VERSION = "0.3"
MAX_INPUT_FEATURES = 256
MAX_OUTPUT_FEATURES = 64
PARAMETER_RTOL = 1e-7
PARAMETER_ATOL = 1e-9


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _issue(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _exact_keys(value: Any, path: str, expected: set[str], issues: list[dict[str, str]]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        issues.append(_issue("TYPE", path, "必须是对象。"))
        return None
    for key in sorted(expected - value.keys()):
        issues.append(_issue("REQUIRED", f"{path}.{key}", "缺少必填字段。"))
    for key in sorted(value.keys() - expected):
        issues.append(_issue("UNKNOWN", f"{path}.{key}", "存在未定义字段。"))
    return value


def _positive_int(value: Any, path: str, maximum: int, issues: list[dict[str, str]]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        issues.append(_issue("RANGE", path, f"必须是 1 到 {maximum} 之间的整数。"))
        return None
    return value


def _sha256(value: Any, path: str, issues: list[dict[str, str]]) -> str | None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        issues.append(_issue("SHA256", path, "必须是 64 位小写十六进制 SHA-256。"))
        return None
    return value


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _matrix(value: Any, rows: int, columns: int, path: str, issues: list[dict[str, str]]) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != rows:
        issues.append(_issue("SHAPE", path, f"必须有 {rows} 行。"))
        return None
    normalized: list[list[float]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            issues.append(_issue("SHAPE", f"{path}[{row_index}]", f"必须有 {columns} 列。"))
            return None
        if not all(_finite_number(item) for item in row):
            issues.append(_issue("FINITE", f"{path}[{row_index}]", "只能包含有限数值。"))
            return None
        normalized.append([float(item) for item in row])
    return np.asarray(normalized, dtype=np.float64)


def _vector_or_null(value: Any, length: int, path: str, has_bias: bool, issues: list[dict[str, str]]) -> np.ndarray | None:
    if not has_bias:
        if value is not None:
            issues.append(_issue("BIAS", path, "has_bias=false 时 bias 必须为 null。"))
        return None
    if not isinstance(value, list) or len(value) != length or not all(_finite_number(item) for item in value):
        issues.append(_issue("SHAPE", path, f"必须是包含 {length} 个有限数值的数组。"))
        return None
    return np.asarray([float(item) for item in value], dtype=np.float64)


def _regularization_matrix(dimension: int, has_bias: bool, regularize_bias: bool) -> np.ndarray:
    diagonal = np.ones(dimension, dtype=np.float64)
    if has_bias and not regularize_bias:
        diagonal[-1] = 0.0
    return np.diag(diagonal)


def _solve(
    gram: np.ndarray,
    cross: np.ndarray,
    strength: float,
    has_bias: bool,
    regularize_bias: bool,
    statistics_path: str = "$.statistics",
) -> tuple[np.ndarray | None, float | None, dict[str, str] | None]:
    system = gram + strength * _regularization_matrix(gram.shape[0], has_bias, regularize_bias)
    try:
        condition_number = float(np.linalg.cond(system))
        coefficients = np.linalg.solve(system, cross)
    except np.linalg.LinAlgError as exc:
        return None, None, _issue("SINGULAR_SYSTEM", statistics_path, f"正则化法方程不可解：{exc}")
    if not np.isfinite(coefficients).all() or not math.isfinite(condition_number):
        return None, None, _issue("NUMERICAL_FAILURE", statistics_path, "求解结果或条件数不是有限数值。")
    return coefficients, condition_number, None


def _parameters(coefficients: np.ndarray, input_features: int, output_features: int, has_bias: bool) -> dict[str, Any]:
    if has_bias:
        weight = coefficients[:input_features, :].T
        bias: list[float] | None = coefficients[input_features, :].tolist()
    else:
        weight = coefficients.T
        bias = None
    return {"weight": weight.tolist(), "bias": bias}


def _validate_plan(value: Any, path: str, issues: list[dict[str, str]]) -> dict[str, Any] | None:
    plan = _exact_keys(value, path, {"task_id", "plan_version", "update_plan_sha256"}, issues)
    if plan is None:
        return None
    if not isinstance(plan.get("task_id"), str) or not plan.get("task_id", "").strip():
        issues.append(_issue("TYPE", f"{path}.task_id", "必须是非空字符串。"))
    if isinstance(plan.get("plan_version"), bool) or not isinstance(plan.get("plan_version"), int) or plan.get("plan_version", 0) < 1:
        issues.append(_issue("RANGE", f"{path}.plan_version", "必须是从 1 开始的整数。"))
    _sha256(plan.get("update_plan_sha256"), f"{path}.update_plan_sha256", issues)
    return plan


def _validate_problem(
    model_value: Any,
    objective_value: Any,
    statistics_value: Any,
    path: str,
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    model_path = f"{path}.model"
    objective_path = f"{path}.objective"
    statistics_path = f"{path}.statistics"
    model = _exact_keys(model_value, model_path, {"input_features", "output_features", "has_bias", "dtype"}, issues)
    objective = _exact_keys(objective_value, objective_path, {"loss", "regularization", "regularization_strength", "regularize_bias"}, issues)
    statistics = _exact_keys(statistics_value, statistics_path, {"sample_count", "base_feature_dim", "design_feature_dim", "bias_feature_position", "G", "C"}, issues)
    if any(item is None for item in (model, objective, statistics)):
        return None, None, None, None, None

    input_features = _positive_int(model.get("input_features"), f"{model_path}.input_features", MAX_INPUT_FEATURES, issues)
    output_features = _positive_int(model.get("output_features"), f"{model_path}.output_features", MAX_OUTPUT_FEATURES, issues)
    has_bias = model.get("has_bias")
    if not isinstance(has_bias, bool):
        issues.append(_issue("TYPE", f"{model_path}.has_bias", "必须是布尔值。"))
    if model.get("dtype") != "float64":
        issues.append(_issue("CONST", f"{model_path}.dtype", "当前版本只支持 float64。"))

    if objective.get("loss") != "sum_squared_error":
        issues.append(_issue("UNSUPPORTED_OBJECTIVE", f"{objective_path}.loss", "解析任务头只支持平方误差总和。"))
    if objective.get("regularization") != "l2":
        issues.append(_issue("UNSUPPORTED_OBJECTIVE", f"{objective_path}.regularization", "解析任务头只支持 L2 参数正则。"))
    strength = objective.get("regularization_strength")
    if not _finite_number(strength) or float(strength) < 0:
        issues.append(_issue("RANGE", f"{objective_path}.regularization_strength", "必须是非负有限数值。"))
    regularize_bias = objective.get("regularize_bias")
    if not isinstance(regularize_bias, bool):
        issues.append(_issue("TYPE", f"{objective_path}.regularize_bias", "必须是布尔值。"))

    sample_count = statistics.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        issues.append(_issue("RANGE", f"{statistics_path}.sample_count", "必须是正整数。"))

    if issues or input_features is None or output_features is None or not isinstance(has_bias, bool) or not isinstance(regularize_bias, bool):
        return None, None, None, None, None

    design_features = input_features + (1 if has_bias else 0)
    if statistics.get("base_feature_dim") != input_features:
        issues.append(_issue("DIMENSION_MISMATCH", f"{statistics_path}.base_feature_dim", "必须与 model.input_features 一致。"))
    if statistics.get("design_feature_dim") != design_features:
        issues.append(_issue("DIMENSION_MISMATCH", f"{statistics_path}.design_feature_dim", "与bias设置对应的设计矩阵维度不一致。"))
    expected_bias_position = "last" if has_bias else None
    if statistics.get("bias_feature_position") != expected_bias_position:
        issues.append(_issue("BIAS_POSITION_MISMATCH", f"{statistics_path}.bias_feature_position", "带bias时必须为 last，不带bias时必须为 null。"))
    gram = _matrix(statistics.get("G"), design_features, design_features, f"{statistics_path}.G", issues)
    cross = _matrix(statistics.get("C"), design_features, output_features, f"{statistics_path}.C", issues)
    if issues or gram is None or cross is None:
        return None, None, None, None, None

    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-10):
        issues.append(_issue("G_NOT_SYMMETRIC", f"{statistics_path}.G", "G 矩阵必须对称。"))
    else:
        minimum_eigenvalue = float(np.linalg.eigvalsh((gram + gram.T) / 2.0).min())
        tolerance = 1e-9 * max(1.0, float(np.linalg.norm(gram, ord=2)))
        if minimum_eigenvalue < -tolerance:
            issues.append(_issue("G_NOT_PSD", f"{statistics_path}.G", "G 矩阵不是半正定矩阵。"))
    if issues:
        return None, None, None, None, None

    coefficients, condition_number, solve_issue = _solve(
        gram,
        cross,
        float(strength),
        has_bias,
        regularize_bias,
        statistics_path,
    )
    if solve_issue is not None or coefficients is None:
        issues.append(solve_issue or _issue("NUMERICAL_FAILURE", statistics_path, "无法求解解析任务头参数。"))
        return None, None, None, None, None

    normalized_model = copy.deepcopy(model)
    normalized_objective = copy.deepcopy(objective)
    normalized_objective["regularization_strength"] = float(strength)
    normalized_statistics = copy.deepcopy(statistics)
    normalized_statistics["G"] = gram.tolist()
    normalized_statistics["C"] = cross.tolist()
    diagnostics = {
        "sample_count": int(sample_count),
        "input_features": input_features,
        "output_features": output_features,
        "has_bias": has_bias,
        "condition_number": condition_number,
        "ill_conditioned": bool(condition_number is not None and condition_number > 1e12),
    }
    return normalized_model, normalized_objective, normalized_statistics, coefficients, diagnostics


def _validate_state(state: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    root_keys = {
        "report_type", "schema_version", "method_id", "method_version", "state_version",
        "model", "objective", "statistics", "parameters", "model_binding", "lineage",
    }
    root = _exact_keys(state, "$", root_keys, issues)
    if root is None:
        return None, None, issues
    constants = {
        "report_type": "IncrementalRidgeState",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
    }
    for key, expected in constants.items():
        if root.get(key) != expected:
            issues.append(_issue("CONST", f"$.{key}", f"必须为 {expected!r}。"))
    if isinstance(root.get("state_version"), bool) or not isinstance(root.get("state_version"), int) or root.get("state_version", 0) < 1:
        issues.append(_issue("RANGE", "$.state_version", "必须是从 1 开始的整数。"))

    model, objective, statistics, coefficients, diagnostics = _validate_problem(
        root.get("model"),
        root.get("objective"),
        root.get("statistics"),
        "$",
        issues,
    )
    parameters = _exact_keys(root.get("parameters"), "$.parameters", {"weight", "bias"}, issues)
    binding = _exact_keys(root.get("model_binding"), "$.model_binding", {"parameter_sha256", "artifact_ref", "artifact_sha256"}, issues)

    if any(item is None for item in (model, objective, statistics, coefficients, diagnostics, parameters, binding)):
        return None, None, issues

    input_features = model["input_features"]
    output_features = model["output_features"]
    has_bias = model["has_bias"]

    _sha256(binding.get("parameter_sha256"), "$.model_binding.parameter_sha256", issues)
    artifact_ref = binding.get("artifact_ref")
    if artifact_ref is not None and (not isinstance(artifact_ref, str) or not artifact_ref.strip()):
        issues.append(_issue("TYPE", "$.model_binding.artifact_ref", "必须是非空字符串或 null。"))
    artifact_sha = binding.get("artifact_sha256")
    if artifact_sha is not None:
        _sha256(artifact_sha, "$.model_binding.artifact_sha256", issues)

    lineage = root.get("lineage")
    if lineage is not None:
        lineage_obj = _exact_keys(lineage, "$.lineage", {"previous_state_sha256", "update_request_sha256"}, issues)
        if lineage_obj is not None:
            _sha256(lineage_obj.get("previous_state_sha256"), "$.lineage.previous_state_sha256", issues)
            _sha256(lineage_obj.get("update_request_sha256"), "$.lineage.update_request_sha256", issues)

    supplied_weight = _matrix(parameters.get("weight"), output_features, input_features, "$.parameters.weight", issues)
    supplied_bias = _vector_or_null(parameters.get("bias"), output_features, "$.parameters.bias", has_bias, issues)
    if issues or supplied_weight is None:
        return None, None, issues
    expected_parameters = _parameters(coefficients, input_features, output_features, has_bias)
    expected_weight = np.asarray(expected_parameters["weight"], dtype=np.float64)
    if not np.allclose(supplied_weight, expected_weight, rtol=PARAMETER_RTOL, atol=PARAMETER_ATOL):
        issues.append(_issue("PARAMETER_STATE_MISMATCH", "$.parameters.weight", "权重与充分统计量及固定目标函数的解不一致。"))
    if has_bias:
        expected_bias = np.asarray(expected_parameters["bias"], dtype=np.float64)
        if supplied_bias is None or not np.allclose(supplied_bias, expected_bias, rtol=PARAMETER_RTOL, atol=PARAMETER_ATOL):
            issues.append(_issue("PARAMETER_STATE_MISMATCH", "$.parameters.bias", "bias 与充分统计量及固定目标函数的解不一致。"))

    normalized_parameters = {
        "weight": supplied_weight.tolist(),
        "bias": supplied_bias.tolist() if supplied_bias is not None else None,
    }
    parameter_digest = canonical_sha256(normalized_parameters)
    if binding.get("parameter_sha256") != parameter_digest:
        issues.append(_issue("PARAMETER_DIGEST_MISMATCH", "$.model_binding.parameter_sha256", "参数摘要与当前参数不一致。"))
    if issues:
        return None, None, issues

    normalized = copy.deepcopy(root)
    normalized["model"] = model
    normalized["objective"] = objective
    normalized["statistics"] = statistics
    normalized["parameters"] = normalized_parameters
    return normalized, diagnostics, issues


def _report(action: str, status: str, issues: list[dict[str, str]], state: dict[str, Any] | None, summary: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "report_type": "IncrementalRidgeToolReport",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "action": action,
        "status": status,
        "issues": issues,
        "state_sha256": canonical_sha256(state) if state is not None else None,
        "state": state,
        "summary": summary,
    }


def initialize_incremental_ridge(request: Any) -> dict[str, Any]:
    """Initialize an analytic task head from historical G and C statistics."""
    issues: list[dict[str, str]] = []
    expected = {
        "report_type", "schema_version", "method_id", "method_version",
        "plan_binding", "model", "objective", "statistics",
    }
    root = _exact_keys(request, "$", expected, issues)
    if root is None:
        return _report("initialize", "rejected", issues, None, None)
    constants = {
        "report_type": "IncrementalRidgeInitializationRequest",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
    }
    for key, expected_value in constants.items():
        if root.get(key) != expected_value:
            issues.append(_issue("CONST", f"$.{key}", f"必须为 {expected_value!r}。"))

    plan = _validate_plan(root.get("plan_binding"), "$.plan_binding", issues)
    model, objective, statistics, coefficients, diagnostics = _validate_problem(
        root.get("model"),
        root.get("objective"),
        root.get("statistics"),
        "$",
        issues,
    )
    if issues or any(item is None for item in (plan, model, objective, statistics, coefficients, diagnostics)):
        return _report("initialize", "rejected", issues, None, None)

    parameters = _parameters(
        coefficients,
        model["input_features"],
        model["output_features"],
        model["has_bias"],
    )
    state = {
        "report_type": "IncrementalRidgeState",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
        "state_version": 1,
        "model": model,
        "objective": objective,
        "statistics": statistics,
        "parameters": parameters,
        "model_binding": {
            "parameter_sha256": canonical_sha256(parameters),
            "artifact_ref": None,
            "artifact_sha256": None,
        },
        "lineage": None,
    }
    normalized, state_diagnostics, state_issues = _validate_state(state)
    if state_issues or normalized is None or state_diagnostics is None:
        return _report("initialize", "rejected", state_issues, None, None)
    summary = {
        "task_id": plan["task_id"],
        "plan_version": plan["plan_version"],
        "sample_count": diagnostics["sample_count"],
        "input_features": diagnostics["input_features"],
        "output_features": diagnostics["output_features"],
        "has_bias": diagnostics["has_bias"],
        "condition_number": diagnostics["condition_number"],
        "ill_conditioned": diagnostics["ill_conditioned"],
        "initialized_from_statistics": True,
        "original_task_head_parameters_used": False,
        "initialization_request_sha256": canonical_sha256(root),
    }
    return _report("initialize", "completed", [], normalized, summary)


def inspect_state(state: Any) -> dict[str, Any]:
    normalized, diagnostics, issues = _validate_state(state)
    return _report("inspect", "passed" if not issues else "rejected", issues, normalized, diagnostics)


def update_incremental_ridge(request: Any) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    expected = {"report_type", "schema_version", "method_id", "method_version", "plan_binding", "old_state", "new_batch"}
    root = _exact_keys(request, "$", expected, issues)
    if root is None:
        return _report("update", "rejected", issues, None, None)
    constants = {
        "report_type": "IncrementalRidgeUpdateRequest",
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
    }
    for key, expected_value in constants.items():
        if root.get(key) != expected_value:
            issues.append(_issue("CONST", f"$.{key}", f"必须为 {expected_value!r}。"))

    plan = _validate_plan(root.get("plan_binding"), "$.plan_binding", issues)
    batch = _exact_keys(root.get("new_batch"), "$.new_batch", {"features", "targets"}, issues)

    old_state, old_diagnostics, state_issues = _validate_state(root.get("old_state"))
    issues.extend({**item, "path": "$.old_state" + item["path"][1:]} for item in state_issues)
    if issues or batch is None or old_state is None or old_diagnostics is None:
        return _report("update", "rejected", issues, None, None)

    input_features = old_state["model"]["input_features"]
    output_features = old_state["model"]["output_features"]
    features_value = batch.get("features")
    rows = len(features_value) if isinstance(features_value, list) else 0
    if rows < 1:
        issues.append(_issue("EMPTY_BATCH", "$.new_batch.features", "新批次至少需要一个样本。"))
    features = _matrix(features_value, rows, input_features, "$.new_batch.features", issues) if rows else None
    targets = _matrix(batch.get("targets"), rows, output_features, "$.new_batch.targets", issues) if rows else None
    if issues or features is None or targets is None:
        return _report("update", "rejected", issues, None, None)

    has_bias = old_state["model"]["has_bias"]
    design = np.concatenate([features, np.ones((rows, 1), dtype=np.float64)], axis=1) if has_bias else features
    old_gram = np.asarray(old_state["statistics"]["G"], dtype=np.float64)
    old_cross = np.asarray(old_state["statistics"]["C"], dtype=np.float64)
    new_gram = old_gram + design.T @ design
    new_cross = old_cross + design.T @ targets
    objective = old_state["objective"]
    coefficients, condition_number, solve_issue = _solve(
        new_gram,
        new_cross,
        objective["regularization_strength"],
        has_bias,
        objective["regularize_bias"],
    )
    if solve_issue is not None or coefficients is None:
        return _report("update", "rejected", [solve_issue or _issue("NUMERICAL_FAILURE", "$.new_batch", "无法完成更新。")], None, None)

    new_parameters = _parameters(coefficients, input_features, output_features, has_bias)
    old_parameters = old_state["parameters"]
    old_weight = np.asarray(old_parameters["weight"], dtype=np.float64)
    old_bias = np.asarray(old_parameters["bias"], dtype=np.float64) if has_bias else np.zeros(output_features)
    new_weight = np.asarray(new_parameters["weight"], dtype=np.float64)
    new_bias = np.asarray(new_parameters["bias"], dtype=np.float64) if has_bias else np.zeros(output_features)
    delta_weight = new_weight - old_weight
    delta_bias = new_bias - old_bias if has_bias else None
    before = features @ old_weight.T + old_bias
    after = features @ new_weight.T + new_bias
    rmse_before = float(np.sqrt(np.mean((targets - before) ** 2)))
    rmse_after = float(np.sqrt(np.mean((targets - after) ** 2)))
    delta_parts = [np.ravel(delta_weight)]
    if has_bias:
        assert delta_bias is not None
        delta_parts.append(np.ravel(delta_bias))
    parameter_delta_l2 = float(np.linalg.norm(np.concatenate(delta_parts)))

    old_state_digest = canonical_sha256(old_state)
    request_digest = canonical_sha256(root)
    new_state = copy.deepcopy(old_state)
    new_state["state_version"] += 1
    new_state["statistics"] = {
        "sample_count": old_state["statistics"]["sample_count"] + rows,
        "base_feature_dim": old_state["statistics"]["base_feature_dim"],
        "design_feature_dim": old_state["statistics"]["design_feature_dim"],
        "bias_feature_position": old_state["statistics"]["bias_feature_position"],
        "G": new_gram.tolist(),
        "C": new_cross.tolist(),
    }
    new_state["parameters"] = new_parameters
    new_state["model_binding"] = {
        "parameter_sha256": canonical_sha256(new_parameters),
        "artifact_ref": None,
        "artifact_sha256": None,
    }
    new_state["lineage"] = {
        "previous_state_sha256": old_state_digest,
        "update_request_sha256": request_digest,
    }
    summary = {
        "task_id": plan["task_id"] if plan is not None else None,
        "plan_version": plan["plan_version"] if plan is not None else None,
        "old_state_version": old_state["state_version"],
        "new_state_version": new_state["state_version"],
        "old_sample_count": old_state["statistics"]["sample_count"],
        "new_batch_sample_count": rows,
        "total_sample_count": new_state["statistics"]["sample_count"],
        "new_batch_rmse_before": rmse_before,
        "new_batch_rmse_after": rmse_after,
        "parameter_delta": {
            "weight": delta_weight.tolist(),
            "bias": delta_bias.tolist() if delta_bias is not None else None,
        },
        "parameter_delta_l2": parameter_delta_l2,
        "condition_number": condition_number,
        "ill_conditioned": bool(condition_number is not None and condition_number > 1e12),
        "old_state_sha256": old_state_digest,
        "original_state_mutated": False,
        "objective_changed": False,
    }
    return _report("update", "completed", [], new_state, summary)
