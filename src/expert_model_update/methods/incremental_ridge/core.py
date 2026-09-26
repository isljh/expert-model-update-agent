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


SCHEMA_VERSION = "0.1"
METHOD_ID = "incremental_ridge_closed_form"
METHOD_VERSION = "0.1"
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


def _vector_or_null(value: Any, length: int, path: str, fit_intercept: bool, issues: list[dict[str, str]]) -> np.ndarray | None:
    if not fit_intercept:
        if value is not None:
            issues.append(_issue("INTERCEPT", path, "fit_intercept=false 时 bias 必须为 null。"))
        return None
    if not isinstance(value, list) or len(value) != length or not all(_finite_number(item) for item in value):
        issues.append(_issue("SHAPE", path, f"必须是包含 {length} 个有限数值的数组。"))
        return None
    return np.asarray([float(item) for item in value], dtype=np.float64)


def _regularization_matrix(dimension: int, fit_intercept: bool, regularize_intercept: bool) -> np.ndarray:
    diagonal = np.ones(dimension, dtype=np.float64)
    if fit_intercept and not regularize_intercept:
        diagonal[-1] = 0.0
    return np.diag(diagonal)


def _solve(gram: np.ndarray, cross: np.ndarray, strength: float, fit_intercept: bool, regularize_intercept: bool) -> tuple[np.ndarray | None, float | None, dict[str, str] | None]:
    system = gram + strength * _regularization_matrix(gram.shape[0], fit_intercept, regularize_intercept)
    try:
        condition_number = float(np.linalg.cond(system))
        coefficients = np.linalg.solve(system, cross)
    except np.linalg.LinAlgError as exc:
        return None, None, _issue("SINGULAR_SYSTEM", "$.statistics", f"正则化法方程不可解：{exc}")
    if not np.isfinite(coefficients).all() or not math.isfinite(condition_number):
        return None, None, _issue("NUMERICAL_FAILURE", "$.statistics", "求解结果或条件数不是有限数值。")
    return coefficients, condition_number, None


def _parameters(coefficients: np.ndarray, input_features: int, output_features: int, fit_intercept: bool) -> dict[str, Any]:
    if fit_intercept:
        weight = coefficients[:input_features, :].T
        bias: list[float] | None = coefficients[input_features, :].tolist()
    else:
        weight = coefficients.T
        bias = None
    return {"weight": weight.tolist(), "bias": bias}


def _validate_state(state: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    root_keys = {
        "report_type", "schema_version", "method_id", "method_version", "state_version",
        "model", "objective", "feature_contract", "statistics", "parameters", "model_binding", "lineage",
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

    model = _exact_keys(root.get("model"), "$.model", {"input_features", "output_features", "fit_intercept", "dtype"}, issues)
    objective = _exact_keys(root.get("objective"), "$.objective", {"loss", "regularization", "regularization_strength", "regularize_intercept"}, issues)
    feature_contract = _exact_keys(root.get("feature_contract"), "$.feature_contract", {"feature_schema_sha256", "target_schema_sha256", "preprocessing_ref"}, issues)
    statistics = _exact_keys(root.get("statistics"), "$.statistics", {"sample_count", "gram_matrix", "feature_target_cross_product"}, issues)
    parameters = _exact_keys(root.get("parameters"), "$.parameters", {"weight", "bias"}, issues)
    binding = _exact_keys(root.get("model_binding"), "$.model_binding", {"parameter_sha256", "artifact_ref", "artifact_sha256"}, issues)

    if any(item is None for item in (model, objective, feature_contract, statistics, parameters, binding)):
        return None, None, issues

    input_features = _positive_int(model.get("input_features"), "$.model.input_features", MAX_INPUT_FEATURES, issues)
    output_features = _positive_int(model.get("output_features"), "$.model.output_features", MAX_OUTPUT_FEATURES, issues)
    fit_intercept = model.get("fit_intercept")
    if not isinstance(fit_intercept, bool):
        issues.append(_issue("TYPE", "$.model.fit_intercept", "必须是布尔值。"))
    if model.get("dtype") != "float64":
        issues.append(_issue("CONST", "$.model.dtype", "第一版只支持 float64。"))

    if objective.get("loss") != "sum_squared_error":
        issues.append(_issue("UNSUPPORTED_OBJECTIVE", "$.objective.loss", "该方法只支持平方误差总和。"))
    if objective.get("regularization") != "l2":
        issues.append(_issue("UNSUPPORTED_OBJECTIVE", "$.objective.regularization", "该方法只支持 L2 参数正则。"))
    strength = objective.get("regularization_strength")
    if not _finite_number(strength) or float(strength) < 0:
        issues.append(_issue("RANGE", "$.objective.regularization_strength", "必须是非负有限数值。"))
    regularize_intercept = objective.get("regularize_intercept")
    if not isinstance(regularize_intercept, bool):
        issues.append(_issue("TYPE", "$.objective.regularize_intercept", "必须是布尔值。"))

    feature_digest = _sha256(feature_contract.get("feature_schema_sha256"), "$.feature_contract.feature_schema_sha256", issues)
    target_digest = _sha256(feature_contract.get("target_schema_sha256"), "$.feature_contract.target_schema_sha256", issues)
    preprocessing_ref = feature_contract.get("preprocessing_ref")
    if preprocessing_ref is not None and (not isinstance(preprocessing_ref, str) or not preprocessing_ref.strip()):
        issues.append(_issue("TYPE", "$.feature_contract.preprocessing_ref", "必须是非空字符串或 null。"))

    sample_count = statistics.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        issues.append(_issue("RANGE", "$.statistics.sample_count", "必须是正整数。"))

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

    if issues or input_features is None or output_features is None or not isinstance(fit_intercept, bool) or not isinstance(regularize_intercept, bool):
        return None, None, issues

    augmented = input_features + (1 if fit_intercept else 0)
    gram = _matrix(statistics.get("gram_matrix"), augmented, augmented, "$.statistics.gram_matrix", issues)
    cross = _matrix(statistics.get("feature_target_cross_product"), augmented, output_features, "$.statistics.feature_target_cross_product", issues)
    supplied_weight = _matrix(parameters.get("weight"), output_features, input_features, "$.parameters.weight", issues)
    supplied_bias = _vector_or_null(parameters.get("bias"), output_features, "$.parameters.bias", fit_intercept, issues)
    if issues or gram is None or cross is None or supplied_weight is None:
        return None, None, issues

    if not np.allclose(gram, gram.T, rtol=0.0, atol=1e-10):
        issues.append(_issue("GRAM_NOT_SYMMETRIC", "$.statistics.gram_matrix", "Gram 矩阵必须对称。"))
    else:
        minimum_eigenvalue = float(np.linalg.eigvalsh((gram + gram.T) / 2.0).min())
        tolerance = 1e-9 * max(1.0, float(np.linalg.norm(gram, ord=2)))
        if minimum_eigenvalue < -tolerance:
            issues.append(_issue("GRAM_NOT_PSD", "$.statistics.gram_matrix", "Gram 矩阵不是半正定矩阵。"))
    if issues:
        return None, None, issues

    coefficients, condition_number, solve_issue = _solve(
        gram, cross, float(strength), fit_intercept, regularize_intercept
    )
    if solve_issue is not None or coefficients is None:
        issues.append(solve_issue or _issue("NUMERICAL_FAILURE", "$.statistics", "无法求解参数。"))
        return None, None, issues
    expected_parameters = _parameters(coefficients, input_features, output_features, fit_intercept)
    expected_weight = np.asarray(expected_parameters["weight"], dtype=np.float64)
    if not np.allclose(supplied_weight, expected_weight, rtol=PARAMETER_RTOL, atol=PARAMETER_ATOL):
        issues.append(_issue("PARAMETER_STATE_MISMATCH", "$.parameters.weight", "权重与充分统计量及固定目标函数的解不一致。"))
    if fit_intercept:
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
    normalized["objective"]["regularization_strength"] = float(strength)
    normalized["statistics"]["gram_matrix"] = gram.tolist()
    normalized["statistics"]["feature_target_cross_product"] = cross.tolist()
    normalized["parameters"] = normalized_parameters
    diagnostics = {
        "sample_count": int(sample_count),
        "input_features": input_features,
        "output_features": output_features,
        "condition_number": condition_number,
        "ill_conditioned": bool(condition_number is not None and condition_number > 1e12),
        "feature_schema_sha256": feature_digest,
        "target_schema_sha256": target_digest,
    }
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

    plan = _exact_keys(root.get("plan_binding"), "$.plan_binding", {"task_id", "plan_version", "update_plan_sha256"}, issues)
    batch = _exact_keys(root.get("new_batch"), "$.new_batch", {"feature_schema_sha256", "target_schema_sha256", "features", "targets"}, issues)
    if plan is not None:
        if not isinstance(plan.get("task_id"), str) or not plan.get("task_id", "").strip():
            issues.append(_issue("TYPE", "$.plan_binding.task_id", "必须是非空字符串。"))
        if isinstance(plan.get("plan_version"), bool) or not isinstance(plan.get("plan_version"), int) or plan.get("plan_version", 0) < 1:
            issues.append(_issue("RANGE", "$.plan_binding.plan_version", "必须是从 1 开始的整数。"))
        _sha256(plan.get("update_plan_sha256"), "$.plan_binding.update_plan_sha256", issues)

    old_state, old_diagnostics, state_issues = _validate_state(root.get("old_state"))
    issues.extend({**item, "path": "$.old_state" + item["path"][1:]} for item in state_issues)
    if issues or batch is None or old_state is None or old_diagnostics is None:
        return _report("update", "rejected", issues, None, None)

    feature_digest = _sha256(batch.get("feature_schema_sha256"), "$.new_batch.feature_schema_sha256", issues)
    target_digest = _sha256(batch.get("target_schema_sha256"), "$.new_batch.target_schema_sha256", issues)
    if feature_digest != old_state["feature_contract"]["feature_schema_sha256"]:
        issues.append(_issue("FEATURE_CONTRACT_MISMATCH", "$.new_batch.feature_schema_sha256", "新批次特征契约与旧状态不一致。"))
    if target_digest != old_state["feature_contract"]["target_schema_sha256"]:
        issues.append(_issue("TARGET_CONTRACT_MISMATCH", "$.new_batch.target_schema_sha256", "新批次目标契约与旧状态不一致。"))

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

    fit_intercept = old_state["model"]["fit_intercept"]
    design = np.concatenate([features, np.ones((rows, 1), dtype=np.float64)], axis=1) if fit_intercept else features
    old_gram = np.asarray(old_state["statistics"]["gram_matrix"], dtype=np.float64)
    old_cross = np.asarray(old_state["statistics"]["feature_target_cross_product"], dtype=np.float64)
    new_gram = old_gram + design.T @ design
    new_cross = old_cross + design.T @ targets
    objective = old_state["objective"]
    coefficients, condition_number, solve_issue = _solve(
        new_gram,
        new_cross,
        objective["regularization_strength"],
        fit_intercept,
        objective["regularize_intercept"],
    )
    if solve_issue is not None or coefficients is None:
        return _report("update", "rejected", [solve_issue or _issue("NUMERICAL_FAILURE", "$.new_batch", "无法完成更新。")], None, None)

    new_parameters = _parameters(coefficients, input_features, output_features, fit_intercept)
    old_parameters = old_state["parameters"]
    old_weight = np.asarray(old_parameters["weight"], dtype=np.float64)
    old_bias = np.asarray(old_parameters["bias"], dtype=np.float64) if fit_intercept else np.zeros(output_features)
    new_weight = np.asarray(new_parameters["weight"], dtype=np.float64)
    new_bias = np.asarray(new_parameters["bias"], dtype=np.float64) if fit_intercept else np.zeros(output_features)
    before = features @ old_weight.T + old_bias
    after = features @ new_weight.T + new_bias
    rmse_before = float(np.sqrt(np.mean((targets - before) ** 2)))
    rmse_after = float(np.sqrt(np.mean((targets - after) ** 2)))
    delta_parts = [np.ravel(new_weight - old_weight)]
    if fit_intercept:
        delta_parts.append(np.ravel(new_bias - old_bias))
    parameter_delta_l2 = float(np.linalg.norm(np.concatenate(delta_parts)))

    old_state_digest = canonical_sha256(old_state)
    request_digest = canonical_sha256(root)
    new_state = copy.deepcopy(old_state)
    new_state["state_version"] += 1
    new_state["statistics"] = {
        "sample_count": old_state["statistics"]["sample_count"] + rows,
        "gram_matrix": new_gram.tolist(),
        "feature_target_cross_product": new_cross.tolist(),
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
        "parameter_delta_l2": parameter_delta_l2,
        "condition_number": condition_number,
        "ill_conditioned": bool(condition_number is not None and condition_number > 1e12),
        "old_state_sha256": old_state_digest,
        "original_state_mutated": False,
        "objective_changed": False,
    }
    return _report("update", "completed", [], new_state, summary)
