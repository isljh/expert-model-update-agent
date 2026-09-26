"""Controlled contract tests for the closed-form incremental ridge package."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert_model_update.methods.incremental_ridge.core import (
    canonical_sha256,
    inspect_state,
    update_incremental_ridge,
)


FEATURE_DIGEST = "1" * 64
TARGET_DIGEST = "2" * 64
PLAN_DIGEST = "3" * 64


def solve_parameters(features: np.ndarray, targets: np.ndarray, strength: float, fit_intercept: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1) if fit_intercept else features
    regularizer = np.eye(design.shape[1])
    if fit_intercept:
        regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + strength * regularizer, design.T @ targets)
    if fit_intercept:
        return coefficients[:-1].T, coefficients[-1]
    return coefficients.T, None


def state_from_data(features: np.ndarray, targets: np.ndarray, strength: float = 0.25, fit_intercept: bool = True) -> dict:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1) if fit_intercept else features
    weight, bias = solve_parameters(features, targets, strength, fit_intercept)
    parameters = {"weight": weight.tolist(), "bias": bias.tolist() if bias is not None else None}
    return {
        "report_type": "IncrementalRidgeState",
        "schema_version": "0.1",
        "method_id": "incremental_ridge_closed_form",
        "method_version": "0.1",
        "state_version": 1,
        "model": {
            "input_features": features.shape[1],
            "output_features": targets.shape[1],
            "fit_intercept": fit_intercept,
            "dtype": "float64",
        },
        "objective": {
            "loss": "sum_squared_error",
            "regularization": "l2",
            "regularization_strength": strength,
            "regularize_intercept": False,
        },
        "feature_contract": {
            "feature_schema_sha256": FEATURE_DIGEST,
            "target_schema_sha256": TARGET_DIGEST,
            "preprocessing_ref": "preprocessing-v1",
        },
        "statistics": {
            "sample_count": features.shape[0],
            "gram_matrix": (design.T @ design).tolist(),
            "feature_target_cross_product": (design.T @ targets).tolist(),
        },
        "parameters": parameters,
        "model_binding": {
            "parameter_sha256": canonical_sha256(parameters),
            "artifact_ref": "old-model.json",
            "artifact_sha256": "4" * 64,
        },
        "lineage": None,
    }


def request(old_state: dict, features: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "report_type": "IncrementalRidgeUpdateRequest",
        "schema_version": "0.1",
        "method_id": "incremental_ridge_closed_form",
        "method_version": "0.1",
        "plan_binding": {
            "task_id": "linear-update-001",
            "plan_version": 1,
            "update_plan_sha256": PLAN_DIGEST,
        },
        "old_state": old_state,
        "new_batch": {
            "feature_schema_sha256": FEATURE_DIGEST,
            "target_schema_sha256": TARGET_DIGEST,
            "features": features.tolist(),
            "targets": targets.tolist(),
        },
    }


class IncrementalRidgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_x = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [3.0, -1.0]], dtype=np.float64)
        self.old_y = np.asarray([[1.0], [2.0], [5.0], [5.5]], dtype=np.float64)
        self.new_x = np.asarray([[4.0, 2.0], [-1.0, 3.0]], dtype=np.float64)
        self.new_y = np.asarray([[9.0], [0.5]], dtype=np.float64)
        self.state = state_from_data(self.old_x, self.old_y)

    def test_valid_state_passes_inspection(self) -> None:
        report = inspect_state(self.state)
        self.assertEqual(report["status"], "passed", report)
        self.assertEqual(report["summary"]["sample_count"], 4)
        self.assertEqual(report["state_sha256"], canonical_sha256(report["state"]))

    def test_method_card_has_expected_identity_and_fixed_objective(self) -> None:
        path = ROOT / "method-packages" / "incremental-ridge-closed-form" / "method-card.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(card["report_type"], "IncrementalLearningMethodCard")
        self.assertEqual(card["method_id"], "incremental_ridge_closed_form")
        self.assertEqual(
            card["decision_contract"]["fixed_objective"],
            "sum_squared_error + lambda * l2_weight_penalty",
        )

    def test_incremental_solution_matches_joint_solution(self) -> None:
        original = copy.deepcopy(self.state)
        report = update_incremental_ridge(request(self.state, self.new_x, self.new_y))
        self.assertEqual(report["status"], "completed", report)
        joint_x = np.concatenate([self.old_x, self.new_x], axis=0)
        joint_y = np.concatenate([self.old_y, self.new_y], axis=0)
        expected_weight, expected_bias = solve_parameters(joint_x, joint_y, 0.25)
        np.testing.assert_allclose(report["state"]["parameters"]["weight"], expected_weight, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(report["state"]["parameters"]["bias"], expected_bias, rtol=1e-10, atol=1e-12)
        self.assertEqual(report["state"]["statistics"]["sample_count"], 6)
        self.assertEqual(report["state"]["state_version"], 2)
        self.assertEqual(report["state"]["lineage"]["previous_state_sha256"], canonical_sha256(original))
        self.assertEqual(self.state, original)
        self.assertFalse(report["summary"]["original_state_mutated"])
        self.assertFalse(report["summary"]["objective_changed"])

    def test_rejects_other_loss(self) -> None:
        altered = copy.deepcopy(self.state)
        altered["objective"]["loss"] = "mean_absolute_error"
        report = inspect_state(altered)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("UNSUPPORTED_OBJECTIVE", {item["code"] for item in report["issues"]})

    def test_rejects_tampered_parameters(self) -> None:
        altered = copy.deepcopy(self.state)
        altered["parameters"]["weight"][0][0] += 1.0
        altered["model_binding"]["parameter_sha256"] = canonical_sha256(altered["parameters"])
        report = inspect_state(altered)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("PARAMETER_STATE_MISMATCH", {item["code"] for item in report["issues"]})

    def test_rejects_feature_contract_change(self) -> None:
        payload = request(self.state, self.new_x, self.new_y)
        payload["new_batch"]["feature_schema_sha256"] = "9" * 64
        report = update_incremental_ridge(payload)
        self.assertEqual(report["status"], "rejected")
        self.assertIn("FEATURE_CONTRACT_MISMATCH", {item["code"] for item in report["issues"]})

    def test_lambda_zero_ordinary_least_squares(self) -> None:
        old_x = np.asarray([[0.0], [1.0], [2.0]], dtype=np.float64)
        old_y = np.asarray([[1.0], [3.0], [5.0]], dtype=np.float64)
        new_x = np.asarray([[3.0]], dtype=np.float64)
        new_y = np.asarray([[7.0]], dtype=np.float64)
        old_state = state_from_data(old_x, old_y, strength=0.0)
        report = update_incremental_ridge(request(old_state, new_x, new_y))
        self.assertEqual(report["status"], "completed", report)
        np.testing.assert_allclose(report["state"]["parameters"]["weight"], [[2.0]], atol=1e-12)
        np.testing.assert_allclose(report["state"]["parameters"]["bias"], [1.0], atol=1e-12)

    def test_model_without_intercept_keeps_bias_null(self) -> None:
        old_x = np.asarray([[1.0], [2.0], [4.0]], dtype=np.float64)
        old_y = np.asarray([[2.0], [4.0], [8.0]], dtype=np.float64)
        new_x = np.asarray([[3.0]], dtype=np.float64)
        new_y = np.asarray([[6.0]], dtype=np.float64)
        old_state = state_from_data(old_x, old_y, strength=0.0, fit_intercept=False)
        report = update_incremental_ridge(request(old_state, new_x, new_y))
        self.assertEqual(report["status"], "completed", report)
        np.testing.assert_allclose(report["state"]["parameters"]["weight"], [[2.0]], atol=1e-12)
        self.assertIsNone(report["state"]["parameters"]["bias"])

    def test_cli_returns_machine_report(self) -> None:
        cli = ROOT / "src" / "expert_model_update" / "methods" / "incremental_ridge" / "cli.py"
        completed = subprocess.run(
            [sys.executable, "-B", "-X", "utf8", str(cli), "inspect", "--stdin"],
            input=json.dumps(self.state, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "passed", report)


if __name__ == "__main__":
    unittest.main()
