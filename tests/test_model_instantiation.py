"""Contract tests use a disposable, benign three-file model delivery."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert_model_update.tools.model_creation.inspect_declared_asset import inspect_declared_asset
from expert_model_update.tools.model_creation.instantiate import _run_bounded, load_validated_model
from expert_model_update.tools.model_creation.instantiate_worker import instantiate
from expert_model_update.tools.model_creation.validate import validate_model_delivery
from expert_model_update.tools.task_assets.preflight import preflight_task_assets


def task_input(delivery: Path, new_data: Path) -> dict:
    return {
        "schema_version": "0.3",
        "task_id": "T1",
        "task_input_version": 1,
        "request": {
            "raw_prompt": "更新模型",
            "update_goal": {"summary": "学习新数据", "reason": "业务变化", "status": "confirmed"},
        },
        "assets": {
            "base_model": {
                "model_file": str(delivery / "model.py"),
                "config": str(delivery / "model-config.json"),
                "checkpoint": str(delivery / "weights.pt"),
            },
            "new_data": {"location": str(new_data), "allowed_uses": ["training"]},
        },
        "requirements": {
            "capabilities_to_preserve": {"mode": "none", "items": []},
            "success_expectation": "能够加载并更新",
        },
        "intake": {
            "required_questions": [],
            "optional_questions": [],
            "human_confirmation": {"status": "confirmed", "confirmed_task_input_version": 1, "notes": None},
        },
    }


class ModelInstantiationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.delivery = Path(self.scratch.name).resolve()
        self.record_scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.record_scratch.cleanup)
        self.record_root = Path(self.record_scratch.name) / "model-structure"
        (self.delivery / "new-data.json").write_text("[]", encoding="utf-8")
        (self.delivery / "model.py").write_text(
            "import torch\n"
            "class ProfessionalModel(torch.nn.Module):\n"
            "    def __init__(self, input_dim, output_dim=1):\n"
            "        super().__init__()\n"
            "        self.linear = torch.nn.Linear(input_dim, output_dim)\n"
            "    def forward(self, value):\n"
            "        return self.linear(value)\n",
            encoding="utf-8",
        )
        self.config = {
            "schema_version": "0.1",
            "framework": "pytorch",
            "model_class": "ProfessionalModel",
            "constructor_kwargs": {"input_dim": 2, "output_dim": 1},
            "checkpoint": {"state_dict_path": "$.state_dict", "key_transforms": [], "strict": True},
        }
        (self.delivery / "model-config.json").write_text(
            json.dumps(self.config),
            encoding="utf-8",
        )
        import torch
        model = torch.nn.Linear(2, 1)
        state = {
            "linear.weight": model.weight.detach().clone(),
            "linear.bias": model.bias.detach().clone(),
        }
        torch.save({"state_dict": state}, self.delivery / "weights.pt")
        self.task = task_input(self.delivery, self.delivery / "new-data.json")
        self.preflight = preflight_task_assets(self.task, base_directory=self.delivery)
        self.assertEqual(self.preflight["overall_status"], "passed", self.preflight)
        self.config_report = inspect_declared_asset(self.task, self.preflight, "config")
        self.checkpoint_report = inspect_declared_asset(self.task, self.preflight, "checkpoint")
        if self.checkpoint_report.get("status") != "inspected":
            self.skipTest("当前解释器不支持项目要求的 weights_only checkpoint 检查；请使用 model-instantiation-safe 环境。")
        self.inputs = {
            "task_input": self.task,
            "asset_preflight_report": self.preflight,
            "config_inspection_report": self.config_report,
            "checkpoint_inspection_report": self.checkpoint_report,
        }
        self.validation = validate_model_delivery(self.inputs, base_directory=self.delivery)
        self.assertTrue(self.validation["readiness"]["ready_for_model_load"], self.validation)

    def test_validation_derives_one_load_plan_without_candidates(self) -> None:
        plan = self.validation["validated_load_plan"]
        self.assertEqual(plan["model"]["class_name"], "ProfessionalModel")
        self.assertEqual(plan["model"]["constructor_kwargs"], {"input_dim": 2, "output_dim": 1})
        self.assertEqual(plan["checkpoint"]["selector_path"], "$.state_dict")
        self.assertTrue(plan["checkpoint"]["strict"])
        self.assertNotIn("proposals", self.validation)

    def test_host_loads_model_and_requires_full_checkpoint_match(self) -> None:
        report = load_validated_model(
            {"delivery_inputs": self.inputs, "validation_report": self.validation},
            base_directory=self.delivery,
        )
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["checkpoint"]["status"], "loaded")
        self.assertTrue(report["checkpoint"]["fully_matched"])
        self.assertEqual(report["model"]["parameter_count"], 3)
        self.assertEqual(report["model"]["trainable_parameter_count"], 3)
        self.assertTrue(report["model"]["has_trainable_parameters"])

    def test_worker_analyzes_loaded_model_in_same_process(self) -> None:
        report = instantiate(self.validation["validated_load_plan"], analyze_structure=True)
        self.assertEqual(report["status"], "completed", report)
        structure = report["structure_analysis"]
        self.assertEqual(structure["schema_version"], "0.2")
        self.assertEqual(structure["load_spec_id"], report["load_spec_id"])
        self.assertTrue(structure["source_model_state"]["checkpoint_fully_matched"])
        self.assertEqual(structure["parameter_summary"]["total_parameter_count"], 3)
        self.assertEqual(structure["parameter_summary"]["trainable_parameter_count"], 3)
        self.assertEqual(structure["forward_trace"]["status"], "not_run")

    def test_host_rejects_changed_validation_report(self) -> None:
        altered = copy.deepcopy(self.validation)
        altered["validated_load_plan"]["model"]["constructor_kwargs"]["input_dim"] = 3
        report = load_validated_model(
            {"delivery_inputs": self.inputs, "validation_report": altered},
            base_directory=self.delivery,
        )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error"]["code"], "VALIDATION_REPORT_MISMATCH")

    def test_missing_constructor_argument_blocks_validation(self) -> None:
        config = copy.deepcopy(self.config)
        del config["constructor_kwargs"]["input_dim"]
        (self.delivery / "model-config.json").write_text(json.dumps(config), encoding="utf-8")
        preflight = preflight_task_assets(self.task, base_directory=self.delivery)
        inputs = {
            "task_input": self.task,
            "asset_preflight_report": preflight,
            "config_inspection_report": inspect_declared_asset(self.task, preflight, "config"),
            "checkpoint_inspection_report": inspect_declared_asset(self.task, preflight, "checkpoint"),
        }
        validation = validate_model_delivery(inputs, base_directory=self.delivery)
        self.assertFalse(validation["readiness"]["ready_for_model_load"])
        self.assertIn("CONSTRUCTOR_ARGUMENT_MISSING", validation["readiness"]["blockers"])

    def test_checkpoint_shape_mismatch_fails_actual_load(self) -> None:
        import torch
        wrong = torch.nn.Linear(3, 1)
        torch.save(
            {"state_dict": {"linear.weight": wrong.weight, "linear.bias": wrong.bias}},
            self.delivery / "weights.pt",
        )
        preflight = preflight_task_assets(self.task, base_directory=self.delivery)
        inputs = {
            "task_input": self.task,
            "asset_preflight_report": preflight,
            "config_inspection_report": inspect_declared_asset(self.task, preflight, "config"),
            "checkpoint_inspection_report": inspect_declared_asset(self.task, preflight, "checkpoint"),
        }
        validation = validate_model_delivery(inputs, base_directory=self.delivery)
        self.assertTrue(validation["readiness"]["ready_for_model_load"], validation)
        report = load_validated_model(
            {"delivery_inputs": inputs, "validation_report": validation},
            base_directory=self.delivery,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["code"], "STATE_DICT_LOAD_FAILED")

    def test_host_analyzes_and_saves_outside_delivery(self) -> None:
        report = load_validated_model(
            {"delivery_inputs": self.inputs, "validation_report": self.validation},
            base_directory=self.delivery,
            analyze_structure=True,
            record_root=self.record_root,
        )
        self.assertEqual(report["status"], "completed", report)
        json_path = Path(report["record"]["json_report_path"])
        tree_path = Path(report["record"]["structure_tree_path"])
        self.assertTrue(json_path.is_file())
        self.assertTrue(tree_path.is_file())
        self.assertFalse(json_path.is_relative_to(self.delivery))
        self.assertIn("ProfessionalModel", tree_path.read_text(encoding="utf-8"))

    def test_host_refuses_report_directory_inside_delivery(self) -> None:
        report = load_validated_model(
            {"delivery_inputs": self.inputs, "validation_report": self.validation},
            base_directory=self.delivery,
            analyze_structure=True,
            record_root=self.delivery / "records",
        )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error"]["code"], "RECORD_ROOT_IN_MODEL_DELIVERY")

    def test_worker_output_is_bounded_while_running(self) -> None:
        command = [sys.executable, "-B", "-c", "import sys; sys.stdout.buffer.write(b'x' * 5000000)"]
        _, _, overflow = _run_bounded(command, b"", cwd=str(self.delivery), env=os.environ.copy())
        self.assertTrue(overflow)


if __name__ == "__main__":
    unittest.main()
