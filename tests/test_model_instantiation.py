"""Contract tests use a disposable, benign project; never a user project."""

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

from expert_model_update.tools.model_creation.instantiate import _run_bounded, _valid_worker_report, instantiate_validated_model
from expert_model_update.tools.model_creation.instantiate_worker import instantiate
from expert_model_update.tools.model_creation.validate import validate_model_creation_proposals


def proposal_set(project: Path) -> dict:
    binding = {"task_id": "T1", "task_input_version": 1, "asset_input_digest": "0" * 64}
    return {
        "report_type": "ModelCreationProposalSet",
        "schema_version": "0.2",
        "task_binding": binding,
        "asset_preflight_report": {
            "report_type": "AssetPreflightReport",
            "overall_status": "passed",
            **binding,
            "checks": [{
                "field": "assets.target_project.location",
                "status": "passed",
                "resolved_path": str(project),
            }],
        },
        "project": {"source_project_path": str(project), "source_revision": None},
        "selected_proposal_id": "P1",
        "proposals": [{
            "proposal_id": "P1",
            "construction_steps": [{
                "step_id": "base",
                "callable": {"origin": "project", "module": "toy", "symbol": "build_model"},
                "positional_arguments": [{"value": {"kind": "literal", "value": 2}, "source_expression": "2"}],
                "keyword_arguments": [],
                "source_anchor": {"path": "trainer.py", "scope": "main", "line": 3, "call_expression": "toy.build_model(2)"},
                "evidence_refs": ["E1"],
            }],
            "result": {
                "runtime_object": {"step_id": "base", "path": "$"},
                "target_model": {"step_id": "base", "path": "$"},
            },
            "pre_checkpoint_load_operations": [],
            "post_checkpoint_load_operations": [],
            "checkpoint": None,
            "evidence_refs": ["E1"],
        }],
    }


class ModelInstantiationTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("toy", None)
        self.addCleanup(lambda: sys.modules.pop("toy", None))
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.project = Path(self.scratch.name).resolve()
        self.record_scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.record_scratch.cleanup)
        self.record_root = Path(self.record_scratch.name) / "model-structure"
        (self.project / "toy.py").write_text(
            "import torch\ndef build_model(width):\n    return torch.nn.Linear(width, 1)\n",
            encoding="utf-8",
        )
        (self.project / "trainer.py").write_text(
            "import toy\ndef main():\n    model = toy.build_model(2)\n    return model\n",
            encoding="utf-8",
        )
        self.proposals = proposal_set(self.project)
        self.validation = validate_model_creation_proposals(self.proposals, base_directory=self.project)
        self.assertTrue(self.validation["readiness"]["ready_for_model_construction"], self.validation)

    def test_host_executes_benign_model_without_os_sandbox(self) -> None:
        report = instantiate_validated_model(
            {"proposal_set": self.proposals, "validation_report": self.validation},
            base_directory=self.project,
        )
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["model"]["parameter_count"], 3)
        self.assertEqual(Path(report["runtime"]["python_executable"]).resolve(), Path(sys.executable).resolve())
        self.assertEqual(report["execution_boundary"]["kind"], "local_subprocess_without_os_isolation")
        self.assertFalse(report["execution_boundary"]["project_filesystem_read_only_guaranteed"])

    def test_host_rejects_changed_validation_report(self) -> None:
        altered = copy.deepcopy(self.validation)
        altered["validated_instantiation_plan"]["construction_steps"][0]["positional_arguments"][0]["value"]["value"] = 3
        report = instantiate_validated_model(
            {"proposal_set": self.proposals, "validation_report": altered},
            base_directory=self.project,
        )
        self.assertEqual(report["error"]["code"], "VALIDATION_REPORT_MISMATCH")

    def test_benign_worker_constructs_model(self) -> None:
        # Direct worker execution is only valid here because this fixture is controlled test code.
        report = instantiate(self.validation["validated_instantiation_plan"])
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["checkpoint"]["status"], "not_provided")
        self.assertEqual(report["model"]["parameter_count"], 3)

    def test_worker_analyzes_live_model_in_same_process(self) -> None:
        report = instantiate(self.validation["validated_instantiation_plan"], analyze_structure=True)
        self.assertEqual(report["status"], "completed", report)
        structure = report["structure_analysis"]
        self.assertEqual(structure["report_type"], "ModelStructureAnalysisReport")
        self.assertEqual(structure["general"]["model_family"], "Linear")
        self.assertEqual([item["module_path"] for item in structure["module_tree"]], ["<root>"])
        self.assertEqual(structure["parameter_summary"]["total_parameter_count"], 3)
        self.assertEqual(structure["forward_trace"]["status"], "not_run")
        self.assertFalse(structure["source_model_state"]["checkpoint_parameters_loaded"])
        self.assertFalse(structure["source_model_state"]["checkpoint_fully_matched"])

    def test_host_analyzes_without_second_construction(self) -> None:
        report = instantiate_validated_model(
            {"proposal_set": self.proposals, "validation_report": self.validation},
            base_directory=self.project,
            analyze_structure=True,
            record_root=self.record_root,
        )
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual(report["steps"], [{"step_id": "base", "status": "created", "type": "Linear"}])
        self.assertEqual(report["structure_analysis"]["parameter_summary"]["total_parameter_count"], 3)
        json_path = Path(report["record"]["json_report_path"])
        tree_path = Path(report["record"]["structure_tree_path"])
        self.assertTrue(json_path.is_file())
        self.assertTrue(tree_path.is_file())
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), report)
        self.assertIn('"<root>"  Linear', tree_path.read_text(encoding="utf-8"))
        self.assertFalse(json_path.is_relative_to(self.project))
        self.assertEqual(json_path.parent, self.record_root / self.project.name)
        self.assertEqual(tree_path.parent, json_path.parent)
        self.assertEqual(report["record"]["project_name"], self.project.name)
        self.assertTrue(json_path.name.startswith(report["record"]["run_id"] + "."))
        self.assertTrue(tree_path.name.startswith(report["record"]["run_id"] + "."))
        self.assertEqual(report["project"]["source_project_path"], str(self.project))

    def test_new_runs_do_not_overwrite_prior_reports(self) -> None:
        payload = {"proposal_set": self.proposals, "validation_report": self.validation}
        first = instantiate_validated_model(payload, base_directory=self.project, analyze_structure=True, record_root=self.record_root)
        second = instantiate_validated_model(payload, base_directory=self.project, analyze_structure=True, record_root=self.record_root)
        self.assertEqual(first["status"], "completed", first)
        self.assertEqual(second["status"], "completed", second)
        self.assertNotEqual(first["record"]["run_id"], second["record"]["run_id"])
        self.assertTrue(Path(first["record"]["json_report_path"]).is_file())
        self.assertTrue(Path(second["record"]["json_report_path"]).is_file())
        self.assertEqual(
            Path(first["record"]["json_report_path"]).parent,
            Path(second["record"]["json_report_path"]).parent,
        )

    def test_host_refuses_report_directory_inside_target_project(self) -> None:
        report = instantiate_validated_model(
            {"proposal_set": self.proposals, "validation_report": self.validation},
            base_directory=self.project,
            analyze_structure=True,
            record_root=self.project / "records",
        )
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["error"]["code"], "RECORD_ROOT_IN_TARGET_PROJECT")
        self.assertFalse((self.project / "records").exists())

    def test_failed_construction_keeps_a_failure_record_without_a_tree(self) -> None:
        (self.project / "toy.py").write_text(
            "def build_model(width):\n    raise RuntimeError('controlled failure')\n",
            encoding="utf-8",
        )
        validation = validate_model_creation_proposals(self.proposals, base_directory=self.project)
        self.assertTrue(validation["readiness"]["ready_for_model_construction"], validation)
        report = instantiate_validated_model(
            {"proposal_set": self.proposals, "validation_report": validation},
            base_directory=self.project,
            analyze_structure=True,
            record_root=self.record_root,
        )
        self.assertEqual(report["status"], "failed", report)
        self.assertEqual(report["error"]["code"], "PROJECT_EXECUTION_FAILED")
        saved = json.loads(Path(report["record"]["json_report_path"]).read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        tree = Path(report["record"]["structure_tree_path"]).read_text(encoding="utf-8")
        self.assertIn("本次没有可用的完整结构树", tree)
        self.assertNotIn("## 完整注册模块树", tree)

    def test_worker_preserves_wrapper_and_target_model_paths(self) -> None:
        plan = copy.deepcopy(self.validation["validated_instantiation_plan"])
        plan["construction_steps"].append({
            "step_id": "wrapped",
            "callable": {"origin": "pytorch", "module": "torch.nn", "symbol": "DataParallel"},
            "positional_arguments": [{
                "value": {"kind": "object_ref", "object": {"step_id": "base", "path": "$"}},
                "source_expression": "model",
            }],
            "keyword_arguments": [],
        })
        plan["result"] = {
            "runtime_object": {"step_id": "wrapped", "path": "$"},
            "target_model": {"step_id": "wrapped", "path": "$.module"},
        }
        plan["project"]["step_sources"].append({
            "step_id": "wrapped",
            "callable_source": None,
            "construction_anchor_source": plan["project"]["step_sources"][0]["construction_anchor_source"],
        })
        report = instantiate(plan, analyze_structure=True)
        self.assertEqual(report["status"], "completed", report)
        self.assertEqual([step["type"] for step in report["steps"]], ["Linear", "DataParallel"])
        self.assertEqual(report["model"]["target_type"], "Linear")
        self.assertEqual(report["model"]["runtime_type"], "DataParallel")
        self.assertEqual(report["structure_analysis"]["general"]["model_class_name"], "Linear")
        self.assertEqual(report["structure_analysis"]["parameter_summary"]["total_parameter_count"], 3)

    def test_worker_detects_source_change_before_import(self) -> None:
        plan = copy.deepcopy(self.validation["validated_instantiation_plan"])
        (self.project / "toy.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
        report = instantiate(plan)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"]["code"], "SOURCE_SNAPSHOT_CHANGED")

    def test_checkpoint_load_never_falls_back_to_pickle(self) -> None:
        import torch

        checkpoint = self.project / "weights.pt"
        torch.save({"state_dict": torch.nn.Linear(2, 1).state_dict()}, checkpoint)
        plan = copy.deepcopy(self.validation["validated_instantiation_plan"])
        plan["checkpoint_load_attempt"] = {
            "path": str(checkpoint),
            "selector_path": "$.state_dict",
            "key_transforms": [],
            "load_target": {"step_id": "base", "path": "$"},
            "strict": True,
            "source_anchor_snapshot": plan["project"]["step_sources"][0]["construction_anchor_source"],
        }
        report = instantiate(plan, analyze_structure=True)
        if report["status"] == "completed":
            self.assertEqual(report["checkpoint"]["status"], "loaded")
            self.assertTrue(report["checkpoint"]["weights_only"])
            self.assertTrue(report["structure_analysis"]["source_model_state"]["checkpoint_parameters_loaded"])
            self.assertTrue(report["structure_analysis"]["source_model_state"]["checkpoint_fully_matched"])
        else:
            self.assertEqual(report["checkpoint"]["status"], "failed")
            self.assertEqual(report["error"]["code"], "WEIGHTS_ONLY_UNSUPPORTED")

    def test_worker_cannot_claim_loaded_weights_without_recipe(self) -> None:
        plan = self.validation["validated_instantiation_plan"]
        invented = {
            "report_type": "ModelInstantiationReport",
            "schema_version": "0.1",
            "proposal_id": "P1",
            "status": "completed",
            "steps": [],
            "checkpoint": {"status": "loaded", "weights_only": True},
            "model": {"target_is_torch_module": True},
            "error": None,
        }
        self.assertFalse(_valid_worker_report(invented, plan))

    def test_worker_output_is_bounded_while_running(self) -> None:
        command = [sys.executable, "-B", "-c", "import sys; sys.stdout.buffer.write(b'x' * 5000000)"]
        _, _, overflow = _run_bounded(command, b"", cwd=str(self.project), env=os.environ.copy())
        self.assertTrue(overflow)


if __name__ == "__main__":
    unittest.main()
