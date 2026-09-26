"""Structure analysis checks use only in-memory, controlled PyTorch fixtures."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert_model_update.tools.model_creation.structure import analyze_model_structure
from expert_model_update.tools.model_creation.record import render_structure_tree


class ModelStructureTests(unittest.TestCase):
    def test_readable_tree_expands_every_basicblock_child(self) -> None:
        import torch

        class BasicBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.bn1 = torch.nn.BatchNorm2d(3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.bn2 = torch.nn.BatchNorm2d(3)
                self.shortcut = torch.nn.Sequential()

            def forward(self, value):
                raise AssertionError("forward must not run")

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer1 = torch.nn.Sequential(BasicBlock())

            def forward(self, value):
                raise AssertionError("forward must not run")

        checkpoint = {"status": "loaded", "missing_keys": [], "unexpected_keys": []}
        structure = analyze_model_structure(Model(), load_spec_id="a" * 64, checkpoint=checkpoint)
        report = {
            "status": "completed", "load_spec_id": "a" * 64, "checkpoint": checkpoint,
            "task_binding": {"task_id": "T1", "task_input_version": 1},
            "delivery": {"model_file": "model.py", "config": "config.json", "checkpoint": "weights.pt"},
            "model": {"class_name": "Model"},
            "structure_analysis": structure,
        }
        tree = render_structure_tree(report)
        for path in ("layer1.0", "layer1.0.conv1", "layer1.0.bn1", "layer1.0.conv2", "layer1.0.bn2", "layer1.0.shortcut"):
            self.assertIn(f'"{path}"', tree)
        self.assertLess(tree.index('"layer1.0"'), tree.index('"layer1.0.conv1"'))
        self.assertIn('"layer1.0.conv1"  Conv2d', tree)

    def test_direct_parameter_linear_family_matches_previous_analyzer(self) -> None:
        import torch

        class DirectLinear(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def forward(self, value):
                raise AssertionError("forward must not run")

        result = analyze_model_structure(
            DirectLinear(),
            load_spec_id="a" * 64,
            checkpoint={"status": "loaded", "missing_keys": [], "unexpected_keys": []},
        )
        self.assertEqual(result["general"]["model_family"], "Linear")
        self.assertTrue(result["general"]["model_family_evidence"]["direct_affine_parameter_pattern"])

    def test_shared_parameters_and_buffers_are_not_double_counted(self) -> None:
        import torch

        class SharedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layer = torch.nn.Linear(2, 2)
                self.first = layer
                self.second = layer
                self.register_buffer("scale", torch.ones(2))

            def forward(self, value):
                raise AssertionError("structure analysis must not run forward")

        model = SharedModel()
        result = analyze_model_structure(
            model,
            load_spec_id="a" * 64,
            checkpoint={"status": "loaded", "missing_keys": [], "unexpected_keys": []},
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["parameter_summary"]["total_parameter_count"], 6)
        self.assertEqual(len(result["sharing_and_quantization"]["shared_module_groups"]), 1)
        self.assertEqual(len(result["sharing_and_quantization"]["shared_parameter_groups"]), 2)
        self.assertEqual(result["buffers"][0]["buffer_path"], "scale")
        self.assertEqual(result["forward_trace"]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()
