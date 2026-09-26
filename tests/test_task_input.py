"""TaskInput v0.3 requires model, config, and checkpoint assets."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert_model_update.tools.task_assets.preflight import preflight_task_assets
from expert_model_update.tools.task_input.validate import missing_requirements, validate_confirmed, validate_shape


def draft() -> dict:
    return {
        "schema_version": "0.3",
        "task_id": "T1",
        "task_input_version": 1,
        "request": {
            "raw_prompt": "用新数据更新模型",
            "update_goal": {"summary": "适应新数据", "reason": "业务变化", "status": "confirmed"},
        },
        "assets": {
            "base_model": {"model_file": None, "config": None, "checkpoint": None},
            "new_data": {"location": "new-data.json", "allowed_uses": ["training"]},
        },
        "requirements": {
            "capabilities_to_preserve": {"mode": "none", "items": []},
            "success_expectation": "新数据表现改善",
        },
        "intake": {
            "required_questions": [],
            "optional_questions": [],
            "human_confirmation": {"status": "confirmed", "confirmed_task_input_version": 1, "notes": None},
        },
    }


class TaskInputTests(unittest.TestCase):
    def test_three_model_assets_are_independent_required_gaps(self) -> None:
        gaps = missing_requirements(draft())
        self.assertEqual(
            {gap.target_fields for gap in gaps},
            {
                ("assets.base_model.model_file",),
                ("assets.base_model.config",),
                ("assets.base_model.checkpoint",),
            },
        )

    def test_confirmed_input_and_preflight_pass_with_three_files(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for name in ("model.py", "config.json", "weights.pt", "new-data.json"):
                (root / name).write_bytes(b"x")
            value = draft()
            value["assets"]["base_model"] = {
                "model_file": "model.py",
                "config": "config.json",
                "checkpoint": "weights.pt",
            }
            self.assertEqual(validate_shape(value), [])
            self.assertEqual(validate_confirmed(value), [])
            report = preflight_task_assets(value, base_directory=root)
            self.assertEqual(report["overall_status"], "passed", report)
            fields = {item["field"] for item in report["checks"]}
            self.assertEqual(
                fields,
                {
                    "assets.base_model.model_file",
                    "assets.base_model.config",
                    "assets.base_model.checkpoint",
                    "assets.new_data.location",
                },
            )


if __name__ == "__main__":
    unittest.main()
