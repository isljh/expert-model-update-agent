"""Contract tests for the DER++ replay method package."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The base repository environment does not install PyTorch.
    torch = None
    nn = None

if torch is not None:
    from expert_model_update.methods.derpp import (
        VersionedReplayBuffer,
        capture_model_outputs,
        compute_derpp_loss,
        derpp_train_step,
        select_output,
    )


@unittest.skipIf(torch is None, "PyTorch is required for DER++ numerical tests")
class DERPPTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_classification_loss_matches_derpp_formula(self) -> None:
        current_outputs = torch.tensor([[1.2, -0.4], [-0.3, 0.7]], requires_grad=True)
        current_targets = torch.tensor([0, 1])
        replay_outputs = torch.tensor([[0.8, 0.2], [0.1, 0.9]], requires_grad=True)
        replay_targets = torch.tensor([0, 1])
        stored_outputs = torch.tensor([[1.0, 0.0], [0.3, 0.7]])
        primary_loss = nn.CrossEntropyLoss()

        result = compute_derpp_loss(
            current_outputs=current_outputs,
            current_targets=current_targets,
            replay_outputs=replay_outputs,
            replay_targets=replay_targets,
            stored_replay_outputs=stored_outputs,
            primary_loss=primary_loss,
            alpha=0.4,
            beta=0.7,
            task_type="classification",
        )

        expected = (
            primary_loss(current_outputs, current_targets)
            + 0.4 * nn.functional.mse_loss(replay_outputs, stored_outputs)
            + 0.7 * primary_loss(replay_outputs, replay_targets)
        )
        torch.testing.assert_close(result.total, expected)
        self.assertEqual(result.metrics()["method_variant"], "derpp")

    def test_regression_extension_uses_same_three_term_structure(self) -> None:
        current_outputs = torch.tensor([[1.0], [3.0]], requires_grad=True)
        current_targets = torch.tensor([[2.0], [2.0]])
        replay_outputs = torch.tensor([[2.5], [0.5]], requires_grad=True)
        replay_targets = torch.tensor([[2.0], [1.0]])
        stored_outputs = torch.tensor([[2.0], [0.0]])
        primary_loss = nn.L1Loss()

        result = compute_derpp_loss(
            current_outputs=current_outputs,
            current_targets=current_targets,
            replay_outputs=replay_outputs,
            replay_targets=replay_targets,
            stored_replay_outputs=stored_outputs,
            primary_loss=primary_loss,
            alpha=0.5,
            beta=0.25,
            task_type="regression",
        )

        expected = (
            primary_loss(current_outputs, current_targets)
            + 0.5 * nn.functional.mse_loss(replay_outputs, stored_outputs)
            + 0.25 * primary_loss(replay_outputs, replay_targets)
        )
        torch.testing.assert_close(result.total, expected)
        self.assertEqual(result.metrics()["method_variant"], "derpp_regression_extension")

    def test_capture_restores_mode_and_buffer_keeps_historical_outputs_fixed(self) -> None:
        model = nn.Sequential(nn.Linear(2, 3), nn.Dropout(p=0.8), nn.Linear(3, 2))
        model.train()
        model[1].eval()
        old_inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        old_targets = torch.tensor([0, 1])

        buffer = VersionedReplayBuffer()
        buffer.add_from_model(
            model=model,
            inputs=old_inputs,
            targets=old_targets,
            source_version="1.0",
            sample_ids=["old-1", "old-2"],
        )
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        old_snapshot = buffer.all().stored_outputs.clone()

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10.0)
        new_inputs = torch.tensor([[5.0, 6.0]])
        buffer.add_from_model(
            model=model,
            inputs=new_inputs,
            targets=torch.tensor([1]),
            source_version="2.0",
            sample_ids=["new-1"],
        )

        torch.testing.assert_close(buffer.all().stored_outputs[:2], old_snapshot)
        self.assertEqual(buffer.all().source_versions, ("1.0", "1.0", "2.0"))

    def test_replay_buffer_state_round_trip_and_sample(self) -> None:
        buffer = VersionedReplayBuffer(capacity=4)
        buffer.add(
            inputs={
                "values": torch.arange(6, dtype=torch.float32).reshape(3, 2),
                "mask": torch.ones(3, 2, dtype=torch.long),
            },
            targets=torch.tensor([0, 1, 0]),
            stored_outputs=torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.5, 0.5]]),
            source_version="1.0",
            sample_ids=["a", "b", "c"],
        )
        restored = VersionedReplayBuffer.from_state_dict(buffer.state_dict())
        sampled = restored.sample(2, generator=torch.Generator().manual_seed(1))

        self.assertEqual(len(restored), 3)
        self.assertEqual(len(sampled), 2)
        self.assertEqual(set(sampled.sample_ids).issubset({"a", "b", "c"}), True)
        self.assertEqual(tuple(sampled.inputs["values"].shape), (2, 2))

    def test_capacity_overflow_never_silently_discards_old_samples(self) -> None:
        buffer = VersionedReplayBuffer(capacity=1)
        buffer.add(
            inputs=torch.tensor([[1.0, 2.0]]),
            targets=torch.tensor([0]),
            stored_outputs=torch.tensor([[1.0, 0.0]]),
            source_version="1.0",
            sample_ids=["old"],
        )
        with self.assertRaisesRegex(ValueError, "capacity"):
            buffer.add(
                inputs=torch.tensor([[3.0, 4.0]]),
                targets=torch.tensor([1]),
                stored_outputs=torch.tensor([[0.0, 1.0]]),
                source_version="2.0",
                sample_ids=["new"],
            )
        self.assertEqual(buffer.all().sample_ids, ("old",))

    def test_invalid_append_does_not_partially_modify_buffer(self) -> None:
        buffer = VersionedReplayBuffer()
        buffer.add(
            inputs=torch.tensor([[1.0, 2.0]]),
            targets=torch.tensor([0]),
            stored_outputs=torch.tensor([[1.0, 0.0]]),
            source_version="1.0",
            sample_ids=["old"],
        )
        before = buffer.all()

        with self.assertRaisesRegex(ValueError, "input structure"):
            buffer.add(
                inputs=torch.tensor([[3.0, 4.0, 5.0]]),
                targets=torch.tensor([1]),
                stored_outputs=torch.tensor([[0.0, 1.0]]),
                source_version="2.0",
                sample_ids=["new"],
            )

        after = buffer.all()
        self.assertEqual(after.sample_ids, before.sample_ids)
        torch.testing.assert_close(after.inputs, before.inputs)
        torch.testing.assert_close(after.targets, before.targets)
        torch.testing.assert_close(after.stored_outputs, before.stored_outputs)

    def test_output_selector_handles_mapping_without_guessing(self) -> None:
        output = {"hidden": torch.ones(2, 3), "logits": torch.ones(2, 2)}
        with self.assertRaisesRegex(TypeError, "explicit output selector"):
            select_output(output)
        self.assertIs(select_output(output, "logits"), output["logits"])

    def test_train_step_updates_parameters_and_reports_all_terms(self) -> None:
        model = nn.Linear(2, 2)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        replay_inputs = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        stored_outputs = capture_model_outputs(model, replay_inputs)
        buffer = VersionedReplayBuffer()
        buffer.add(
            inputs=replay_inputs,
            targets=torch.tensor([1, 0]),
            stored_outputs=stored_outputs,
            source_version="1.0",
        )
        old_weight = model.weight.detach().clone()

        metrics = derpp_train_step(
            model=model,
            optimizer=optimizer,
            current_inputs=torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
            current_targets=torch.tensor([0, 1]),
            replay_batch=buffer.all(),
            primary_loss=nn.CrossEntropyLoss(),
            alpha=0.5,
            beta=0.5,
            task_type="classification",
        )

        self.assertFalse(torch.equal(model.weight.detach(), old_weight))
        self.assertEqual(metrics["current_batch_size"], 2)
        self.assertEqual(metrics["replay_batch_size"], 2)
        self.assertIn("replay_consistency_loss", metrics)
        torch.testing.assert_close(buffer.all().stored_outputs, stored_outputs)

    def test_output_dimension_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the same shape"):
            compute_derpp_loss(
                current_outputs=torch.ones(1, 3),
                current_targets=torch.tensor([0]),
                replay_outputs=torch.ones(1, 3),
                replay_targets=torch.tensor([0]),
                stored_replay_outputs=torch.ones(1, 2),
                primary_loss=nn.CrossEntropyLoss(),
                alpha=0.5,
                beta=0.5,
                task_type="classification",
            )


if __name__ == "__main__":
    unittest.main()
