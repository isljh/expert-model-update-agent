"""Contract tests for the STAR weight-perturbation method package."""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # The base repository environment may omit PyTorch.
    torch = None
    nn = None

if torch is not None:
    from expert_model_update.methods.derpp import compute_derpp_loss
    from expert_model_update.methods.star import (
        STARConfig,
        accumulate_star_gradients,
        compute_star_gradients,
    )


def _config(**overrides):
    values = {
        "regularization_weight": 0.2,
        "perturbation_ratio": 0.05,
        "noise_scale": 1e-6,
        "ascent_steps": 1,
        "min_correct_samples": 2,
    }
    values.update(overrides)
    return STARConfig(**values)


@unittest.skipIf(torch is None, "PyTorch is required for STAR numerical tests")
class STARTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_computes_classification_gradients_and_restores_model(self) -> None:
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
            model.bias.zero_()
        model.train()
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        targets = torch.tensor([0, 1])
        before = copy.deepcopy(model.state_dict())

        result = compute_star_gradients(
            model=model,
            replay_inputs=inputs,
            replay_targets=targets,
            config=_config(),
            generator=torch.Generator().manual_seed(3),
        )

        self.assertFalse(result.skipped)
        self.assertEqual(result.correct_sample_count, 2)
        self.assertEqual(set(result.gradients), {"weight"})
        self.assertGreater(result.raw_star_loss, 0.0)
        self.assertTrue(model.training)
        self.assertIsNone(model.weight.grad)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_accumulates_star_before_base_loss_backward(self) -> None:
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.5, -0.5], [-0.5, 1.5]]))
            model.bias.zero_()
        replay_inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        replay_targets = torch.tensor([0, 1])
        current_inputs = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        current_targets = torch.tensor([0, 1])

        model.zero_grad(set_to_none=True)
        result = accumulate_star_gradients(
            model=model,
            replay_inputs=replay_inputs,
            replay_targets=replay_targets,
            config=_config(),
            generator=torch.Generator().manual_seed(4),
        )
        star_gradient = model.weight.grad.detach().clone()
        base_loss = nn.functional.cross_entropy(model(current_inputs), current_targets)
        base_gradient = torch.autograd.grad(base_loss, model.weight, retain_graph=True)[0]
        base_loss.backward()

        self.assertFalse(result.skipped)
        torch.testing.assert_close(model.weight.grad, star_gradient + base_gradient)

    def test_composes_with_derpp_in_one_optimizer_step(self) -> None:
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
            model.bias.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        replay_inputs = torch.eye(2)
        replay_targets = torch.tensor([0, 1])
        stored_outputs = model(replay_inputs).detach().clone()
        current_inputs = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
        current_targets = torch.tensor([0, 1])
        before = model.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        star_result = accumulate_star_gradients(
            model=model,
            replay_inputs=replay_inputs,
            replay_targets=replay_targets,
            config=_config(),
            generator=torch.Generator().manual_seed(5),
        )
        breakdown = compute_derpp_loss(
            current_outputs=model(current_inputs),
            current_targets=current_targets,
            replay_outputs=model(replay_inputs),
            replay_targets=replay_targets,
            stored_replay_outputs=stored_outputs,
            primary_loss=nn.CrossEntropyLoss(),
            alpha=0.5,
            beta=0.5,
            task_type="classification",
        )
        breakdown.total.backward()
        optimizer.step()

        self.assertFalse(star_result.skipped)
        self.assertFalse(torch.equal(model.weight.detach(), before))
        self.assertEqual(breakdown.metrics()["method_variant"], "derpp")

    def test_skips_when_too_few_replay_samples_are_correct(self) -> None:
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
        result = accumulate_star_gradients(
            model=model,
            replay_inputs=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            replay_targets=torch.tensor([0, 0]),
            config=_config(min_correct_samples=2),
            generator=torch.Generator().manual_seed(1),
        )

        self.assertTrue(result.skipped)
        self.assertEqual(result.skip_reason, "insufficient_correct_replay_samples")
        self.assertEqual(result.correct_sample_count, 1)
        self.assertIsNone(model.weight.grad)

    def test_rejects_regression_outputs(self) -> None:
        model = nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "classification logits"):
            compute_star_gradients(
                model=model,
                replay_inputs=torch.ones(2, 2),
                replay_targets=torch.tensor([0, 0]),
                config=_config(),
            )

    def test_selects_logits_from_mapping_output(self) -> None:
        class MappingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.head.weight.copy_(torch.eye(2) * 2)

            def forward(self, values):
                logits = self.head(values)
                return {"logits": logits, "features": values}

        model = MappingModel()
        result = compute_star_gradients(
            model=model,
            replay_inputs=torch.eye(2),
            replay_targets=torch.tensor([0, 1]),
            config=_config(),
            output_selector="logits",
            generator=torch.Generator().manual_seed(8),
        )
        self.assertFalse(result.skipped)
        self.assertEqual(set(result.gradients), {"head.weight"})

    def test_restores_parameters_and_mixed_modes_after_failure(self) -> None:
        class FailingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))
                self.calls = 0

            def forward(self, values):
                self.calls += 1
                self.forward_count.add_(1)
                if self.calls == 2:
                    raise RuntimeError("synthetic forward failure")
                return self.head(values)

        model = FailingModel()
        with torch.no_grad():
            model.head.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
            model.head.bias.zero_()
        model.train()
        model.dropout.eval()
        before = copy.deepcopy(model.state_dict())

        with self.assertRaisesRegex(RuntimeError, "synthetic forward failure"):
            compute_star_gradients(
                model=model,
                replay_inputs=torch.eye(2),
                replay_targets=torch.tensor([0, 1]),
                config=_config(),
                generator=torch.Generator().manual_seed(2),
            )

        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_explicit_scope_rejects_bias_and_frozen_parameters(self) -> None:
        model = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            compute_star_gradients(
                model=model,
                replay_inputs=torch.eye(2),
                replay_targets=torch.tensor([0, 1]),
                config=_config(parameter_names=("bias",)),
            )
        model.weight.requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "frozen"):
            compute_star_gradients(
                model=model,
                replay_inputs=torch.eye(2),
                replay_targets=torch.tensor([0, 1]),
                config=_config(parameter_names=("weight",)),
            )

    def test_method_card_declares_enhancement_and_classification_boundary(self) -> None:
        path = ROOT / "method-packages" / "star-weight-perturbation" / "method-card.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(card["method_id"], "star_weight_perturbation")
        self.assertEqual(card["method_version"], "0.1")
        self.assertEqual(card["artifact_kind"], "mechanism")
        self.assertEqual(card["decision_contract"]["supported_task"], "single_label_classification")
        self.assertTrue(card["decision_contract"]["requires_base_rehearsal_method"])


if __name__ == "__main__":
    unittest.main()
