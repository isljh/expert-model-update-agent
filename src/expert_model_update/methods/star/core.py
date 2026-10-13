"""STAR weight-perturbation gradients for classification replay methods.

The implementation is intentionally independent from any particular replay
buffer or optimizer.  A task-specific executor owns the base continual-
learning loss and calls :func:`accumulate_star_gradients` after clearing
gradients and before backpropagating that base loss.

STAR is defined here only for single-label classification logits.  Regression
extensions would need a different stability objective and sample-selection
rule and are therefore outside this package version.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F


OutputSelector = Union[str, int, None]
_NORM_EPSILON = 1e-20


@dataclass(frozen=True)
class STARConfig:
    """Hyperparameters that must be fixed by an approved update plan."""

    regularization_weight: float
    perturbation_ratio: float
    noise_scale: float
    ascent_steps: int
    min_correct_samples: int
    parameter_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        for name, value in (
            ("regularization_weight", self.regularization_weight),
            ("perturbation_ratio", self.perturbation_ratio),
            ("noise_scale", self.noise_scale),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
            if float(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.ascent_steps, int) or self.ascent_steps <= 0:
            raise ValueError("ascent_steps must be a positive integer")
        if not isinstance(self.min_correct_samples, int) or self.min_correct_samples <= 0:
            raise ValueError("min_correct_samples must be a positive integer")
        if self.parameter_names is not None:
            if not isinstance(self.parameter_names, tuple) or not self.parameter_names:
                raise ValueError("parameter_names must be a non-empty tuple or None")
            if any(not isinstance(name, str) or not name for name in self.parameter_names):
                raise ValueError("parameter_names must contain non-empty strings")
            if len(set(self.parameter_names)) != len(self.parameter_names):
                raise ValueError("parameter_names must not contain duplicates")


@dataclass(frozen=True)
class STARGradientResult:
    """Detached STAR gradients and audit-friendly step diagnostics."""

    gradients: Dict[str, Tensor] = field(repr=False)
    skipped: bool
    skip_reason: Optional[str]
    total_sample_count: int
    correct_sample_count: int
    parameter_count: int
    raw_star_loss: float
    weighted_star_loss: float
    perturbation_l2: Dict[str, float]

    def metrics(self) -> Dict[str, Any]:
        squared_total = sum(value * value for value in self.perturbation_l2.values())
        return {
            "method_id": "star_weight_perturbation",
            "method_version": "0.1",
            "task_type": "classification",
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "total_sample_count": self.total_sample_count,
            "correct_sample_count": self.correct_sample_count,
            "parameter_count": self.parameter_count,
            "raw_star_loss": self.raw_star_loss,
            "weighted_star_loss": self.weighted_star_loss,
            "perturbation_l2_total": math.sqrt(squared_total),
            "perturbation_l2_by_parameter": dict(self.perturbation_l2),
        }


def _call_model(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, Tensor):
        return model(inputs)
    if isinstance(inputs, Mapping):
        if not inputs:
            raise ValueError("replay input mapping must not be empty")
        return model(**inputs)
    if isinstance(inputs, tuple):
        if not inputs:
            raise ValueError("replay input tuple must not be empty")
        return model(*inputs)
    raise TypeError("replay inputs must be a tensor, a mapping, or a tuple")


def _input_batch_size(inputs: Any) -> int:
    sizes = []

    def collect(value: Any) -> None:
        if isinstance(value, Tensor):
            if value.ndim == 0:
                raise ValueError("every replay input tensor must include a sample dimension")
            sizes.append(int(value.shape[0]))
            return
        if isinstance(value, Mapping):
            if not value:
                raise ValueError("replay input mapping must not be empty")
            for child in value.values():
                collect(child)
            return
        if isinstance(value, tuple):
            if not value:
                raise ValueError("replay input tuple must not be empty")
            for child in value:
                collect(child)
            return
        raise TypeError("replay inputs must contain only tensors")

    collect(inputs)
    if not sizes or any(size != sizes[0] for size in sizes[1:]):
        raise ValueError("all replay input tensors must have the same sample count")
    return sizes[0]


def _select_logits(model_output: Any, selector: OutputSelector) -> Tensor:
    selected = model_output
    if isinstance(selector, str):
        if isinstance(model_output, Mapping):
            if selector not in model_output:
                raise KeyError(f"model output does not contain key {selector!r}")
            selected = model_output[selector]
        elif hasattr(model_output, selector):
            selected = getattr(model_output, selector)
        else:
            raise KeyError(f"model output does not expose field {selector!r}")
    elif isinstance(selector, int):
        if not isinstance(model_output, (tuple, list)):
            raise TypeError("an integer output selector requires a tuple or list model output")
        selected = model_output[selector]
    elif selector is not None:
        raise TypeError("output_selector must be a string, an integer, or None")

    if not isinstance(selected, Tensor):
        if selector is None:
            raise TypeError("model output is not a tensor; provide an explicit output_selector")
        raise TypeError("the selected model output is not a tensor")
    if selected.ndim != 2 or int(selected.shape[1]) < 2:
        raise ValueError("STAR v0.1 requires classification logits shaped [batch, classes] with classes >= 2")
    return selected


def _validate_targets(targets: Tensor, batch_size: int, class_count: int) -> None:
    if not isinstance(targets, Tensor):
        raise TypeError("replay_targets must be a tensor")
    if targets.ndim != 1 or int(targets.shape[0]) != batch_size:
        raise ValueError("STAR v0.1 requires one class-index target per replay sample")
    if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
        raise ValueError("replay_targets must contain integer class indices")
    if batch_size and (int(targets.min()) < 0 or int(targets.max()) >= class_count):
        raise ValueError("replay_targets contain a class index outside the logits range")


def _select_parameters(
    model: torch.nn.Module,
    names: Optional[Sequence[str]],
) -> Tuple[Tuple[str, Tensor], ...]:
    available = dict(model.named_parameters())
    if names is None:
        selected_names = [
            name
            for name, parameter in available.items()
            if parameter.requires_grad and parameter.ndim > 1
        ]
    else:
        missing = [name for name in names if name not in available]
        if missing:
            raise ValueError(f"parameter_names contain unknown parameters: {missing}")
        selected_names = list(names)

    selected = []
    for name in selected_names:
        parameter = available[name]
        if not parameter.requires_grad:
            raise ValueError(f"STAR parameter {name!r} is frozen")
        if parameter.ndim <= 1:
            raise ValueError(
                f"STAR v0.1 does not perturb one-dimensional parameter {name!r}; "
                "biases and normalization affine vectors are excluded"
            )
        if not parameter.is_floating_point():
            raise ValueError(f"STAR parameter {name!r} must use a floating dtype")
        selected.append((name, parameter))
    if not selected:
        raise ValueError("STAR found no eligible trainable parameters with at least two dimensions")
    return tuple(selected)


@contextmanager
def _temporary_eval_mode(model: torch.nn.Module) -> Iterator[None]:
    modes = tuple((module, module.training) for module in model.modules())
    buffers = tuple((buffer, buffer.detach().clone()) for buffer in model.buffers())
    model.eval()
    try:
        yield
    finally:
        try:
            with torch.no_grad():
                for buffer, original in buffers:
                    if buffer.shape != original.shape:
                        raise RuntimeError("a model buffer changed shape during STAR evaluation")
                    buffer.copy_(original)
        finally:
            for module, training in modes:
                module.training = training


@contextmanager
def _temporary_perturbation(
    named_parameters: Sequence[Tuple[str, Tensor]],
    perturbations: Mapping[str, Tensor],
) -> Iterator[None]:
    applied = []
    try:
        with torch.no_grad():
            for name, parameter in named_parameters:
                original = parameter.detach().clone()
                parameter.add_(perturbations[name])
                applied.append((name, parameter, original))
        yield
    finally:
        with torch.no_grad():
            for _, parameter, original in reversed(applied):
                parameter.copy_(original)


def _masked_kl(
    perturbed_logits: Tensor,
    reference_probabilities: Tensor,
    correct_mask: Tensor,
) -> Tensor:
    if perturbed_logits.shape != reference_probabilities.shape:
        raise ValueError("classification output shape changed while calculating STAR")
    if not bool(torch.isfinite(perturbed_logits).all()):
        raise ValueError("model produced non-finite perturbed logits")
    per_sample = F.kl_div(
        F.log_softmax(perturbed_logits, dim=1),
        reference_probabilities,
        reduction="none",
    ).sum(dim=1)
    loss = per_sample[correct_mask].mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError("STAR KL loss is not finite")
    return loss


def _random_like(parameter: Tensor, generator: Optional[torch.Generator]) -> Tensor:
    return torch.randn(
        tuple(parameter.shape),
        dtype=parameter.dtype,
        device=parameter.device,
        generator=generator,
    )


def compute_star_gradients(
    *,
    model: torch.nn.Module,
    replay_inputs: Any,
    replay_targets: Tensor,
    config: STARConfig,
    output_selector: OutputSelector = None,
    generator: Optional[torch.Generator] = None,
) -> STARGradientResult:
    """Compute detached STAR gradients without changing parameters or ``.grad``.

    STAR forwards run in evaluation mode so that dropout and mutable
    normalization buffers do not masquerade as sensitivity to weight changes.
    Every module's prior training flag is restored before returning.
    """

    if not isinstance(config, STARConfig):
        raise TypeError("config must be a STARConfig")
    batch_size = _input_batch_size(replay_inputs)
    if batch_size == 0:
        raise ValueError("replay_inputs must not be empty")
    named_parameters = _select_parameters(model, config.parameter_names)
    parameters = tuple(parameter for _, parameter in named_parameters)

    with _temporary_eval_mode(model):
        with torch.no_grad():
            reference_logits = _select_logits(_call_model(model, replay_inputs), output_selector)
            if int(reference_logits.shape[0]) != batch_size:
                raise ValueError("classification logits do not preserve the replay sample dimension")
            _validate_targets(replay_targets, batch_size, int(reference_logits.shape[1]))
            if not bool(torch.isfinite(reference_logits).all()):
                raise ValueError("model produced non-finite reference logits")
            reference_probabilities = F.softmax(reference_logits, dim=1).detach()
            correct_mask = reference_logits.argmax(dim=1).eq(replay_targets)
            correct_count = int(correct_mask.sum().item())

        if correct_count < config.min_correct_samples:
            return STARGradientResult(
                gradients={},
                skipped=True,
                skip_reason="insufficient_correct_replay_samples",
                total_sample_count=batch_size,
                correct_sample_count=correct_count,
                parameter_count=len(named_parameters),
                raw_star_loss=0.0,
                weighted_star_loss=0.0,
                perturbation_l2={},
            )

        perturbations = {}
        for name, parameter in named_parameters:
            parameter_norm = parameter.detach().norm()
            perturbations[name] = (
                _random_like(parameter, generator)
                * float(config.noise_scale)
                * parameter_norm
            )

        step_ratio = float(config.perturbation_ratio) / config.ascent_steps
        for _ in range(config.ascent_steps):
            with _temporary_perturbation(named_parameters, perturbations):
                perturbed_logits = _select_logits(_call_model(model, replay_inputs), output_selector)
                ascent_loss = _masked_kl(
                    perturbed_logits,
                    reference_probabilities,
                    correct_mask,
                )
                ascent_gradients = torch.autograd.grad(
                    ascent_loss,
                    parameters,
                    allow_unused=True,
                )

            for (name, parameter), gradient in zip(named_parameters, ascent_gradients):
                if gradient is None:
                    raise ValueError(f"STAR parameter {name!r} did not contribute to classification logits")
                if not bool(torch.isfinite(gradient).all()):
                    raise ValueError(f"STAR ascent gradient for {name!r} is not finite")
                gradient_norm = gradient.detach().norm()
                parameter_norm = parameter.detach().norm()
                if float(gradient_norm) > _NORM_EPSILON and float(parameter_norm) > 0:
                    perturbations[name] = perturbations[name] + (
                        step_ratio * parameter_norm / (gradient_norm + _NORM_EPSILON)
                    ) * gradient.detach()

        with _temporary_perturbation(named_parameters, perturbations):
            final_logits = _select_logits(_call_model(model, replay_inputs), output_selector)
            raw_loss = _masked_kl(final_logits, reference_probabilities, correct_mask)
            weighted_loss = float(config.regularization_weight) * raw_loss
            final_gradients = torch.autograd.grad(
                weighted_loss,
                parameters,
                allow_unused=True,
            )

        detached_gradients = {}
        for (name, _), gradient in zip(named_parameters, final_gradients):
            if gradient is None:
                raise ValueError(f"STAR parameter {name!r} did not contribute to the final loss")
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError(f"STAR gradient for {name!r} is not finite")
            detached_gradients[name] = gradient.detach().clone()

        return STARGradientResult(
            gradients=detached_gradients,
            skipped=False,
            skip_reason=None,
            total_sample_count=batch_size,
            correct_sample_count=correct_count,
            parameter_count=len(named_parameters),
            raw_star_loss=float(raw_loss.detach().cpu()),
            weighted_star_loss=float(weighted_loss.detach().cpu()),
            perturbation_l2={
                name: float(perturbation.detach().norm().cpu())
                for name, perturbation in perturbations.items()
            },
        )


def accumulate_star_gradients(
    *,
    model: torch.nn.Module,
    replay_inputs: Any,
    replay_targets: Tensor,
    config: STARConfig,
    output_selector: OutputSelector = None,
    generator: Optional[torch.Generator] = None,
) -> STARGradientResult:
    """Add STAR gradients to existing parameter gradients without zeroing them."""

    result = compute_star_gradients(
        model=model,
        replay_inputs=replay_inputs,
        replay_targets=replay_targets,
        config=config,
        output_selector=output_selector,
        generator=generator,
    )
    if result.skipped:
        return result

    available = dict(model.named_parameters())
    with torch.no_grad():
        for name, gradient in result.gradients.items():
            parameter = available[name]
            if parameter.grad is None:
                parameter.grad = gradient.clone()
            else:
                if parameter.grad.shape != gradient.shape:
                    raise ValueError(f"existing gradient shape for {name!r} does not match STAR")
                parameter.grad.add_(gradient)
    return result
