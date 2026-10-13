"""Reusable DER++ replay components for task-specific PyTorch executors.

This module deliberately does not load model code, checkpoints, or datasets.  A
task-specific execution layer remains responsible for those operations and for
selecting representative replay samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F


TensorTree = Union[Tensor, Mapping[str, "TensorTree"], Tuple["TensorTree", ...]]
TaskType = Literal["classification", "regression"]
OutputSelector = Union[str, int, None]


def _clone_tree_to_cpu(value: TensorTree) -> TensorTree:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        if not value:
            raise ValueError("input mapping must not be empty")
        return {key: _clone_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        if not value:
            raise ValueError("input tuple must not be empty")
        return tuple(_clone_tree_to_cpu(item) for item in value)
    raise TypeError("inputs must be a tensor, a mapping of tensor inputs, or a tuple of tensor inputs")


def _tree_to(value: TensorTree, device: torch.device | str) -> TensorTree:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _tree_to(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_to(item, device) for item in value)
    raise TypeError("unsupported replay input structure")


def _tree_batch_size(value: TensorTree) -> int:
    sizes: list[int] = []

    def collect(item: TensorTree) -> None:
        if isinstance(item, Tensor):
            if item.ndim == 0:
                raise ValueError("every input tensor must include a leading sample dimension")
            sizes.append(int(item.shape[0]))
            return
        if isinstance(item, Mapping):
            if not item:
                raise ValueError("input mapping must not be empty")
            for child in item.values():
                collect(child)
            return
        if isinstance(item, tuple):
            if not item:
                raise ValueError("input tuple must not be empty")
            for child in item:
                collect(child)
            return
        raise TypeError("unsupported replay input structure")

    collect(value)
    if not sizes or any(size != sizes[0] for size in sizes[1:]):
        raise ValueError("all input tensors must have the same leading sample dimension")
    return sizes[0]


def _tree_structure(value: TensorTree) -> Any:
    if isinstance(value, Tensor):
        return ("tensor", tuple(value.shape[1:]), value.dtype)
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _tree_structure(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return ("tuple", tuple(_tree_structure(item) for item in value))
    raise TypeError("unsupported replay input structure")


def _tree_cat(left: TensorTree, right: TensorTree) -> TensorTree:
    if _tree_structure(left) != _tree_structure(right):
        raise ValueError("new replay inputs do not match the stored input structure")
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return torch.cat((left, right), dim=0)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return {key: _tree_cat(left[key], right[key]) for key in left}
    if isinstance(left, tuple) and isinstance(right, tuple):
        return tuple(_tree_cat(a, b) for a, b in zip(left, right))
    raise TypeError("unsupported replay input structure")


def _tree_index_select(value: TensorTree, indices: Tensor) -> TensorTree:
    if isinstance(value, Tensor):
        return value.index_select(0, indices)
    if isinstance(value, Mapping):
        return {key: _tree_index_select(item, indices) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_index_select(item, indices) for item in value)
    raise TypeError("unsupported replay input structure")


def call_model(model: torch.nn.Module, inputs: TensorTree) -> Any:
    """Call a model using one tensor, keyword tensors, or positional tensors."""

    if isinstance(inputs, Tensor):
        return model(inputs)
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, tuple):
        return model(*inputs)
    raise TypeError("unsupported model input structure")


def select_output(model_output: Any, selector: OutputSelector = None) -> Tensor:
    """Select the tensor used by DER++ without guessing ambiguous outputs."""

    selected: Any
    if selector is None:
        selected = model_output
    elif isinstance(selector, str):
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
    else:
        raise TypeError("output selector must be a string, an integer, or None")

    if not isinstance(selected, Tensor):
        if selector is None:
            raise TypeError("model output is not a tensor; provide an explicit output selector")
        raise TypeError("the selected model output is not a tensor")
    if selected.ndim == 0:
        raise ValueError("the selected model output must include a leading sample dimension")
    return selected


@dataclass(frozen=True)
class ReplayBatch:
    """A detached replay batch and the historical outputs saved with it."""

    inputs: TensorTree
    targets: Tensor
    stored_outputs: Tensor
    sample_ids: tuple[str, ...]
    source_versions: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def to(self, device: torch.device | str) -> "ReplayBatch":
        return ReplayBatch(
            inputs=_tree_to(self.inputs, device),
            targets=self.targets.to(device),
            stored_outputs=self.stored_outputs.to(device),
            sample_ids=self.sample_ids,
            source_versions=self.source_versions,
        )


class VersionedReplayBuffer:
    """Append-only replay storage for version snapshots.

    Capacity overflow raises an error instead of silently replacing historical
    samples.  Replay selection and retention policy therefore remain explicit
    decisions of the update plan.
    """

    FORMAT_VERSION = "0.1"

    def __init__(self, capacity: int | None = None) -> None:
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive or None")
        self.capacity = capacity
        self._inputs: TensorTree | None = None
        self._targets: Tensor | None = None
        self._stored_outputs: Tensor | None = None
        self._sample_ids: list[str] = []
        self._source_versions: list[str] = []
        self._next_auto_id = 0

    def __len__(self) -> int:
        return len(self._sample_ids)

    def add(
        self,
        *,
        inputs: TensorTree,
        targets: Tensor,
        stored_outputs: Tensor,
        source_version: str,
        sample_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Append one model-version snapshot without recomputing older outputs."""

        if not isinstance(source_version, str) or not source_version.strip():
            raise ValueError("source_version must be a non-empty string")
        batch_size = _tree_batch_size(inputs)
        if targets.ndim == 0 or stored_outputs.ndim == 0:
            raise ValueError("targets and stored_outputs must include a leading sample dimension")
        if int(targets.shape[0]) != batch_size or int(stored_outputs.shape[0]) != batch_size:
            raise ValueError("inputs, targets, and stored_outputs must contain the same number of samples")
        if batch_size == 0:
            raise ValueError("cannot add an empty replay snapshot")
        if self.capacity is not None and len(self) + batch_size > self.capacity:
            raise ValueError("replay capacity would be exceeded; choose an explicit retention policy first")

        if sample_ids is None:
            ids = tuple(
                f"{source_version}:{self._next_auto_id + offset}"
                for offset in range(batch_size)
            )
        else:
            ids = tuple(sample_ids)
            if len(ids) != batch_size:
                raise ValueError("sample_ids length must match the replay snapshot size")
            if any(not isinstance(item, str) or not item for item in ids):
                raise ValueError("sample_ids must contain non-empty strings")
        if len(set(ids)) != len(ids) or set(ids).intersection(self._sample_ids):
            raise ValueError("sample_ids must be unique across the replay buffer")

        frozen_inputs = _clone_tree_to_cpu(inputs)
        frozen_targets = targets.detach().cpu().clone()
        frozen_outputs = stored_outputs.detach().cpu().clone()

        if self._inputs is None:
            self._inputs = frozen_inputs
            self._targets = frozen_targets
            self._stored_outputs = frozen_outputs
        else:
            assert self._targets is not None and self._stored_outputs is not None
            if self._targets.shape[1:] != frozen_targets.shape[1:] or self._targets.dtype != frozen_targets.dtype:
                raise ValueError("new replay targets do not match the stored target shape and dtype")
            if (
                self._stored_outputs.shape[1:] != frozen_outputs.shape[1:]
                or self._stored_outputs.dtype != frozen_outputs.dtype
            ):
                raise ValueError("new stored outputs do not match the historical output shape and dtype")
            combined_inputs = _tree_cat(self._inputs, frozen_inputs)
            combined_targets = torch.cat((self._targets, frozen_targets), dim=0)
            combined_outputs = torch.cat((self._stored_outputs, frozen_outputs), dim=0)
            self._inputs = combined_inputs
            self._targets = combined_targets
            self._stored_outputs = combined_outputs

        self._sample_ids.extend(ids)
        self._source_versions.extend(source_version for _ in range(batch_size))
        self._next_auto_id += batch_size

    def add_from_model(
        self,
        *,
        model: torch.nn.Module,
        inputs: TensorTree,
        targets: Tensor,
        source_version: str,
        sample_ids: list[str] | tuple[str, ...] | None = None,
        output_selector: OutputSelector = None,
    ) -> None:
        """Capture one version's outputs in evaluation mode and append them."""

        stored_outputs = capture_model_outputs(model, inputs, output_selector=output_selector)
        self.add(
            inputs=inputs,
            targets=targets,
            stored_outputs=stored_outputs,
            source_version=source_version,
            sample_ids=sample_ids,
        )

    def all(self, device: torch.device | str | None = None) -> ReplayBatch:
        if self._inputs is None or self._targets is None or self._stored_outputs is None:
            raise ValueError("replay buffer is empty")
        batch = ReplayBatch(
            inputs=_clone_tree_to_cpu(self._inputs),
            targets=self._targets.clone(),
            stored_outputs=self._stored_outputs.clone(),
            sample_ids=tuple(self._sample_ids),
            source_versions=tuple(self._source_versions),
        )
        return batch if device is None else batch.to(device)

    def sample(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed the number of stored replay samples")
        all_samples = self.all()
        indices = torch.randperm(len(self), generator=generator)[:batch_size]
        selected = ReplayBatch(
            inputs=_tree_index_select(all_samples.inputs, indices),
            targets=all_samples.targets.index_select(0, indices),
            stored_outputs=all_samples.stored_outputs.index_select(0, indices),
            sample_ids=tuple(all_samples.sample_ids[int(index)] for index in indices),
            source_versions=tuple(all_samples.source_versions[int(index)] for index in indices),
        )
        return selected if device is None else selected.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Return a detached state suitable for an execution layer to persist."""

        state: dict[str, Any] = {
            "format_version": self.FORMAT_VERSION,
            "capacity": self.capacity,
            "next_auto_id": self._next_auto_id,
            "sample_ids": tuple(self._sample_ids),
            "source_versions": tuple(self._source_versions),
        }
        if len(self):
            batch = self.all()
            state.update(
                inputs=batch.inputs,
                targets=batch.targets,
                stored_outputs=batch.stored_outputs,
            )
        else:
            state.update(inputs=None, targets=None, stored_outputs=None)
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "VersionedReplayBuffer":
        if state.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("unsupported replay-buffer format_version")
        buffer = cls(capacity=state.get("capacity"))
        sample_ids = tuple(state.get("sample_ids", ()))
        source_versions = tuple(state.get("source_versions", ()))
        if len(sample_ids) != len(source_versions):
            raise ValueError("sample_ids and source_versions lengths do not match")
        inputs = state.get("inputs")
        targets = state.get("targets")
        stored_outputs = state.get("stored_outputs")
        if sample_ids:
            if inputs is None or not isinstance(targets, Tensor) or not isinstance(stored_outputs, Tensor):
                raise ValueError("non-empty replay state is missing inputs, targets, or stored_outputs")
            start = 0
            while start < len(source_versions):
                version = source_versions[start]
                end = start + 1
                while end < len(source_versions) and source_versions[end] == version:
                    end += 1
                indices = list(range(start, end))
                index_tensor = torch.tensor(indices, dtype=torch.long)
                buffer.add(
                    inputs=_tree_index_select(inputs, index_tensor),
                    targets=targets.index_select(0, index_tensor),
                    stored_outputs=stored_outputs.index_select(0, index_tensor),
                    source_version=version,
                    sample_ids=[sample_ids[index] for index in indices],
                )
                start = end
        buffer._next_auto_id = max(int(state.get("next_auto_id", len(buffer))), len(buffer))
        return buffer


def capture_model_outputs(
    model: torch.nn.Module,
    inputs: TensorTree,
    *,
    output_selector: OutputSelector = None,
) -> Tensor:
    """Capture fixed historical outputs without updating model state."""

    previous_training_modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            output = select_output(call_model(model, inputs), output_selector)
            if int(output.shape[0]) != _tree_batch_size(inputs):
                raise ValueError("selected model output does not preserve the input sample dimension")
            return output.detach().cpu().clone()
    finally:
        for module, training in previous_training_modes.items():
            module.training = training


@dataclass(frozen=True)
class DERPPLossBreakdown:
    total: Tensor
    new_task: Tensor
    replay_consistency: Tensor
    replay_supervised: Tensor
    alpha: float
    beta: float
    task_type: TaskType

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "method_variant": "derpp" if self.task_type == "classification" else "derpp_regression_extension",
            "total_loss": float(self.total.detach().cpu()),
            "new_task_loss": float(self.new_task.detach().cpu()),
            "replay_consistency_loss": float(self.replay_consistency.detach().cpu()),
            "replay_supervised_loss": float(self.replay_supervised.detach().cpu()),
            "alpha": self.alpha,
            "beta": self.beta,
        }


def compute_derpp_loss(
    *,
    current_outputs: Tensor,
    current_targets: Tensor,
    replay_outputs: Tensor,
    replay_targets: Tensor,
    stored_replay_outputs: Tensor,
    primary_loss: Callable[[Tensor, Tensor], Tensor],
    alpha: float,
    beta: float,
    task_type: TaskType,
    consistency_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> DERPPLossBreakdown:
    """Compute DER++ loss; regression is an explicit project extension."""

    if task_type not in ("classification", "regression"):
        raise ValueError("task_type must be 'classification' or 'regression'")
    if alpha < 0 or beta < 0 or (alpha == 0 and beta == 0):
        raise ValueError("alpha and beta must be non-negative, with at least one positive value")
    if replay_outputs.shape != stored_replay_outputs.shape:
        raise ValueError("current and stored replay outputs must have exactly the same shape")
    if int(current_outputs.shape[0]) != int(current_targets.shape[0]):
        raise ValueError("current outputs and targets must have the same sample count")
    if int(replay_outputs.shape[0]) != int(replay_targets.shape[0]):
        raise ValueError("replay outputs and targets must have the same sample count")

    new_task = primary_loss(current_outputs, current_targets)
    replay_supervised = primary_loss(replay_outputs, replay_targets)
    replay_consistency = (consistency_loss or F.mse_loss)(replay_outputs, stored_replay_outputs)
    for name, value in (
        ("new task loss", new_task),
        ("replay supervised loss", replay_supervised),
        ("replay consistency loss", replay_consistency),
    ):
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise ValueError(f"{name} must be a scalar tensor")

    total = new_task + alpha * replay_consistency + beta * replay_supervised
    return DERPPLossBreakdown(
        total=total,
        new_task=new_task,
        replay_consistency=replay_consistency,
        replay_supervised=replay_supervised,
        alpha=float(alpha),
        beta=float(beta),
        task_type=task_type,
    )


def derpp_train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    current_inputs: TensorTree,
    current_targets: Tensor,
    replay_batch: ReplayBatch,
    primary_loss: Callable[[Tensor, Tensor], Tensor],
    alpha: float,
    beta: float,
    task_type: TaskType,
    output_selector: OutputSelector = None,
    consistency_loss: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> dict[str, float | int | str]:
    """Execute one DER++ optimizer step using caller-configured model modes."""

    if int(current_targets.shape[0]) != _tree_batch_size(current_inputs):
        raise ValueError("current inputs and targets must contain the same number of samples")
    if len(replay_batch) != _tree_batch_size(replay_batch.inputs):
        raise ValueError("replay inputs and targets must contain the same number of samples")

    optimizer.zero_grad(set_to_none=True)
    current_outputs = select_output(call_model(model, current_inputs), output_selector)
    replay_outputs = select_output(call_model(model, replay_batch.inputs), output_selector)
    breakdown = compute_derpp_loss(
        current_outputs=current_outputs,
        current_targets=current_targets,
        replay_outputs=replay_outputs,
        replay_targets=replay_batch.targets,
        stored_replay_outputs=replay_batch.stored_outputs,
        primary_loss=primary_loss,
        alpha=alpha,
        beta=beta,
        task_type=task_type,
        consistency_loss=consistency_loss,
    )
    breakdown.total.backward()
    optimizer.step()

    metrics = breakdown.metrics()
    metrics.update(
        current_batch_size=int(current_targets.shape[0]),
        replay_batch_size=len(replay_batch),
    )
    return metrics
