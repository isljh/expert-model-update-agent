"""STAR classification stability components for rehearsal-based executors."""

from .core import (
    STARConfig,
    STARGradientResult,
    accumulate_star_gradients,
    compute_star_gradients,
)

__all__ = [
    "STARConfig",
    "STARGradientResult",
    "accumulate_star_gradients",
    "compute_star_gradients",
]
