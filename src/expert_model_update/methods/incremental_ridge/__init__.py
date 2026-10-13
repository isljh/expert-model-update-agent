"""Closed-form incremental ridge-regression method components."""

from .core import initialize_incremental_ridge, inspect_state, update_incremental_ridge

__all__ = ["initialize_incremental_ridge", "inspect_state", "update_incremental_ridge"]
