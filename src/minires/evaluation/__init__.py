"""Evaluate MiniRes estimates against held-out sliced resin mass."""

from .baseline import EvaluationConfig, PhysicalBaseline, evaluate_records

__all__ = ["EvaluationConfig", "PhysicalBaseline", "evaluate_records"]
