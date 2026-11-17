"""Polars namespace registration for technical analysis expressions and frames."""

from __future__ import annotations

from ._binders import FrameType
from ._expr_ns import TechnicalAnalysisExpr
from ._frame_ns import TechnicalAnalysisFrame

__all__ = [
    "FrameType",
    "TechnicalAnalysisExpr",
    "TechnicalAnalysisFrame",
]
