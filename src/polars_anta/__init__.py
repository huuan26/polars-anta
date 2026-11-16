"""polars-anta: Cross-sectional alpha engineering, diagnostics, and cost-aware allocation in Polars."""

from __future__ import annotations

from . import (
    allocation,
    alpha,
    backtest,
    conditioning,
    covariance,
    diagnostics,
    evaluation,
    inference,
    momentum,
    namespace,  # noqa: F401 - triggers Polars namespace registration
    resample,
    stats,
    trend,
    volatility,
    volume,
)
from .allocation import *  # noqa: F403
from .alpha import *  # noqa: F403
from .backtest import *  # noqa: F403
from .conditioning import *  # noqa: F403
from .covariance import *  # noqa: F403
from .diagnostics import *  # noqa: F403
from .evaluation import *  # noqa: F403
from .inference import *  # noqa: F403
from .momentum import *  # noqa: F403
from .resample import *  # noqa: F403
from .stats import *  # noqa: F403
from .trend import *  # noqa: F403
from .volatility import *  # noqa: F403
from .volume import *  # noqa: F403

__version__ = "0.5.0"

__all__ = [
    *conditioning.__all__,
    *diagnostics.__all__,
    *covariance.__all__,
    *allocation.__all__,
    *trend.__all__,
    *momentum.__all__,
    *volatility.__all__,
    *volume.__all__,
    *stats.__all__,
    *alpha.__all__,
    *evaluation.__all__,
    *backtest.__all__,
    *resample.__all__,
    *inference.__all__,
]
