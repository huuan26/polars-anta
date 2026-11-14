"""polars-anta: Alpha & Numerical Technical Analysis library powered natively by Polars."""

from __future__ import annotations

from . import (
    alpha,
    backtest,
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
from .alpha import *  # noqa: F403
from .backtest import *  # noqa: F403
from .evaluation import *  # noqa: F403
from .inference import *  # noqa: F403
from .momentum import *  # noqa: F403
from .resample import *  # noqa: F403
from .stats import *  # noqa: F403
from .trend import *  # noqa: F403
from .volatility import *  # noqa: F403
from .volume import *  # noqa: F403

__version__ = "0.4.0"

__all__ = [
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
