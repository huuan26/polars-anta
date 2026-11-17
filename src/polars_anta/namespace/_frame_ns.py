"""Polars DataFrame and LazyFrame namespace registration (.anta and .ta)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic

import polars as pl

from .. import alpha, momentum, stats, trend, volatility, volume
from ._binders import FrameType, _bind_frame_expr, _bind_frame_passthrough
from ._frame_specs import _FRAME_EXPR_SPECS, _FRAME_PASSTHROUGHS


@pl.api.register_dataframe_namespace("anta")
@pl.api.register_dataframe_namespace("ta")
@pl.api.register_lazyframe_namespace("anta")
@pl.api.register_lazyframe_namespace("ta")
class TechnicalAnalysisFrame(Generic[FrameType]):
    """Namespace for technical analysis operations on Polars DataFrames and LazyFrames.

    Allows chaining indicators, multi-timeframe resampling, and data integrity validation
    directly on `df.anta...` or `df.ta...` (and `lf.anta...` or `lf.ta...`).
    """

    def __init__(self, df: FrameType) -> None:
        self._df: FrameType = df

    def add_indicators(
        self,
        *exprs: pl.Expr | Sequence[pl.Expr],
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Append one or more indicator expressions in a single vectorized pass.

        :param exprs: Expressions or lists of expressions to append.
        :param group_by: Optional column(s) to evaluate expressions over groups.
        :return: DataFrame or LazyFrame with new columns.
        """
        flat_exprs: list[pl.Expr] = []
        for item in exprs:
            if isinstance(item, pl.Expr):
                flat_exprs.append(item)
            else:
                flat_exprs.extend(item)
        if group_by is not None:
            flat_exprs = [e.over(group_by) for e in flat_exprs]
        return self._df.with_columns(flat_exprs)

    def add_all(
        self,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str | None = "volume",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Compute and append a comprehensive standard set of technical indicators in one pass.

        Computes in parallel within Polars' query engine:
        - Trend: SMA(10, 20, 50), EMA(12, 26), ADX(14), SuperTrend(10, 3.0)
        - Momentum: RSI(14), MACD(12, 26, 9), Stochastic(14, 3, 3), Williams %R(14), CCI(20), ROC(10)
        - Volatility: ATR(14), Bollinger Bands(20, 2.0), Keltner Channels(20, 10, 2.0)
        - Volume (if volume_col present): OBV, VWAP, CMF(20), MFI(14)
        - Stats & Alpha: Z-Score(20), Percentile Rank(20), TS Rank(10)
        """
        schema = (
            self._df.collect_schema() if isinstance(self._df, pl.LazyFrame) else self._df.schema
        )
        has_volume = volume_col is not None and volume_col in schema

        exprs: list[pl.Expr] = [
            trend.sma(close_col, period=10).alias("sma_10"),
            trend.sma(close_col, period=20).alias("sma_20"),
            trend.sma(close_col, period=50).alias("sma_50"),
            trend.ema(close_col, period=12).alias("ema_12"),
            trend.ema(close_col, period=26).alias("ema_26"),
            trend.adx(high=high_col, low=low_col, close=close_col, period=14).alias("adx_14"),
            trend.supertrend(
                high=high_col, low=low_col, close=close_col, period=10, multiplier=3.0
            ).alias("supertrend"),
            momentum.rsi(close_col, period=14).alias("rsi_14"),
            momentum.macd(close_col, fast_period=12, slow_period=26, signal_period=9).alias("macd"),
            momentum.stochastic(
                high=high_col, low=low_col, close=close_col, k_period=14, d_period=3, smooth_k=3
            ).alias("stochastic"),
            momentum.williams_r(high=high_col, low=low_col, close=close_col, period=14).alias(
                "williams_r_14"
            ),
            momentum.cci(high=high_col, low=low_col, close=close_col, period=20).alias("cci_20"),
            momentum.roc(close_col, period=10).alias("roc_10"),
            volatility.atr(high=high_col, low=low_col, close=close_col, period=14).alias("atr_14"),
            volatility.bollinger_bands(close_col, period=20, num_std=2.0).alias("bollinger_bands"),
            volatility.keltner_channels(
                high=high_col,
                low=low_col,
                close=close_col,
                ema_period=20,
                atr_period=10,
                multiplier=2.0,
            ).alias("keltner_channels"),
            stats.zscore(close_col, period=20).alias("zscore_20"),
            stats.percentile_rank(close_col, period=20).alias("percentile_rank_20"),
            alpha.ts_rank(close_col, period=10).alias("ts_rank_10"),
        ]

        if has_volume and volume_col is not None:
            exprs.extend(
                [
                    volume.obv(close=close_col, volume=volume_col).alias("obv"),
                    volume.vwap(
                        high=high_col, low=low_col, close=close_col, volume=volume_col
                    ).alias("vwap"),
                    volume.cmf(
                        high=high_col, low=low_col, close=close_col, volume=volume_col, period=20
                    ).alias("cmf_20"),
                    volume.mfi(
                        high=high_col, low=low_col, close=close_col, volume=volume_col, period=14
                    ).alias("mfi_14"),
                ]
            )

        return self.add_indicators(exprs, group_by=group_by)

    def _apply_expr(
        self,
        expr: pl.Expr,
        alias: str,
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        if group_by is not None:
            expr = expr.over(group_by)
        df_res = self._df.with_columns(expr.alias(alias))
        if unnest:
            schema = df_res.collect_schema() if isinstance(df_res, pl.LazyFrame) else df_res.schema
            if isinstance(schema[alias], pl.Struct):
                if prefix is not None:
                    return df_res.with_columns(
                        pl.col(alias).struct.field("*").name.prefix(prefix)
                    ).drop(alias)
                return df_res.unnest(alias)
        return df_res


for _p_fn, _p_alias in _FRAME_PASSTHROUGHS:
    _m_name, _m_fn = _bind_frame_passthrough(_p_fn, _p_alias)
    setattr(TechnicalAnalysisFrame, _m_name, _m_fn)


for _spec in _FRAME_EXPR_SPECS:
    _m_name, _m_fn = _bind_frame_expr(_spec)
    setattr(TechnicalAnalysisFrame, _m_name, _m_fn)
