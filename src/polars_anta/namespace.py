"""Polars namespace registration for technical analysis expressions and frames."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import polars as pl

from . import (
    alpha,
    backtest,
    evaluation,
    inference,
    momentum,
    resample,
    stats,
    trend,
    volatility,
    volume,
)

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)

_EMPTY = inspect.Parameter.empty
_PK = inspect.Parameter.POSITIONAL_OR_KEYWORD
_SELF_PARAM = inspect.Parameter("self", _PK)
_FRAME_CTRL_KEYS = frozenset({"alias", "unnest", "prefix", "group_by"})
_EXPR_ALPHA_DEFAULTS: dict[str, str] = {
    "open_": "open",
    "high": "high",
    "low": "low",
    "volume": "volume",
}
_FRAME_ALPHA_DEFAULTS: dict[str, str] = {**_EXPR_ALPHA_DEFAULTS, "close": "close"}


def _bind_expr(
    fn: Callable[..., pl.Expr],
    *,
    self_param: str | None = "close",
    renames: Mapping[str, str] | None = None,
    defaults: Mapping[str, Any] | None = None,
    params: Sequence[tuple[str, Any]] | None = None,
) -> Callable[..., pl.Expr]:
    """Build a TechnicalAnalysisExpr method forwarding ``self._expr`` to ``fn``."""
    rename_map = dict(renames) if renames else {}
    inv_renames = {v: k for k, v in rename_map.items()}
    if params is not None:
        mparams = [inspect.Parameter(n, _PK, default=d) for n, d in params]
    else:
        mparams = []
        for fp in inspect.signature(fn).parameters.values():
            if fp.name == self_param:
                continue
            m_name = rename_map.get(fp.name, fp.name)
            dflt = (
                defaults[m_name]
                if (fp.default is _EMPTY and defaults and m_name in defaults)
                else fp.default
            )
            mparams.append(inspect.Parameter(m_name, _PK, default=dflt))
    msig = inspect.Signature(parameters=[_SELF_PARAM, *mparams])

    def method(self: TechnicalAnalysisExpr, *args: Any, **kwargs: Any) -> pl.Expr:
        bound = msig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call_kwargs: dict[str, Any] = {}
        if self_param is not None:
            call_kwargs[self_param] = self._expr
        for m_name, val in list(bound.arguments.items())[1:]:
            call_kwargs[inv_renames.get(m_name, m_name)] = val
        return fn(**call_kwargs)

    method.__name__ = fn.__name__
    method.__qualname__ = f"TechnicalAnalysisExpr.{fn.__name__}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return method


def _bind_frame_passthrough(
    fn: Callable[..., Any],
    name: str | None = None,
) -> tuple[str, Callable[..., Any]]:
    """Build a TechnicalAnalysisFrame method forwarding ``self._df`` to ``fn``."""
    meth_name = name or fn.__name__
    fsig = inspect.signature(fn)
    msig = fsig.replace(parameters=[_SELF_PARAM, *list(fsig.parameters.values())[1:]])

    def method(self: TechnicalAnalysisFrame[Any], *args: Any, **kwargs: Any) -> Any:
        return fn(self._df, *args, **kwargs)

    method.__name__ = meth_name
    method.__qualname__ = f"TechnicalAnalysisFrame.{meth_name}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return meth_name, method


@dataclass(frozen=True)
class _FrameExprSpec:
    fn: Callable[..., pl.Expr]
    params: tuple[tuple[str, Any], ...]
    renames: Mapping[str, str] | None = None
    alias_fmt: str | Callable[[Mapping[str, Any]], str] | None = None


def _col_period_spec(
    fn: Callable[..., pl.Expr],
    period: int,
    alias_base: str | None = None,
    col_target: str = "close",
) -> _FrameExprSpec:
    base = alias_base or fn.__name__
    return _FrameExprSpec(
        fn=fn,
        params=(("period", period), ("col", "close"), ("alias", None), ("group_by", None)),
        renames={"col": col_target},
        alias_fmt=f"{base}_{{period}}",
    )


def _struct_spec(
    fn: Callable[..., pl.Expr],
    /,
    *,
    _alias: str | None = None,
    _prefix: str | None = None,
    _renames: Mapping[str, str] | None = None,
    **params: Any,
) -> _FrameExprSpec:
    return _FrameExprSpec(
        fn=fn,
        params=(
            *params.items(),
            ("alias", _alias or fn.__name__),
            ("unnest", False),
            ("prefix", _prefix),
            ("group_by", None),
        ),
        renames=_renames,
    )


def _scalar_spec(
    fn: Callable[..., pl.Expr],
    /,
    *,
    _alias_fmt: str | Callable[[Mapping[str, Any]], str] | None = None,
    _alias: str | None = None,
    _renames: Mapping[str, str] | None = None,
    _has_gb: bool = True,
    **params: Any,
) -> _FrameExprSpec:
    tail: tuple[tuple[str, Any], ...] = (
        (("alias", _alias), ("group_by", None)) if _has_gb else (("alias", _alias),)
    )
    return _FrameExprSpec(
        fn=fn,
        params=(*params.items(), *tail),
        renames=_renames,
        alias_fmt=_alias_fmt,
    )


def _bind_frame_expr(spec: _FrameExprSpec) -> tuple[str, Callable[..., Any]]:
    """Build a TechnicalAnalysisFrame indicator method backed by ``_apply_expr``."""
    fn = spec.fn
    meth_name = fn.__name__
    rename_map = dict(spec.renames) if spec.renames else {}
    msig = inspect.Signature(
        parameters=[_SELF_PARAM, *(inspect.Parameter(n, _PK, default=d) for n, d in spec.params)]
    )

    def method(self: TechnicalAnalysisFrame[Any], *args: Any, **kwargs: Any) -> Any:
        bound = msig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        raw_args = dict(list(bound.arguments.items())[1:])
        call_kwargs = {
            rename_map.get(k, k): v for k, v in raw_args.items() if k not in _FRAME_CTRL_KEYS
        }
        alias_val = raw_args["alias"]
        if spec.alias_fmt is not None:
            fallback = (
                spec.alias_fmt(raw_args)
                if callable(spec.alias_fmt)
                else spec.alias_fmt.format_map(raw_args)
            )
            alias_name = alias_val or fallback
        else:
            alias_name = alias_val
        return self._apply_expr(
            fn(**call_kwargs),
            alias=alias_name,
            unnest=raw_args.get("unnest", False),
            prefix=raw_args.get("prefix"),
            group_by=raw_args.get("group_by"),
        )

    method.__name__ = meth_name
    method.__qualname__ = f"TechnicalAnalysisFrame.{meth_name}"
    method.__doc__ = fn.__doc__
    method.__module__ = __name__
    method.__signature__ = msig  # type: ignore[attr-defined]
    return meth_name, method


@pl.api.register_expr_namespace("anta")
@pl.api.register_expr_namespace("ta")
class TechnicalAnalysisExpr:
    """Namespace for technical analysis and alpha factor operations on Polars expressions.

    Allows chaining indicators directly on `pl.col(...)` via `.anta.<indicator>()` or `.ta.<indicator>()`.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    bollinger_bands = _bind_expr(
        volatility.bollinger_bands,
        params=(("period", 20), ("num_std", 2.0)),
    )
    keltner_channels = _bind_expr(
        volatility.keltner_channels,
        params=(
            ("high", _EMPTY),
            ("low", _EMPTY),
            ("ema_period", 20),
            ("atr_period", 10),
            ("multiplier", 2.0),
        ),
    )


_EXPR_CLOSE_FUNCS: tuple[Callable[..., pl.Expr], ...] = (
    trend.sma,
    trend.ema,
    trend.wma,
    trend.hma,
    trend.dema,
    trend.tema,
    trend.zlema,
    trend.vortex,
    trend.adx,
    trend.chop,
    trend.t3,
    trend.alma,
    trend.supertrend,
    trend.kama,
    trend.mcginley_dynamic,
    trend.vidya,
    trend.center_of_gravity,
    trend.ichimoku,
    trend.pivot_points,
    trend.fibonacci_pivots,
    trend.camarilla_pivots,
    trend.woodie_pivots,
    trend.ssl_channel,
    trend.trend_intensity_index,
    trend.heikin_ashi,
    momentum.rsi,
    momentum.connors_rsi,
    momentum.macd,
    momentum.stochastic,
    momentum.williams_r,
    momentum.cci,
    momentum.roc,
    momentum.stoch_rsi,
    momentum.stc,
    momentum.tsi,
    momentum.cmo,
    momentum.ppo,
    momentum.trix,
    momentum.coppock,
    momentum.ultimate_oscillator,
    momentum.rvi,
    momentum.kst,
    momentum.dpo,
    momentum.bop,
    momentum.cfo,
    momentum.squeeze_momentum,
    volatility.true_range,
    volatility.atr,
    volatility.stddev,
    volatility.ulcer_index,
    volatility.chandelier_exit,
    volatility.natr,
    volatility.historical_volatility,
    volatility.garman_klass_volatility,
    volatility.rogers_satchell_volatility,
    volatility.yang_zhang_volatility,
    volatility.envelopes,
    volatility.keltner_width,
    volatility.bollinger_delta,
    volatility.starc_bands,
    stats.zscore,
    stats.percentile_rank,
    stats.skewness,
    stats.kurtosis,
    stats.linear_regression_slope,
    stats.linear_regression_intercept,
    stats.linear_regression_angle,
    stats.linear_regression_forecast,
    stats.linear_regression_r2,
    stats.linear_regression_stderr,
    stats.frac_diff,
    alpha.ts_rank,
    alpha.ts_argmax,
    alpha.ts_argmin,
    alpha.decay_linear,
    alpha.ts_zscore,
    alpha.returns_divergence,
    alpha.co_ratio,
    alpha.volume_weighted_momentum,
    alpha.entropy,
    alpha.hurst_exponent,
    alpha.amihud_illiquidity,
    alpha.roll_spread,
    alpha.kyle_lambda,
    alpha.vpin,
    alpha.oi_regime,
    alpha.alpha_001,
    alpha.alpha_009,
    alpha.alpha_010,
    alpha.alpha_019,
    alpha.alpha_024,
    alpha.alpha_029,
    evaluation.triple_barrier_labels,
    evaluation.trend_scanning_labels,
)

_EXPR_GROUP_BINDINGS: tuple[
    tuple[
        str | None,
        Mapping[str, str] | None,
        Mapping[str, Any] | None,
        Sequence[Callable[..., pl.Expr]],
    ],
    ...,
] = (
    ("close", None, None, _EXPR_CLOSE_FUNCS),
    (
        None,
        None,
        None,
        (
            trend.aroon,
            trend.mass_index,
            trend.psar,
            momentum.awesome_oscillator,
            momentum.fisher_transform,
            volatility.donchian_channels,
            volatility.parkinson_volatility,
            volatility.chaikin_volatility,
            volatility.donchian_width,
        ),
    ),
    (
        "close",
        None,
        _EXPR_ALPHA_DEFAULTS,
        (
            alpha.alpha_002,
            alpha.alpha_005,
            alpha.alpha_007,
            alpha.alpha_008,
            alpha.alpha_011,
            alpha.alpha_012,
            alpha.alpha_013,
            alpha.alpha_014,
            alpha.alpha_017,
            alpha.alpha_018,
            alpha.alpha_020,
            alpha.alpha_021,
            alpha.alpha_022,
            alpha.alpha_025,
            alpha.alpha_027,
            alpha.alpha_028,
            alpha.alpha_030,
        ),
    ),
    (
        "close",
        {"volume": "volume_col"},
        None,
        (
            volume.obv,
            volume.vwap,
            volume.vwap_bands,
            volume.session_vwap,
            volume.vwma,
            volume.cmf,
            volume.adl,
            volume.mfi,
            volume.klinger,
            volume.nvi,
            volume.pvi,
            volume.pvt,
            volume.force_index,
            volume.vpci,
            volume.vr,
        ),
    ),
    (None, {"volume": "volume_col"}, None, (volume.eom,)),
    (
        "x",
        {"y": "other"},
        None,
        (stats.rolling_cov, stats.rolling_corr, alpha.ts_correlation, alpha.ts_covariance),
    ),
    ("asset_returns", None, None, (stats.rolling_beta, stats.rolling_alpha)),
    (
        "high",
        None,
        None,
        (
            alpha.hl_ratio,
            alpha.relative_spread,
            alpha.twap,
            alpha.fractal_dimension,
            alpha.corwin_schultz_spread,
            alpha.alpha_023,
            alpha.alpha_041,
        ),
    ),
    ("high", None, _EXPR_ALPHA_DEFAULTS, (alpha.alpha_015, alpha.alpha_016, alpha.alpha_026)),
    (
        "expr",
        None,
        None,
        (
            alpha.cs_rank,
            alpha.cs_zscore,
            alpha.cs_winsorize,
            alpha.neutralize,
            alpha.orthogonalize,
            alpha.ts_scale,
            alpha.ts_delta,
            alpha.ts_delay,
            alpha.signed_power,
            alpha.ts_min,
            alpha.ts_max,
            alpha.ts_sum,
            alpha.ts_std,
            alpha.ts_product,
        ),
    ),
    ("bid_price", None, None, (alpha.ofi,)),
    ("futures_close", None, None, (alpha.futures_basis,)),
    ("open_", None, _EXPR_ALPHA_DEFAULTS, (alpha.alpha_003, alpha.alpha_006)),
    ("open_", None, None, (alpha.alpha_054, alpha.alpha_101)),
    ("low", None, None, (alpha.alpha_004,)),
    ("returns_or_equity", None, None, (evaluation.drawdown_series, evaluation.max_drawdown)),
    (
        "returns",
        None,
        None,
        (
            evaluation.sharpe_ratio,
            evaluation.sortino_ratio,
            evaluation.calmar_ratio,
            evaluation.profit_factor,
            evaluation.win_rate,
            evaluation.value_at_risk,
            evaluation.conditional_var,
            evaluation.omega_ratio,
            evaluation.tail_ratio,
            evaluation.probabilistic_sharpe_ratio,
            evaluation.deflated_sharpe_ratio,
        ),
    ),
    ("a", {"b": "other"}, None, (backtest.crosses_above, backtest.crosses_below)),
    ("entries", None, None, (backtest.signals_to_positions,)),
)

for _self_p, _renames, _defaults, _fns in _EXPR_GROUP_BINDINGS:
    for _fn in _fns:
        setattr(
            TechnicalAnalysisExpr,
            _fn.__name__,
            _bind_expr(_fn, self_param=_self_p, renames=_renames, defaults=_defaults),
        )


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
        self._df = df

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
            if isinstance(item, (list, tuple)):
                flat_exprs.extend(item)
            else:
                flat_exprs.append(item)
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


_FRAME_PASSTHROUGHS: tuple[tuple[Callable[..., Any], str | None], ...] = (
    (resample.resample_ohlcv, "resample"),
    (resample.validate_ohlcv, None),
    (resample.resample_volume_bars, None),
    (resample.resample_dollar_bars, None),
    (resample.resample_imbalance_bars, None),
    (resample.stitch_futures_roll, None),
    (evaluation.forward_returns, None),
    (evaluation.information_coefficient, None),
    (evaluation.ic_summary, None),
    (evaluation.ic_decay, None),
    (evaluation.quantile_analysis, None),
    (evaluation.factor_turnover, None),
    (inference.ic_matrix, None),
    (inference.long_short_returns, None),
    (inference.falsify_factors, None),
    (inference.sweep_robustness, None),
    (backtest.vectorized_backtest, "backtest"),
    (backtest.parameter_sweep, None),
)

for _p_fn, _p_alias in _FRAME_PASSTHROUGHS:
    _m_name, _m_fn = _bind_frame_passthrough(_p_fn, _p_alias)
    setattr(TechnicalAnalysisFrame, _m_name, _m_fn)

_ALPHA_FUNCS: tuple[Callable[..., pl.Expr], ...] = (
    alpha.alpha_001,
    alpha.alpha_002,
    alpha.alpha_003,
    alpha.alpha_004,
    alpha.alpha_005,
    alpha.alpha_006,
    alpha.alpha_007,
    alpha.alpha_008,
    alpha.alpha_009,
    alpha.alpha_010,
    alpha.alpha_011,
    alpha.alpha_012,
    alpha.alpha_013,
    alpha.alpha_014,
    alpha.alpha_015,
    alpha.alpha_016,
    alpha.alpha_017,
    alpha.alpha_018,
    alpha.alpha_019,
    alpha.alpha_020,
    alpha.alpha_021,
    alpha.alpha_022,
    alpha.alpha_023,
    alpha.alpha_024,
    alpha.alpha_025,
    alpha.alpha_026,
    alpha.alpha_027,
    alpha.alpha_028,
    alpha.alpha_029,
    alpha.alpha_030,
    alpha.alpha_041,
    alpha.alpha_054,
    alpha.alpha_101,
)
_ALPHA_FRAME_GROUP_BY = frozenset(
    {
        "alpha_006",
        "alpha_007",
        "alpha_009",
        "alpha_012",
        "alpha_021",
        "alpha_023",
        "alpha_024",
        "alpha_026",
    }
)


def _make_alpha_frame_spec(fn: Callable[..., pl.Expr]) -> _FrameExprSpec:
    p_list: list[tuple[str, Any]] = [
        (
            p.name,
            _FRAME_ALPHA_DEFAULTS.get(p.name, p.default) if p.default is _EMPTY else p.default,
        )
        for p in inspect.signature(fn).parameters.values()
    ]
    p_list.append(("alias", fn.__name__))
    if fn.__name__ in _ALPHA_FRAME_GROUP_BY:
        p_list.append(("group_by", None))
    return _FrameExprSpec(fn=fn, params=tuple(p_list))


_FRAME_EXPR_SPECS: tuple[_FrameExprSpec, ...] = (
    # Trend
    _col_period_spec(trend.sma, 10),
    _col_period_spec(trend.ema, 10),
    _col_period_spec(trend.wma, 10),
    _col_period_spec(trend.hma, 10),
    _col_period_spec(trend.dema, 10),
    _col_period_spec(trend.tema, 10),
    _col_period_spec(trend.zlema, 10),
    _struct_spec(
        trend.supertrend, period=10, multiplier=3.0, high="high", low="low", close="close"
    ),
    _struct_spec(trend.ssl_channel, period=10, high="high", low="low", close="close"),
    _struct_spec(trend.adx, period=14, high="high", low="low", close="close"),
    _struct_spec(trend.aroon, period=25, high="high", low="low"),
    _struct_spec(
        trend.trend_intensity_index,
        _alias="tii",
        _renames={"col": "close"},
        period=30,
        signal_period=None,
        col="close",
    ),
    _struct_spec(trend.heikin_ashi, open="open", high="high", low="low", close="close"),
    # Momentum
    _col_period_spec(momentum.rsi, 14),
    _scalar_spec(
        momentum.connors_rsi,
        _alias="connors_rsi",
        _renames={"col": "close"},
        rsi_period=3,
        streak_period=2,
        rank_period=100,
        col="close",
    ),
    _struct_spec(
        momentum.macd,
        _renames={"col": "close"},
        fast_period=12,
        slow_period=26,
        signal_period=9,
        col="close",
    ),
    _struct_spec(
        momentum.stochastic,
        k_period=14,
        d_period=3,
        smooth_k=3,
        high="high",
        low="low",
        close="close",
    ),
    _struct_spec(
        momentum.stoch_rsi,
        _renames={"col": "close"},
        rsi_period=14,
        stoch_period=14,
        k_period=3,
        d_period=3,
        col="close",
    ),
    _scalar_spec(
        momentum.stc,
        _alias="stc",
        _renames={"col": "close"},
        fast_period=23,
        slow_period=50,
        cycle_period=10,
        factor=0.5,
        col="close",
    ),
    _struct_spec(
        momentum.squeeze_momentum,
        bb_period=20,
        bb_mult=2.0,
        kc_period=20,
        kc_mult=1.5,
        mom_period=None,
        high="high",
        low="low",
        close="close",
    ),
    _scalar_spec(
        momentum.williams_r,
        _alias_fmt="williams_r_{period}",
        period=14,
        high="high",
        low="low",
        close="close",
    ),
    _scalar_spec(
        momentum.cci,
        _alias_fmt="cci_{period}",
        period=20,
        constant=0.015,
        high="high",
        low="low",
        close="close",
    ),
    _col_period_spec(momentum.roc, 10),
    _struct_spec(momentum.trix, _renames={"col": "close"}, period=15, signal_period=9, col="close"),
    # Volatility
    _struct_spec(
        volatility.bollinger_bands,
        _prefix="bb_",
        _renames={"col": "close"},
        period=20,
        num_std=2.0,
        col="close",
    ),
    _struct_spec(
        volatility.keltner_channels,
        _prefix="kc_",
        ema_period=20,
        atr_period=10,
        multiplier=2.0,
        high="high",
        low="low",
        close="close",
    ),
    _struct_spec(volatility.donchian_channels, _prefix="dc_", period=20, high="high", low="low"),
    _scalar_spec(
        volatility.atr, _alias_fmt="atr_{period}", period=14, high="high", low="low", close="close"
    ),
    _scalar_spec(
        volatility.true_range, _alias_fmt="true_range", high="high", low="low", close="close"
    ),
    _col_period_spec(volatility.stddev, 20),
    # Volume
    _scalar_spec(
        volume.obv,
        _alias_fmt="obv",
        _renames={"volume_col": "volume"},
        close="close",
        volume_col="volume",
    ),
    _scalar_spec(
        volume.vwap,
        _alias_fmt="vwap",
        _renames={"volume_col": "volume"},
        high="high",
        low="low",
        close="close",
        volume_col="volume",
        period=None,
    ),
    _struct_spec(
        volume.vwap_bands,
        _prefix="vwap_",
        _renames={"volume_col": "volume"},
        high="high",
        low="low",
        close="close",
        volume_col="volume",
        period=None,
        mult1=1.0,
        mult2=2.0,
        mult3=3.0,
    ),
    _struct_spec(
        volume.session_vwap,
        _alias="session_vwap_bands",
        _prefix="session_",
        _renames={"volume_col": "volume"},
        high="high",
        low="low",
        close="close",
        volume_col="volume",
        session_by=None,
        mult1=1.0,
        mult2=2.0,
    ),
    _scalar_spec(
        volume.cmf,
        _alias_fmt="cmf_{period}",
        _renames={"volume_col": "volume"},
        period=20,
        high="high",
        low="low",
        close="close",
        volume_col="volume",
    ),
    _scalar_spec(
        volume.mfi,
        _alias_fmt="mfi_{period}",
        _renames={"volume_col": "volume"},
        period=14,
        high="high",
        low="low",
        close="close",
        volume_col="volume",
    ),
    # Statistics
    _col_period_spec(stats.zscore, 20),
    _col_period_spec(stats.percentile_rank, 20),
    _scalar_spec(
        stats.frac_diff,
        _alias_fmt="frac_diff_{window}",
        _renames={"col": "close"},
        d=0.4,
        window=20,
        col="close",
    ),
    # Alpha & Microstructure
    _col_period_spec(alpha.ts_rank, 10),
    _scalar_spec(
        alpha.fractal_dimension,
        _alias_fmt="fractal_dimension_{period}",
        period=20,
        high="high",
        low="low",
    ),
    _col_period_spec(alpha.hurst_exponent, 50, alias_base="hurst"),
    _scalar_spec(
        alpha.cs_rank,
        _alias_fmt="cs_rank",
        _renames={"col": "expr"},
        _has_gb=False,
        col="close",
        over=None,
        method="average",
        percentile=True,
    ),
    _scalar_spec(
        alpha.neutralize,
        _alias_fmt="neutralized",
        _renames={"col": "expr"},
        _has_gb=False,
        col="close",
        by="sector",
        over=None,
        standardize=False,
    ),
    _scalar_spec(
        alpha.ts_scale,
        _alias_fmt="scaled",
        _renames={"col": "expr"},
        _has_gb=False,
        col="close",
        a=1.0,
        over=None,
    ),
    _col_period_spec(alpha.ts_delta, 1, alias_base="delta", col_target="expr"),
    _col_period_spec(alpha.ts_delay, 1, alias_base="delay", col_target="expr"),
    _scalar_spec(
        alpha.signed_power,
        _alias_fmt="signed_power_{exponent}",
        _renames={"col": "expr"},
        exponent=2.0,
        col="close",
    ),
    _col_period_spec(alpha.ts_min, 10, col_target="expr"),
    _col_period_spec(alpha.ts_max, 10, col_target="expr"),
    _col_period_spec(alpha.ts_sum, 10, col_target="expr"),
    _col_period_spec(alpha.ts_std, 10, col_target="expr"),
    _scalar_spec(
        alpha.ts_correlation,
        _alias_fmt="ts_corr_{period}",
        _renames={"col": "x", "other": "y"},
        other=_EMPTY,
        period=10,
        col="close",
    ),
    _scalar_spec(
        alpha.ts_covariance,
        _alias_fmt="ts_cov_{period}",
        _renames={"col": "x", "other": "y"},
        other=_EMPTY,
        period=10,
        col="close",
    ),
    _col_period_spec(alpha.ts_product, 10, alias_base="ts_prod", col_target="expr"),
    _scalar_spec(
        alpha.cs_zscore,
        _alias="cs_zscore",
        _renames={"col": "expr"},
        _has_gb=False,
        col="close",
        over=None,
        clip_bound=None,
    ),
    _scalar_spec(
        alpha.cs_winsorize,
        _alias="cs_winsorized",
        _renames={"col": "expr"},
        _has_gb=False,
        col="close",
        over=None,
        method="quantile",
        lower=0.01,
        upper=0.99,
        threshold=3.0,
    ),
    _scalar_spec(
        alpha.orthogonalize,
        _alias="orthogonalized",
        _renames={"col": "expr"},
        _has_gb=False,
        col=_EMPTY,
        factors=_EMPTY,
        over=None,
        standardize=True,
        ridge=0.0,
    ),
    _scalar_spec(
        alpha.amihud_illiquidity,
        _alias_fmt="amihud_{period}",
        period=20,
        scale=1_000_000.0,
        close="close",
        volume="volume",
    ),
    _col_period_spec(alpha.roll_spread, 20),
    _scalar_spec(
        alpha.corwin_schultz_spread,
        _alias_fmt="cs_spread_{period}",
        period=20,
        high="high",
        low="low",
    ),
    _scalar_spec(
        alpha.kyle_lambda,
        _alias_fmt="kyle_lambda_{period}",
        period=20,
        scale=1_000_000.0,
        close="close",
        volume="volume",
    ),
    _scalar_spec(alpha.vpin, _alias_fmt="vpin_{period}", period=20, close="close", volume="volume"),
    _scalar_spec(
        alpha.ofi,
        _alias_fmt=lambda a: f"ofi_{a['period']}" if a["period"] > 1 else "ofi",
        bid_price="bid_price",
        bid_size="bid_size",
        ask_price="ask_price",
        ask_size="ask_size",
        period=1,
    ),
    _struct_spec(
        alpha.futures_basis,
        futures_close="close",
        spot_close="spot_close",
        days_to_expiry=None,
        period=20,
        annualization=365.0,
    ),
    _struct_spec(alpha.oi_regime, close="close", open_interest="open_interest", period=1),
    *(_make_alpha_frame_spec(fn) for fn in _ALPHA_FUNCS),
    # Evaluation & Labeling
    _scalar_spec(
        evaluation.drawdown_series,
        _alias="drawdown",
        _renames={"col": "returns_or_equity"},
        col="close",
        is_returns=False,
    ),
    _struct_spec(
        evaluation.triple_barrier_labels,
        _alias="triple_barrier",
        _prefix="tb_",
        close="close",
        volatility=None,
        horizon=10,
        pt_mult=2.0,
        sl_mult=2.0,
        min_ret=0.0,
        vol_period=20,
        side=None,
    ),
    _struct_spec(
        evaluation.trend_scanning_labels,
        _alias="trend_scan",
        _prefix="ts_label_",
        close="close",
        min_window=5,
        max_window=20,
        step=1,
    ),
)

for _spec in _FRAME_EXPR_SPECS:
    _m_name, _m_fn = _bind_frame_expr(_spec)
    setattr(TechnicalAnalysisFrame, _m_name, _m_fn)
