"""Polars namespace registration for technical analysis expressions and frames."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import polars as pl

from . import alpha, momentum, resample, stats, trend, volatility, volume

if TYPE_CHECKING:
    from .core import IntoExpr

FrameType = TypeVar("FrameType", pl.DataFrame, pl.LazyFrame)


@pl.api.register_expr_namespace("anta")
@pl.api.register_expr_namespace("ta")
class TechnicalAnalysisExpr:
    """Namespace for technical analysis and alpha factor operations on Polars expressions.

    Allows chaining indicators directly on `pl.col(...)` via `.anta.<indicator>()` or `.ta.<indicator>()`.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # =========================================================================
    # Trend Indicators (19)
    # =========================================================================

    def sma(self, period: int = 10) -> pl.Expr:
        """Simple Moving Average (SMA)."""
        return trend.sma(self._expr, period=period)

    def ema(self, period: int = 10) -> pl.Expr:
        """Exponential Moving Average (EMA)."""
        return trend.ema(self._expr, period=period)

    def wma(self, period: int = 10) -> pl.Expr:
        """Weighted Moving Average (WMA)."""
        return trend.wma(self._expr, period=period)

    def hma(self, period: int = 10) -> pl.Expr:
        """Hull Moving Average (HMA)."""
        return trend.hma(self._expr, period=period)

    def dema(self, period: int = 10) -> pl.Expr:
        """Double Exponential Moving Average (DEMA)."""
        return trend.dema(self._expr, period=period)

    def tema(self, period: int = 10) -> pl.Expr:
        """Triple Exponential Moving Average (TEMA)."""
        return trend.tema(self._expr, period=period)

    def zlema(self, period: int = 10) -> pl.Expr:
        """Zero-Lag Exponential Moving Average (ZLEMA)."""
        return trend.zlema(self._expr, period=period)

    def vortex(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Vortex Indicator (VI)."""
        return trend.vortex(high=high, low=low, close=self._expr, period=period)

    def adx(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Average Directional Index (ADX) and (+DI, -DI)."""
        return trend.adx(high=high, low=low, close=self._expr, period=period)

    def aroon(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 25,
    ) -> pl.Expr:
        """Aroon Indicator and Oscillator."""
        return trend.aroon(high=high, low=low, period=period)

    def chop(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Choppiness Index (CHOP)."""
        return trend.chop(high=high, low=low, close=self._expr, period=period)

    def t3(self, period: int = 5, volume_factor: float = 0.7) -> pl.Expr:
        """Tillson T3 Moving Average."""
        return trend.t3(self._expr, period=period, volume_factor=volume_factor)

    def alma(
        self,
        period: int = 9,
        offset: float = 0.85,
        sigma: float = 6.0,
    ) -> pl.Expr:
        """Arnaud Legoux Moving Average (ALMA)."""
        return trend.alma(self._expr, period=period, offset=offset, sigma=sigma)

    def mass_index(
        self,
        high: IntoExpr,
        low: IntoExpr,
        fast_period: int = 9,
        slow_period: int = 25,
    ) -> pl.Expr:
        """Mass Index."""
        return trend.mass_index(
            high=high, low=low, fast_period=fast_period, slow_period=slow_period
        )

    def supertrend(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pl.Expr:
        """SuperTrend Indicator."""
        return trend.supertrend(
            high=high, low=low, close=self._expr, period=period, multiplier=multiplier
        )

    def psar(
        self,
        high: IntoExpr,
        low: IntoExpr,
        af_step: float = 0.02,
        af_max: float = 0.2,
    ) -> pl.Expr:
        """Parabolic SAR (Stop and Reverse)."""
        return trend.psar(high=high, low=low, af_step=af_step, af_max=af_max)

    def kama(
        self,
        period: int = 10,
        fast_span: int = 2,
        slow_span: int = 30,
    ) -> pl.Expr:
        """Kaufman's Adaptive Moving Average (KAMA)."""
        return trend.kama(self._expr, period=period, fast_span=fast_span, slow_span=slow_span)

    def mcginley_dynamic(self, period: int = 10, k: float = 0.6) -> pl.Expr:
        """McGinley Dynamic tracking average."""
        return trend.mcginley_dynamic(self._expr, period=period, k=k)

    def vidya(self, period: int = 9, cmo_period: int = 9) -> pl.Expr:
        """Variable Index Dynamic Average (VIDYA)."""
        return trend.vidya(self._expr, period=period, cmo_period=cmo_period)

    def center_of_gravity(self, period: int = 10) -> pl.Expr:
        """Center of Gravity (CG) Oscillator."""
        return trend.center_of_gravity(self._expr, period=period)

    def ichimoku(
        self,
        high: IntoExpr,
        low: IntoExpr,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26,
    ) -> pl.Expr:
        """Ichimoku Kinko Hyo Cloud."""
        return trend.ichimoku(
            high=high,
            low=low,
            close=self._expr,
            tenkan_period=tenkan_period,
            kijun_period=kijun_period,
            senkou_b_period=senkou_b_period,
            displacement=displacement,
        )

    def pivot_points(self, high: IntoExpr, low: IntoExpr) -> pl.Expr:
        """Floor Trader Standard Pivot Points."""
        return trend.pivot_points(high=high, low=low, close=self._expr)

    def fibonacci_pivots(self, high: IntoExpr, low: IntoExpr) -> pl.Expr:
        """Fibonacci Pivot Points."""
        return trend.fibonacci_pivots(high=high, low=low, close=self._expr)

    def camarilla_pivots(self, high: IntoExpr, low: IntoExpr) -> pl.Expr:
        """Camarilla Pivot Points."""
        return trend.camarilla_pivots(high=high, low=low, close=self._expr)

    def woodie_pivots(self, open_: IntoExpr, high: IntoExpr, low: IntoExpr) -> pl.Expr:
        """Woodie's Pivot Points."""
        return trend.woodie_pivots(open_=open_, high=high, low=low, close=self._expr)

    def ssl_channel(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 10,
    ) -> pl.Expr:
        """SSL Channel (Securement Stop Loss Channel)."""
        return trend.ssl_channel(high=high, low=low, close=self._expr, period=period)

    def trend_intensity_index(
        self,
        period: int = 30,
        signal_period: int | None = None,
    ) -> pl.Expr:
        """Trend Intensity Index (TII)."""
        return trend.trend_intensity_index(
            close=self._expr,
            period=period,
            signal_period=signal_period,
        )

    # =========================================================================
    # Momentum Indicators (20)
    # =========================================================================

    def rsi(self, period: int = 14) -> pl.Expr:
        """Relative Strength Index (RSI)."""
        return momentum.rsi(self._expr, period=period)

    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Moving Average Convergence Divergence (MACD)."""
        return momentum.macd(
            self._expr,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def stochastic(
        self,
        high: IntoExpr,
        low: IntoExpr,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> pl.Expr:
        """Stochastic Oscillator (%K, %D)."""
        return momentum.stochastic(
            high=high,
            low=low,
            close=self._expr,
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
        )

    def williams_r(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Williams %R."""
        return momentum.williams_r(
            high=high,
            low=low,
            close=self._expr,
            period=period,
        )

    def cci(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        constant: float = 0.015,
    ) -> pl.Expr:
        """Commodity Channel Index (CCI)."""
        return momentum.cci(
            high=high,
            low=low,
            close=self._expr,
            period=period,
            constant=constant,
        )

    def roc(self, period: int = 10) -> pl.Expr:
        """Rate of Change (ROC)."""
        return momentum.roc(self._expr, period=period)

    def stoch_rsi(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> pl.Expr:
        """Stochastic RSI (StochRSI)."""
        return momentum.stoch_rsi(
            self._expr,
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            k_period=k_period,
            d_period=d_period,
        )

    def stc(
        self,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        factor: float = 0.5,
    ) -> pl.Expr:
        """Schaff Trend Cycle (STC) Oscillator."""
        return momentum.stc(
            self._expr,
            fast_period=fast_period,
            slow_period=slow_period,
            cycle_period=cycle_period,
            factor=factor,
        )

    def awesome_oscillator(
        self,
        high: IntoExpr,
        low: IntoExpr,
        fast_period: int = 5,
        slow_period: int = 34,
    ) -> pl.Expr:
        """Awesome Oscillator (AO)."""
        return momentum.awesome_oscillator(
            high=high,
            low=low,
            fast_period=fast_period,
            slow_period=slow_period,
        )

    def tsi(
        self,
        long_period: int = 25,
        short_period: int = 13,
    ) -> pl.Expr:
        """True Strength Index (TSI)."""
        return momentum.tsi(
            self._expr,
            long_period=long_period,
            short_period=short_period,
        )

    def cmo(self, period: int = 14) -> pl.Expr:
        """Chande Momentum Oscillator (CMO)."""
        return momentum.cmo(self._expr, period=period)

    def ppo(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Percentage Price Oscillator (PPO)."""
        return momentum.ppo(
            self._expr,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def coppock(
        self,
        wma_period: int = 10,
        fast_roc: int = 11,
        slow_roc: int = 14,
    ) -> pl.Expr:
        """Coppock Curve."""
        return momentum.coppock(
            self._expr, wma_period=wma_period, fast_roc=fast_roc, slow_roc=slow_roc
        )

    def ultimate_oscillator(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> pl.Expr:
        """Ultimate Oscillator (UO)."""
        return momentum.ultimate_oscillator(
            high=high,
            low=low,
            close=self._expr,
            period1=period1,
            period2=period2,
            period3=period3,
        )

    def fisher_transform(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 9,
    ) -> pl.Expr:
        """Fisher Transform."""
        return momentum.fisher_transform(high=high, low=low, period=period)

    def rvi(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Relative Vigor Index (RVI)."""
        return momentum.rvi(open=open, high=high, low=low, close=self._expr, period=period)

    def kst(
        self,
        r1: int = 10,
        r2: int = 15,
        r3: int = 20,
        r4: int = 30,
        s1: int = 10,
        s2: int = 10,
        s3: int = 10,
        s4: int = 15,
        signal_period: int = 9,
    ) -> pl.Expr:
        """Know Sure Thing (KST)."""
        return momentum.kst(
            self._expr,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            s1=s1,
            s2=s2,
            s3=s3,
            s4=s4,
            signal_period=signal_period,
        )

    def dpo(self, period: int = 20) -> pl.Expr:
        """Detrended Price Oscillator (DPO)."""
        return momentum.dpo(self._expr, period=period)

    def bop(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
    ) -> pl.Expr:
        """Balance of Power (BOP)."""
        return momentum.bop(open=open, high=high, low=low, close=self._expr)

    def cfo(self, period: int = 14) -> pl.Expr:
        """Chande Forecast Oscillator (CFO)."""
        return momentum.cfo(self._expr, period=period)

    def squeeze_momentum(
        self,
        high: IntoExpr,
        low: IntoExpr,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_period: int | None = None,
    ) -> pl.Expr:
        """Squeeze Momentum Indicator (John Carter / LazyBear)."""
        return momentum.squeeze_momentum(
            high=high,
            low=low,
            close=self._expr,
            bb_period=bb_period,
            bb_mult=bb_mult,
            kc_period=kc_period,
            kc_mult=kc_mult,
            mom_period=mom_period,
        )

    # =========================================================================
    # Volatility Indicators (14)
    # =========================================================================

    def true_range(
        self,
        high: IntoExpr,
        low: IntoExpr,
    ) -> pl.Expr:
        """True Range (TR)."""
        return volatility.true_range(high=high, low=low, close=self._expr)

    def atr(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Average True Range (ATR)."""
        return volatility.atr(high=high, low=low, close=self._expr, period=period)

    def bollinger_bands(
        self,
        period: int = 20,
        num_std: float = 2.0,
    ) -> pl.Expr:
        """Bollinger Bands."""
        return volatility.bollinger_bands(self._expr, period=period, num_std=num_std)

    def keltner_channels(
        self,
        high: IntoExpr,
        low: IntoExpr,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pl.Expr:
        """Keltner Channels."""
        return volatility.keltner_channels(
            high=high,
            low=low,
            close=self._expr,
            ema_period=ema_period,
            atr_period=atr_period,
            multiplier=multiplier,
        )

    def donchian_channels(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Donchian Channels."""
        return volatility.donchian_channels(high=high, low=low, period=period)

    def stddev(self, period: int = 20) -> pl.Expr:
        """Rolling Standard Deviation."""
        return volatility.stddev(self._expr, period=period)

    def ulcer_index(self, period: int = 14) -> pl.Expr:
        """Ulcer Index (UI)."""
        return volatility.ulcer_index(self._expr, period=period)

    def chandelier_exit(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 22,
        multiplier: float = 3.0,
    ) -> pl.Expr:
        """Chandelier Exit."""
        return volatility.chandelier_exit(
            high=high,
            low=low,
            close=self._expr,
            period=period,
            multiplier=multiplier,
        )

    def natr(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Normalized Average True Range (NATR)."""
        return volatility.natr(high=high, low=low, close=self._expr, period=period)

    def historical_volatility(
        self,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Annualized Historical Volatility (HV)."""
        return volatility.historical_volatility(
            self._expr, period=period, trading_periods=trading_periods
        )

    def parkinson_volatility(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Parkinson Volatility."""
        return volatility.parkinson_volatility(
            high=high,
            low=low,
            period=period,
            trading_periods=trading_periods,
        )

    def garman_klass_volatility(
        self,
        open: IntoExpr,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        trading_periods: int = 252,
    ) -> pl.Expr:
        """Garman-Klass Volatility."""
        return volatility.garman_klass_volatility(
            open=open,
            high=high,
            low=low,
            close=self._expr,
            period=period,
            trading_periods=trading_periods,
        )

    def chaikin_volatility(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 10,
        roc_period: int = 10,
    ) -> pl.Expr:
        """Chaikin's Volatility."""
        return volatility.chaikin_volatility(
            high=high, low=low, period=period, roc_period=roc_period
        )

    def donchian_width(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Donchian Channel Width."""
        return volatility.donchian_width(high=high, low=low, period=period)

    def envelopes(self, period: int = 20, percentage: float = 0.025) -> pl.Expr:
        """Moving Average Envelopes."""
        return volatility.envelopes(self._expr, period=period, percentage=percentage)

    def keltner_width(
        self,
        high: IntoExpr,
        low: IntoExpr,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pl.Expr:
        """Normalized Keltner Channel Width (Squeeze metric)."""
        return volatility.keltner_width(
            high=high,
            low=low,
            close=self._expr,
            ema_period=ema_period,
            atr_period=atr_period,
            multiplier=multiplier,
        )

    def bollinger_delta(self, period: int = 20, num_std: float = 2.0) -> pl.Expr:
        """Bollinger Band Delta (Band Spread)."""
        return volatility.bollinger_delta(self._expr, period=period, num_std=num_std)

    def starc_bands(
        self,
        high: IntoExpr,
        low: IntoExpr,
        period: int = 20,
        atr_period: int = 15,
        multiplier: float = 2.0,
    ) -> pl.Expr:
        """STARC Bands (Stoller Average Range Channels)."""
        return volatility.starc_bands(
            high=high,
            low=low,
            close=self._expr,
            period=period,
            atr_period=atr_period,
            multiplier=multiplier,
        )

    # =========================================================================
    # Volume Indicators (14)
    # =========================================================================

    def obv(self, volume_col: IntoExpr) -> pl.Expr:
        """On-Balance Volume (OBV)."""
        return volume.obv(close=self._expr, volume=volume_col)

    def vwap(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int | None = None,
    ) -> pl.Expr:
        """Volume-Weighted Average Price (VWAP)."""
        return volume.vwap(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def vwap_bands(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int | None = None,
        mult1: float = 1.0,
        mult2: float = 2.0,
        mult3: float = 3.0,
    ) -> pl.Expr:
        """Volume-Weighted Average Price with Standard Deviation Bands."""
        return volume.vwap_bands(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
            mult1=mult1,
            mult2=mult2,
            mult3=mult3,
        )

    def vwma(self, volume_col: IntoExpr, period: int = 20) -> pl.Expr:
        """Volume-Weighted Moving Average (VWMA)."""
        return volume.vwma(close=self._expr, volume=volume_col, period=period)

    def cmf(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 20,
    ) -> pl.Expr:
        """Chaikin Money Flow (CMF)."""
        return volume.cmf(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def adl(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
    ) -> pl.Expr:
        """Accumulation/Distribution Line (ADL)."""
        return volume.adl(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
        )

    def mfi(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 14,
    ) -> pl.Expr:
        """Money Flow Index (MFI)."""
        return volume.mfi(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            period=period,
        )

    def eom(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        period: int = 14,
        divisor: float = 100_000_000.0,
    ) -> pl.Expr:
        """Ease of Movement (EOM)."""
        return volume.eom(
            high=high,
            low=low,
            volume=volume_col,
            period=period,
            divisor=divisor,
        )

    def klinger(
        self,
        high: IntoExpr,
        low: IntoExpr,
        volume_col: IntoExpr,
        fast_period: int = 34,
        slow_period: int = 55,
        signal_period: int = 13,
    ) -> pl.Expr:
        """Klinger Volume Oscillator."""
        return volume.klinger(
            high=high,
            low=low,
            close=self._expr,
            volume=volume_col,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

    def nvi(
        self,
        volume_col: IntoExpr,
        initial: float = 1000.0,
    ) -> pl.Expr:
        """Negative Volume Index (NVI)."""
        return volume.nvi(close=self._expr, volume=volume_col, initial=initial)

    def pvi(
        self,
        volume_col: IntoExpr,
        initial: float = 1000.0,
    ) -> pl.Expr:
        """Positive Volume Index (PVI)."""
        return volume.pvi(close=self._expr, volume=volume_col, initial=initial)

    def pvt(self, volume_col: IntoExpr) -> pl.Expr:
        """Price Volume Trend (PVT)."""
        return volume.pvt(close=self._expr, volume=volume_col)

    def force_index(self, volume_col: IntoExpr, period: int = 13) -> pl.Expr:
        """Elder's Force Index (EFI)."""
        return volume.force_index(close=self._expr, volume=volume_col, period=period)

    def vpci(
        self,
        volume_col: IntoExpr,
        period_short: int = 5,
        period_long: int = 25,
    ) -> pl.Expr:
        """Volume Price Confirmation Indicator (VPCI)."""
        return volume.vpci(
            close=self._expr,
            volume=volume_col,
            period_short=period_short,
            period_long=period_long,
        )

    def vr(self, volume_col: IntoExpr, period: int = 26) -> pl.Expr:
        """Volume Ratio (VR)."""
        return volume.vr(close=self._expr, volume=volume_col, period=period)

    # =========================================================================
    # Quantitative Statistics & Regression Transforms (12)
    # =========================================================================

    def zscore(self, period: int = 20) -> pl.Expr:
        """Rolling Z-Score."""
        return stats.zscore(self._expr, period=period)

    def percentile_rank(self, period: int = 20) -> pl.Expr:
        """Rolling Percentile Rank."""
        return stats.percentile_rank(self._expr, period=period)

    def skewness(self, period: int = 20) -> pl.Expr:
        """Rolling Sample Skewness."""
        return stats.skewness(self._expr, period=period)

    def kurtosis(self, period: int = 20) -> pl.Expr:
        """Rolling Excess Kurtosis."""
        return stats.kurtosis(self._expr, period=period)

    def rolling_cov(self, other: IntoExpr, period: int = 20) -> pl.Expr:
        """Rolling Covariance with another series."""
        return stats.rolling_cov(self._expr, other, period=period)

    def rolling_corr(self, other: IntoExpr, period: int = 20) -> pl.Expr:
        """Rolling Correlation with another series."""
        return stats.rolling_corr(self._expr, other, period=period)

    def rolling_beta(self, benchmark_returns: IntoExpr, period: int = 60) -> pl.Expr:
        """Rolling CAPM Beta relative to a benchmark."""
        return stats.rolling_beta(self._expr, benchmark_returns, period=period)

    def rolling_alpha(
        self,
        benchmark_returns: IntoExpr,
        rf: float = 0.0,
        period: int = 60,
    ) -> pl.Expr:
        """Rolling Jensen's Alpha relative to a benchmark."""
        return stats.rolling_alpha(self._expr, benchmark_returns, rf=rf, period=period)

    def linear_regression_slope(self, period: int = 14) -> pl.Expr:
        """Rolling OLS Linear Regression Slope."""
        return stats.linear_regression_slope(self._expr, period=period)

    def linear_regression_intercept(self, period: int = 14) -> pl.Expr:
        """Rolling OLS Linear Regression Intercept."""
        return stats.linear_regression_intercept(self._expr, period=period)

    def linear_regression_angle(self, period: int = 14) -> pl.Expr:
        """Rolling Linear Regression Angle (degrees)."""
        return stats.linear_regression_angle(self._expr, period=period)

    def linear_regression_forecast(self, period: int = 14) -> pl.Expr:
        """Rolling Linear Regression Forecast value."""
        return stats.linear_regression_forecast(self._expr, period=period)

    # =========================================================================
    # Alpha Factor Primitives & Signal Transforms (12)
    # =========================================================================

    def ts_rank(self, period: int = 10) -> pl.Expr:
        """Time-series Rank."""
        return alpha.ts_rank(self._expr, period=period)

    def ts_argmax(self, period: int = 10) -> pl.Expr:
        """Time-series ArgMax (bars ago of maximum)."""
        return alpha.ts_argmax(self._expr, period=period)

    def ts_argmin(self, period: int = 10) -> pl.Expr:
        """Time-series ArgMin (bars ago of minimum)."""
        return alpha.ts_argmin(self._expr, period=period)

    def decay_linear(self, period: int = 10) -> pl.Expr:
        """Linearly Decayed Weighted Moving Average."""
        return alpha.decay_linear(self._expr, period=period)

    def ts_zscore(self, period: int = 20) -> pl.Expr:
        """Time-series Rolling Z-Score."""
        return alpha.ts_zscore(self._expr, period=period)

    def returns_divergence(self, volume: IntoExpr, period: int = 14) -> pl.Expr:
        """Price-Volume Returns Divergence."""
        return alpha.returns_divergence(self._expr, volume=volume, period=period)

    def hl_ratio(self, low: IntoExpr) -> pl.Expr:
        """Log High-to-Low Ratio."""
        return alpha.hl_ratio(high=self._expr, low=low)

    def co_ratio(self, open_: IntoExpr) -> pl.Expr:
        """Log Close-to-Open Ratio."""
        return alpha.co_ratio(close=self._expr, open_=open_)

    def volume_weighted_momentum(self, volume: IntoExpr, period: int = 10) -> pl.Expr:
        """Volume-Weighted Momentum."""
        return alpha.volume_weighted_momentum(self._expr, volume=volume, period=period)

    def relative_spread(self, low: IntoExpr, close: IntoExpr) -> pl.Expr:
        """Normalized Relative Spread ((High - Low) / Close)."""
        return alpha.relative_spread(high=self._expr, low=low, close=close)

    def twap(self, low: IntoExpr, close: IntoExpr, period: int = 20) -> pl.Expr:
        """Rolling Time-Weighted Average Price (TWAP)."""
        return alpha.twap(high=self._expr, low=low, close=close, period=period)

    def entropy(self, period: int = 20, bins: int = 5) -> pl.Expr:
        """Rolling Shannon Entropy of returns."""
        return alpha.entropy(self._expr, period=period, bins=bins)

    def fractal_dimension(self, low: IntoExpr, period: int = 20) -> pl.Expr:
        """Ehlers' Fractal Dimension."""
        return alpha.fractal_dimension(high=self._expr, low=low, period=period)

    def hurst_exponent(self, period: int = 50) -> pl.Expr:
        """Rolling Hurst Exponent."""
        return alpha.hurst_exponent(close=self._expr, period=period)

    def cs_rank(
        self,
        over: IntoExpr | Sequence[str] | None = None,
        method: Literal["average", "min", "max", "dense", "ordinal", "random"] = "average",
        percentile: bool = True,
    ) -> pl.Expr:
        """Cross-Sectional Rank (WorldQuant Alpha primitive 'rank(x)')."""
        return alpha.cs_rank(self._expr, over=over, method=method, percentile=percentile)

    def neutralize(
        self,
        by: IntoExpr | Sequence[str],
        over: IntoExpr | Sequence[str] | None = None,
        standardize: bool = False,
    ) -> pl.Expr:
        """Group / Industry Neutralization (WorldQuant Alpha primitive 'indneutralize(x, g)')."""
        return alpha.neutralize(self._expr, by=by, over=over, standardize=standardize)

    def ts_scale(
        self,
        a: float = 1.0,
        over: IntoExpr | Sequence[str] | None = None,
    ) -> pl.Expr:
        """Scale Primitive (WorldQuant Alpha primitive 'scale(x, a)')."""
        return alpha.ts_scale(self._expr, a=a, over=over)

    def ts_delta(self, period: int = 1) -> pl.Expr:
        """Time-series Delta (WorldQuant Alpha primitive 'delta(x, d)')."""
        return alpha.ts_delta(self._expr, period=period)

    def ts_delay(self, period: int = 1) -> pl.Expr:
        """Time-series Delay (WorldQuant Alpha primitive 'delay(x, d)')."""
        return alpha.ts_delay(self._expr, period=period)

    def signed_power(self, exponent: float = 2.0) -> pl.Expr:
        """Signed Power (WorldQuant Alpha primitive 'signed_power(x, a)')."""
        return alpha.signed_power(self._expr, exponent=exponent)

    def ts_min(self, period: int = 10) -> pl.Expr:
        """Time-series Rolling Minimum (WorldQuant Alpha primitive 'ts_min(x, d)')."""
        return alpha.ts_min(self._expr, period=period)

    def ts_max(self, period: int = 10) -> pl.Expr:
        """Time-series Rolling Maximum (WorldQuant Alpha primitive 'ts_max(x, d)')."""
        return alpha.ts_max(self._expr, period=period)

    def ts_sum(self, period: int = 10) -> pl.Expr:
        """Time-series Rolling Sum (WorldQuant Alpha primitive 'ts_sum(x, d)')."""
        return alpha.ts_sum(self._expr, period=period)

    def ts_std(self, period: int = 10) -> pl.Expr:
        """Time-series Rolling Standard Deviation (WorldQuant Alpha primitive 'ts_std(x, d)')."""
        return alpha.ts_std(self._expr, period=period)

    def ts_correlation(self, other: IntoExpr, period: int = 10) -> pl.Expr:
        """Time-series Correlation (WorldQuant Alpha primitive 'correlation(x, y, d)')."""
        return alpha.ts_correlation(self._expr, other, period=period)

    def ts_covariance(self, other: IntoExpr, period: int = 10) -> pl.Expr:
        """Time-series Covariance (WorldQuant Alpha primitive 'covariance(x, y, d)')."""
        return alpha.ts_covariance(self._expr, other, period=period)

    def ts_product(self, period: int = 10) -> pl.Expr:
        """Time-series Rolling Product (WorldQuant Alpha primitive 'product(x, d)')."""
        return alpha.ts_product(self._expr, period=period)

    def alpha_001(
        self,
        returns: IntoExpr | None = None,
        over: IntoExpr | Sequence[str] | None = None,
    ) -> pl.Expr:
        """WorldQuant Formulaic Alpha #1."""
        return alpha.alpha_001(close=self._expr, returns=returns, over=over)

    def alpha_041(
        self,
        low: IntoExpr,
        close: IntoExpr,
        volume: IntoExpr,
    ) -> pl.Expr:
        """WorldQuant Formulaic Alpha #41."""
        return alpha.alpha_041(high=self._expr, low=low, close=close, volume=volume)

    def alpha_054(
        self,
        high: IntoExpr,
        low: IntoExpr,
        close: IntoExpr,
    ) -> pl.Expr:
        """WorldQuant Formulaic Alpha #54."""
        return alpha.alpha_054(open_=self._expr, high=high, low=low, close=close)

    def alpha_101(
        self,
        high: IntoExpr,
        low: IntoExpr,
        close: IntoExpr,
    ) -> pl.Expr:
        """WorldQuant Formulaic Alpha #101."""
        return alpha.alpha_101(open_=self._expr, high=high, low=low, close=close)


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

    # =========================================================================
    # Resampling & Validation
    # =========================================================================

    def resample(
        self,
        every: str,
        time_col: str = "timestamp",
        group_by: str | Sequence[str] | None = None,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str | None = "volume",
        closed: Literal["left", "right"] = "left",
        label: Literal["left", "right", "datapoint"] = "left",
        additional_aggs: Sequence[pl.Expr] | None = None,
        sort: bool = True,
    ) -> FrameType:
        """Resample lower-timeframe OHLCV data into higher-timeframe bars.

        Supports single-asset and multi-ticker datasets (`group_by='symbol'`).
        """
        return resample.resample_ohlcv(
            self._df,
            every=every,
            time_col=time_col,
            group_by=group_by,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            closed=closed,
            label=label,
            additional_aggs=additional_aggs,
            sort=sort,
        )

    def validate_ohlcv(
        self,
        time_col: str | None = "timestamp",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str | None = "volume",
        group_by: str | Sequence[str] | None = None,
        check_data: bool = True,
        return_report: bool = False,
    ) -> FrameType | dict[str, Any]:
        """Validate OHLCV schema, structural integrity, and price consistency."""
        return resample.validate_ohlcv(
            self._df,
            time_col=time_col,
            open_col=open_col,
            high_col=high_col,
            low_col=low_col,
            close_col=close_col,
            volume_col=volume_col,
            group_by=group_by,
            check_data=check_data,
            return_report=return_report,
        )

    # =========================================================================
    # Batch Indicator Builders
    # =========================================================================

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

    # =========================================================================
    # Trend Indicators
    # =========================================================================

    def sma(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Simple Moving Average (SMA)."""
        alias_name = alias or f"sma_{period}"
        return self._apply_expr(trend.sma(col, period=period), alias_name, group_by=group_by)

    def ema(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Exponential Moving Average (EMA)."""
        alias_name = alias or f"ema_{period}"
        return self._apply_expr(trend.ema(col, period=period), alias_name, group_by=group_by)

    def wma(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Weighted Moving Average (WMA)."""
        alias_name = alias or f"wma_{period}"
        return self._apply_expr(trend.wma(col, period=period), alias_name, group_by=group_by)

    def hma(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Hull Moving Average (HMA)."""
        alias_name = alias or f"hma_{period}"
        return self._apply_expr(trend.hma(col, period=period), alias_name, group_by=group_by)

    def dema(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Double Exponential Moving Average (DEMA)."""
        alias_name = alias or f"dema_{period}"
        return self._apply_expr(trend.dema(col, period=period), alias_name, group_by=group_by)

    def tema(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Triple Exponential Moving Average (TEMA)."""
        alias_name = alias or f"tema_{period}"
        return self._apply_expr(trend.tema(col, period=period), alias_name, group_by=group_by)

    def zlema(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Zero-Lag Exponential Moving Average (ZLEMA)."""
        alias_name = alias or f"zlema_{period}"
        return self._apply_expr(trend.zlema(col, period=period), alias_name, group_by=group_by)

    def supertrend(
        self,
        period: int = 10,
        multiplier: float = 3.0,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "supertrend",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """SuperTrend Indicator."""
        expr = trend.supertrend(
            high=high, low=low, close=close, period=period, multiplier=multiplier
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def ssl_channel(
        self,
        period: int = 10,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "ssl_channel",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """SSL Channel (Securement Stop Loss Channel)."""
        expr = trend.ssl_channel(high=high, low=low, close=close, period=period)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def adx(
        self,
        period: int = 14,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "adx",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Average Directional Index (ADX) and (+DI, -DI)."""
        expr = trend.adx(high=high, low=low, close=close, period=period)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def aroon(
        self,
        period: int = 25,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        alias: str = "aroon",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Aroon Indicator and Oscillator."""
        expr = trend.aroon(high=high, low=low, period=period)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def trend_intensity_index(
        self,
        period: int = 30,
        signal_period: int | None = None,
        col: IntoExpr = "close",
        alias: str = "tii",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Trend Intensity Index (TII)."""
        expr = trend.trend_intensity_index(close=col, period=period, signal_period=signal_period)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    # =========================================================================
    # Momentum Indicators
    # =========================================================================

    def rsi(
        self,
        period: int = 14,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Relative Strength Index (RSI)."""
        alias_name = alias or f"rsi_{period}"
        return self._apply_expr(momentum.rsi(col, period=period), alias_name, group_by=group_by)

    def macd(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        col: IntoExpr = "close",
        alias: str = "macd",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Moving Average Convergence Divergence (MACD)."""
        expr = momentum.macd(
            col,
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def stochastic(
        self,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "stochastic",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Stochastic Oscillator (%K, %D)."""
        expr = momentum.stochastic(
            high=high,
            low=low,
            close=close,
            k_period=k_period,
            d_period=d_period,
            smooth_k=smooth_k,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def stoch_rsi(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
        col: IntoExpr = "close",
        alias: str = "stoch_rsi",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Stochastic RSI (StochRSI)."""
        expr = momentum.stoch_rsi(
            col,
            rsi_period=rsi_period,
            stoch_period=stoch_period,
            k_period=k_period,
            d_period=d_period,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def stc(
        self,
        fast_period: int = 23,
        slow_period: int = 50,
        cycle_period: int = 10,
        factor: float = 0.5,
        col: IntoExpr = "close",
        alias: str = "stc",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Schaff Trend Cycle (STC) Oscillator."""
        expr = momentum.stc(
            col,
            fast_period=fast_period,
            slow_period=slow_period,
            cycle_period=cycle_period,
            factor=factor,
        )
        return self._apply_expr(expr, alias=alias, group_by=group_by)

    def squeeze_momentum(
        self,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        mom_period: int | None = None,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "squeeze_momentum",
        unnest: bool = False,
        prefix: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Squeeze Momentum Indicator (John Carter / LazyBear)."""
        expr = momentum.squeeze_momentum(
            high=high,
            low=low,
            close=close,
            bb_period=bb_period,
            bb_mult=bb_mult,
            kc_period=kc_period,
            kc_mult=kc_mult,
            mom_period=mom_period,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def williams_r(
        self,
        period: int = 14,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Williams %R."""
        alias_name = alias or f"williams_r_{period}"
        return self._apply_expr(
            momentum.williams_r(high=high, low=low, close=close, period=period),
            alias_name,
            group_by=group_by,
        )

    def cci(
        self,
        period: int = 20,
        constant: float = 0.015,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Commodity Channel Index (CCI)."""
        alias_name = alias or f"cci_{period}"
        return self._apply_expr(
            momentum.cci(high=high, low=low, close=close, period=period, constant=constant),
            alias_name,
            group_by=group_by,
        )

    def roc(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Rate of Change (ROC)."""
        alias_name = alias or f"roc_{period}"
        return self._apply_expr(momentum.roc(col, period=period), alias_name, group_by=group_by)

    # =========================================================================
    # Volatility Indicators
    # =========================================================================

    def bollinger_bands(
        self,
        period: int = 20,
        num_std: float = 2.0,
        col: IntoExpr = "close",
        alias: str = "bollinger_bands",
        unnest: bool = False,
        prefix: str | None = "bb_",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Bollinger Bands."""
        expr = volatility.bollinger_bands(col, period=period, num_std=num_std)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def keltner_channels(
        self,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "keltner_channels",
        unnest: bool = False,
        prefix: str | None = "kc_",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Keltner Channels."""
        expr = volatility.keltner_channels(
            high=high,
            low=low,
            close=close,
            ema_period=ema_period,
            atr_period=atr_period,
            multiplier=multiplier,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def donchian_channels(
        self,
        period: int = 20,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        alias: str = "donchian_channels",
        unnest: bool = False,
        prefix: str | None = "dc_",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Donchian Channels."""
        expr = volatility.donchian_channels(high=high, low=low, period=period)
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def atr(
        self,
        period: int = 14,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Average True Range (ATR)."""
        alias_name = alias or f"atr_{period}"
        return self._apply_expr(
            volatility.atr(high=high, low=low, close=close, period=period),
            alias_name,
            group_by=group_by,
        )

    def true_range(
        self,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """True Range (TR)."""
        alias_name = alias or "true_range"
        return self._apply_expr(
            volatility.true_range(high=high, low=low, close=close),
            alias_name,
            group_by=group_by,
        )

    def stddev(
        self,
        period: int = 20,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Rolling Standard Deviation."""
        alias_name = alias or f"stddev_{period}"
        return self._apply_expr(
            volatility.stddev(col, period=period), alias_name, group_by=group_by
        )

    # =========================================================================
    # Volume Indicators
    # =========================================================================

    def obv(
        self,
        close: IntoExpr = "close",
        volume_col: IntoExpr = "volume",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """On-Balance Volume (OBV)."""
        alias_name = alias or "obv"
        return self._apply_expr(
            volume.obv(close=close, volume=volume_col), alias_name, group_by=group_by
        )

    def vwap(
        self,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        volume_col: IntoExpr = "volume",
        period: int | None = None,
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Volume-Weighted Average Price (VWAP)."""
        alias_name = alias or "vwap"
        return self._apply_expr(
            volume.vwap(high=high, low=low, close=close, volume=volume_col, period=period),
            alias_name,
            group_by=group_by,
        )

    def vwap_bands(
        self,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        volume_col: IntoExpr = "volume",
        period: int | None = None,
        mult1: float = 1.0,
        mult2: float = 2.0,
        mult3: float = 3.0,
        alias: str = "vwap_bands",
        unnest: bool = False,
        prefix: str | None = "vwap_",
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Volume-Weighted Average Price with Standard Deviation Bands."""
        expr = volume.vwap_bands(
            high=high,
            low=low,
            close=close,
            volume=volume_col,
            period=period,
            mult1=mult1,
            mult2=mult2,
            mult3=mult3,
        )
        return self._apply_expr(expr, alias=alias, unnest=unnest, prefix=prefix, group_by=group_by)

    def cmf(
        self,
        period: int = 20,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        volume_col: IntoExpr = "volume",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Chaikin Money Flow (CMF)."""
        alias_name = alias or f"cmf_{period}"
        return self._apply_expr(
            volume.cmf(high=high, low=low, close=close, volume=volume_col, period=period),
            alias_name,
            group_by=group_by,
        )

    def mfi(
        self,
        period: int = 14,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        volume_col: IntoExpr = "volume",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Money Flow Index (MFI)."""
        alias_name = alias or f"mfi_{period}"
        return self._apply_expr(
            volume.mfi(high=high, low=low, close=close, volume=volume_col, period=period),
            alias_name,
            group_by=group_by,
        )

    # =========================================================================
    # Stats & Alpha Indicators
    # =========================================================================

    def zscore(
        self,
        period: int = 20,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Rolling Z-Score."""
        alias_name = alias or f"zscore_{period}"
        return self._apply_expr(stats.zscore(col, period=period), alias_name, group_by=group_by)

    def percentile_rank(
        self,
        period: int = 20,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Rolling Percentile Rank."""
        alias_name = alias or f"percentile_rank_{period}"
        return self._apply_expr(
            stats.percentile_rank(col, period=period), alias_name, group_by=group_by
        )

    def ts_rank(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rank."""
        alias_name = alias or f"ts_rank_{period}"
        return self._apply_expr(alpha.ts_rank(col, period=period), alias_name, group_by=group_by)

    def fractal_dimension(
        self,
        period: int = 20,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Ehlers' Fractal Dimension."""
        alias_name = alias or f"fractal_dimension_{period}"
        return self._apply_expr(
            alpha.fractal_dimension(high=high, low=low, period=period),
            alias_name,
            group_by=group_by,
        )

    def hurst_exponent(
        self,
        period: int = 50,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Rolling Hurst Exponent."""
        alias_name = alias or f"hurst_{period}"
        return self._apply_expr(
            alpha.hurst_exponent(close=col, period=period),
            alias_name,
            group_by=group_by,
        )

    def cs_rank(
        self,
        col: IntoExpr = "close",
        over: IntoExpr | Sequence[str] | None = None,
        method: Literal["average", "min", "max", "dense", "ordinal", "random"] = "average",
        percentile: bool = True,
        alias: str | None = None,
    ) -> FrameType:
        """Cross-Sectional Rank (WorldQuant Alpha primitive 'rank(x)')."""
        alias_name = alias or "cs_rank"
        return self._apply_expr(
            alpha.cs_rank(col, over=over, method=method, percentile=percentile),
            alias_name,
        )

    def neutralize(
        self,
        col: IntoExpr = "close",
        by: IntoExpr | Sequence[str] = "sector",
        over: IntoExpr | Sequence[str] | None = None,
        standardize: bool = False,
        alias: str | None = None,
    ) -> FrameType:
        """Group / Industry Neutralization (WorldQuant Alpha primitive 'indneutralize(x, g)')."""
        alias_name = alias or "neutralized"
        return self._apply_expr(
            alpha.neutralize(col, by=by, over=over, standardize=standardize),
            alias_name,
        )

    def ts_scale(
        self,
        col: IntoExpr = "close",
        a: float = 1.0,
        over: IntoExpr | Sequence[str] | None = None,
        alias: str | None = None,
    ) -> FrameType:
        """Scale Primitive (WorldQuant Alpha primitive 'scale(x, a)')."""
        alias_name = alias or "scaled"
        return self._apply_expr(
            alpha.ts_scale(col, a=a, over=over),
            alias_name,
        )

    def ts_delta(
        self,
        period: int = 1,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Delta (WorldQuant Alpha primitive 'delta(x, d)')."""
        alias_name = alias or f"delta_{period}"
        return self._apply_expr(
            alpha.ts_delta(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_delay(
        self,
        period: int = 1,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Delay (WorldQuant Alpha primitive 'delay(x, d)')."""
        alias_name = alias or f"delay_{period}"
        return self._apply_expr(
            alpha.ts_delay(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def signed_power(
        self,
        exponent: float = 2.0,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Signed Power (WorldQuant Alpha primitive 'signed_power(x, a)')."""
        alias_name = alias or f"signed_power_{exponent}"
        return self._apply_expr(
            alpha.signed_power(col, exponent=exponent),
            alias_name,
            group_by=group_by,
        )

    def ts_min(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rolling Minimum (WorldQuant Alpha primitive 'ts_min(x, d)')."""
        alias_name = alias or f"ts_min_{period}"
        return self._apply_expr(
            alpha.ts_min(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_max(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rolling Maximum (WorldQuant Alpha primitive 'ts_max(x, d)')."""
        alias_name = alias or f"ts_max_{period}"
        return self._apply_expr(
            alpha.ts_max(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_sum(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rolling Sum (WorldQuant Alpha primitive 'ts_sum(x, d)')."""
        alias_name = alias or f"ts_sum_{period}"
        return self._apply_expr(
            alpha.ts_sum(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_std(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rolling Standard Deviation (WorldQuant Alpha primitive 'ts_std(x, d)')."""
        alias_name = alias or f"ts_std_{period}"
        return self._apply_expr(
            alpha.ts_std(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_correlation(
        self,
        other: IntoExpr,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Correlation (WorldQuant Alpha primitive 'correlation(x, y, d)')."""
        alias_name = alias or f"ts_corr_{period}"
        return self._apply_expr(
            alpha.ts_correlation(col, other, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_covariance(
        self,
        other: IntoExpr,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Covariance (WorldQuant Alpha primitive 'covariance(x, y, d)')."""
        alias_name = alias or f"ts_cov_{period}"
        return self._apply_expr(
            alpha.ts_covariance(col, other, period=period),
            alias_name,
            group_by=group_by,
        )

    def ts_product(
        self,
        period: int = 10,
        col: IntoExpr = "close",
        alias: str | None = None,
        group_by: str | Sequence[str] | None = None,
    ) -> FrameType:
        """Time-series Rolling Product (WorldQuant Alpha primitive 'product(x, d)')."""
        alias_name = alias or f"ts_prod_{period}"
        return self._apply_expr(
            alpha.ts_product(col, period=period),
            alias_name,
            group_by=group_by,
        )

    def alpha_001(
        self,
        close: IntoExpr = "close",
        returns: IntoExpr | None = None,
        over: IntoExpr | Sequence[str] | None = None,
        alias: str = "alpha_001",
    ) -> FrameType:
        """WorldQuant Formulaic Alpha #1."""
        return self._apply_expr(
            alpha.alpha_001(close=close, returns=returns, over=over),
            alias,
        )

    def alpha_041(
        self,
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        volume: IntoExpr = "volume",
        alias: str = "alpha_041",
    ) -> FrameType:
        """WorldQuant Formulaic Alpha #41."""
        return self._apply_expr(
            alpha.alpha_041(high=high, low=low, close=close, volume=volume),
            alias,
        )

    def alpha_054(
        self,
        open_: IntoExpr = "open",
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "alpha_054",
    ) -> FrameType:
        """WorldQuant Formulaic Alpha #54."""
        return self._apply_expr(
            alpha.alpha_054(open_=open_, high=high, low=low, close=close),
            alias,
        )

    def alpha_101(
        self,
        open_: IntoExpr = "open",
        high: IntoExpr = "high",
        low: IntoExpr = "low",
        close: IntoExpr = "close",
        alias: str = "alpha_101",
    ) -> FrameType:
        """WorldQuant Formulaic Alpha #101."""
        return self._apply_expr(
            alpha.alpha_101(open_=open_, high=high, low=low, close=close),
            alias,
        )
