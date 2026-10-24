# Trend Indicators & Overlays (26 Indicators)

Trend indicators identify the direction and momentum of price movements, smooth market noise, and generate trailing stop and support/resistance levels.

---

## Moving Averages

### `sma(close, period=10)`
Simple Moving Average. Arithmetic mean over a sliding lookback window.

### `ema(close, period=10)`
Exponential Moving Average. Weights recent prices exponentially higher.

### `wma(close, period=10)`
Linearly-Weighted Moving Average. Weights linearly increase up to the current bar.

### `hma(close, period=10)`
Hull Moving Average. Extremely responsive moving average with virtually zero lag.

### `dema(close, period=10)`
Double Exponential Moving Average: $2 \cdot \text{EMA} - \text{EMA}(\text{EMA})$.

### `tema(close, period=10)`
Triple Exponential Moving Average: $3 \cdot \text{EMA} - 3 \cdot \text{EMA}(\text{EMA}) + \text{EMA}(\text{EMA}(\text{EMA}))$.

### `zlema(close, period=10)`
Zero-Lag Exponential Moving Average. Uses lagged differences to eliminate lag.

### `t3(close, period=5, vfactor=0.7)`
Tillson T3 Moving Average. Smooth multi-stage exponential filter.

### `alma(close, period=9, offset=0.85, sigma=6.0)`
Arnaud Legoux Moving Average. Gaussian-distribution weighted moving average.

### `kama(close, period=10, fast_span=2, slow_span=30)`
Kaufman's Adaptive Moving Average. Dynamically adjusts smoothing based on the Efficiency Ratio (market volatility).

### `mcginley_dynamic(close, period=14, k=0.6)`
McGinley Dynamic Average. Self-adjusting tracking indicator designed to prevent whipsaws.

### `vidya(close, period=9, cmo_period=9)`
Variable Index Dynamic Average (Chande's VIDYA). Scales EMA smoothing by the Chande Momentum Oscillator.

---

## Directional Movement & Trailing Stops

### `adx(high, low, close, period=14)`
Average Directional Movement Index. Returns Struct: `adx`, `plus_di`, `minus_di`.

### `vortex(high, low, close, period=14)`
Vortex Indicator. Returns Struct: `vi_plus`, `vi_minus`.

### `aroon(high, low, period=25)`
Aroon Indicator & Oscillator. Returns Struct: `up`, `down`, `oscillator`.

### `chop(high, low, close, period=14)`
Choppiness Index. Quantifies whether the market is trending ($< 38.2$) or consolidating ($> 61.8$).

### `mass_index(high, low, fast_period=9, slow_period=25)`
Mass Index. Predicts trend reversals by analyzing the expansion and contraction of price range.

### `supertrend(high, low, close, period=10, multiplier=3.0)`
SuperTrend Volatility Stop. Returns Struct: `supertrend`, `direction` ($1$ for bull, $-1$ for bear).

### `psar(high, low, af_step=0.02, af_max=0.2)`
Parabolic Stop and Reverse (Trailing SAR).

### `ssl_channel(high, low, close, period=10)`
SSL Channel (Securement Stop Loss Channel). Returns Struct: `ssl_up`, `ssl_down`, `hlv` ($1$ for bull, $-1$ for bear).

### `center_of_gravity(close, period=10)`
John Ehlers' Center of Gravity Oscillator. Detects cycle turning points with zero lag.

---

## Cloud & Floor Trader Pivots

### `ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52, displacement=26)`
Ichimoku Kinko Hyo Cloud. Returns Struct: `tenkan_sen`, `kijun_sen`, `senkou_span_a`, `senkou_span_b`, `chikou_span`.

### `pivot_points(high, low, close)`
Standard Floor Trader Pivot Points. Returns Struct: `pivot`, `r1`, `s1`, `r2`, `s2`, `r3`, `s3`.

### `fibonacci_pivots(high, low, close)`
Fibonacci Pivot Points (38.2%, 61.8%, 100% ratios). Returns Struct: `pivot`, `r1-r3`, `s1-s3`.

### `camarilla_pivots(high, low, close)`
Camarilla Intraday Pivot Points (8 tight levels). Returns Struct: `r1-r4`, `s1-s4`.

### `woodie_pivots(open, high, low, close)`
Woodie's Pivot Points. Weights closing price more heavily: $(H + L + 2C) / 4$. Returns Struct: `pivot`, `r1`, `s1`, `r2`, `s2`.
