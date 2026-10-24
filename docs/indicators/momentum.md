# Momentum Oscillators (20 Indicators)

Momentum indicators measure the velocity, acceleration, and overbought/oversold extremes of price movements.

---

### `rsi(close, period=14)`
Relative Strength Index using Wilder's smoothed moving averages. Oscillates between 0 and 100.

### `macd(close, fast=12, slow=26, signal=9)`
Moving Average Convergence Divergence. Returns Struct: `macd`, `signal`, `histogram`.

### `stochastic(high, low, close, k=14, d=3)`
Stochastic Oscillator. Returns Struct: `%K`, `%D`.

### `stoch_rsi(close, rsi_period=14, stoch_period=14)`
Stochastic RSI. Applies the Stochastic formula to RSI values for heightened sensitivity. Returns Struct: `%K`, `%D`.

### `williams_r(high, low, close, period=14)`
Williams %R. Reflects the level of close relative to the highest high over the lookback period.

### `cci(high, low, close, period=20)`
Commodity Channel Index. Identifies cyclical turns by measuring deviations from statistical mean price.

### `roc(close, period=10)`
Rate of Change. Measures percentage price change: $(P_t - P_{t-n}) / P_{t-n} \times 100$.

### `awesome_oscillator(high, low, fast=5, slow=34)`
Bill Williams' Awesome Oscillator (AO). Difference between 5-period and 34-period SMAs of median price.

### `tsi(close, long_period=25, short_period=13)`
True Strength Index. Double-smoothed momentum indicator that filters noise.

### `cmo(close, period=14)`
Chande Momentum Oscillator. Calculates momentum on both up and down days, bounded between $-100$ and $+100$.

### `ppo(close, fast=12, slow=26, signal=9)`
Percentage Price Oscillator. MACD expressed in percentage terms. Returns Struct: `ppo`, `signal`, `histogram`.

### `coppock(close, wma_period=10, fast_roc=11, slow_roc=14)`
Coppock Curve. Long-term momentum oscillator designed for spotting major market bottoms.

### `ultimate_oscillator(high, low, close, p1=7, p2=14, p3=28)`
Larry Williams' Ultimate Oscillator (UO). Combines momentum across three different timeframes.

### `fisher_transform(high, low, period=10)`
Ehlers Fisher Transform. Converts price into a Gaussian normal distribution to pinpoint turning points. Returns Struct: `fisher`, `signal`.

### `rvi(open, high, low, close, period=10)`
Relative Vigor Index. Compares bar closing price relative to bar range. Returns Struct: `rvi`, `signal`.

### `kst(close)`
"Know Sure Thing" Oscillator by Martin Pring. Sum of four smoothed rate-of-change indicators. Returns Struct: `kst`, `signal`.

### `dpo(close, period=20)`
Detrended Price Oscillator. Removes longer-term trends to highlight short-term cycles.

### `bop(open, high, low, close)`
Balance of Market Power: $(Close - Open) / (High - Low)$.

### `cfo(close, period=14)`
Chande Forecast Oscillator. Percentage difference between actual price and linear regression forecast.

### `squeeze_momentum(high, low, close, bb_period=20, bb_mult=2.0, kc_period=20, kc_mult=1.5, mom_period=None)`
John Carter's Squeeze Momentum Indicator (popularized by LazyBear). Identifies compression consolidations and explosive breakouts using Bollinger Bands and Keltner Channels, accompanied by an OLS linear regression momentum histogram. Returns Struct: `squeeze_on`, `squeeze_off`, `no_squeeze`, `momentum`.
