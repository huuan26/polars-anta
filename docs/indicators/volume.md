# Volume Indicators (15 Indicators)

Volume indicators evaluate institutional buying and selling pressure, money flow, and volume-price divergence.

---

### `obv(close, volume)`
On-Balance Volume. Cumulative volume added on up bars and subtracted on down bars.

### `vwap(high, low, close, volume, period=None)`
Volume Weighted Average Price. Computes cumulative VWAP (default) or rolling VWAP when `period` is specified.

### `vwap_bands(high, low, close, volume, period=None, mult1=1.0, mult2=2.0, mult3=3.0)`
Volume Weighted Average Price with Standard Deviation Bands. Returns Struct: `vwap`, `upper_1`, `lower_1`, `upper_2`, `lower_2`, `upper_3`, `lower_3`.

### `vwma(close, volume, period=20)`
Volume Weighted Moving Average. Moving average weighted by bar volume.

### `cmf(high, low, close, volume, period=20)`
Chaikin Money Flow. Quantifies money accumulation and distribution over a lookback window.

### `adl(high, low, close, volume)`
Accumulation/Distribution Line. Cumulative volume based on the Close Location Value (CLV).

### `mfi(high, low, close, volume, period=14)`
Money Flow Index. Volume-weighted RSI bounded between 0 and 100.

### `eom(high, low, volume, period=14)`
Ease of Movement (EMV). Relates price change to trading volume.

### `klinger(high, low, close, volume, fast=34, slow=55, signal=13)`
Klinger Volume Oscillator. Determines long-term money flow while remaining sensitive to short-term fluctuations. Returns Struct: `klinger`, `signal`.

### `nvi(close, volume, initial=1000.0)`
Negative Volume Index. Follows smart money by updating only when volume decreases.

### `pvi(close, volume, initial=1000.0)`
Positive Volume Index. Follows retail crowd activity by updating only when volume increases.

### `pvt(close, volume)`
Price Volume Trend. Cumulative volume scaled by percentage price changes.

### `force_index(close, volume, period=13)`
Elder's Force Index (EFI). Measures the raw price and volume force behind market moves.

### `vpci(close, volume, period_short=5, period_long=25)`
Volume Price Confirmation Indicator. Compares price-weighted volume against volume-weighted price.

### `vr(close, volume, period=26)`
Volume Ratio. Ratio of volume on rising bars vs volume on falling bars.
