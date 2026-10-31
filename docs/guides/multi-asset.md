# Parallel Multi-Asset & Cross-Sectional Scaling with `.over()`

In institutional quantitative research, a pipeline operates simultaneously along **two orthogonal axes** of a panel dataset $(i \in \{1, \dots, N_{\text{assets}}\},\; t \in \{1, \dots, T_{\text{bars}}\})$:

1. **Time-Series Axis (`.over("symbol")`)**: Rolling moving averages, fractional differentiation (`frac_diff`), volatility estimators (`garman_klass_volatility`), and path-dependent labels (`triple_barrier_labels`) computed independently per instrument across time.
2. **Cross-Sectional Axis (`.over("timestamp")`)**: Winsorization (`cs_winsorize`), standardization (`cs_zscore`), sector neutralization (`neutralize`), and multi-factor OLS orthogonalization (`orthogonalize`) computed across all instruments at each point in time.

---

## Combining Time-Series & Cross-Sectional Expressions in One Query

Because every `polars-anta` indicator is a native `pl.Expr`, Polars' query optimizer fuses both time-series `.over("symbol")` and cross-sectional `.over("timestamp")` operations into a single parallel execution graph:

```python
import polars as pl
import polars_anta as pa

panel_alpha = (
    universe_df.with_columns(
        [
            # Step 1: Time-series features partitioned per asset (.over("symbol"))
            pl.col("close").anta.frac_diff(d=0.4, window=20).over("symbol").alias("ffd_04"),
            pa.garman_klass_volatility("open", "high", "low", "close", period=20)
            .over("symbol")
            .alias("gk_vol"),
            pa.amihud_illiquidity("close", "volume", period=20)
            .over("symbol")
            .alias("amihud"),
            pl.col("close").anta.roc(10).over("symbol").alias("mom_10"),
        ]
    )
    .with_columns(
        [
            # Step 2: Cross-sectional outlier clipping & standardization per timestamp
            pl.col("mom_10")
            .anta.cs_winsorize(over="timestamp", method="mad", threshold=3.0)
            .anta.cs_zscore(over="timestamp")
            .alias("mom_z"),
        ]
    )
    .with_columns(
        [
            # Step 3: Cross-sectional Gram-Schmidt orthogonalization against risk factors
            pl.col("mom_z")
            .anta.orthogonalize(
                factors=["gk_vol", "amihud"],
                over="timestamp",
                standardize=True,
            )
            .alias("ortho_alpha"),
        ]
    )
)
```

---

## Why `.over()` Outperforms Pandas `groupby` by 8x–112x

1. **Rayon Work-Stealing Thread Pool**: Polars partitions the hash table of `"symbol"` or `"timestamp"` and distributes disjoint index slices across all CPU cores in Rust without acquiring Python's GIL.
2. **Cache-Local Arrow Slices**: Group operations read contiguous 64-byte aligned Arrow arrays rather than allocating a new Python `pd.DataFrame` object per ticker.
3. **Expression Graph Fusion**: Multiple `.over("symbol")` expressions inside the same `.with_columns([...])` block share the same group partition index map, computing the partition hash only **once**.
