"""Frozen pre-registered configuration for the Vietnamese Equity Factor Zoo study."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_CACHE_PATH = REPO_ROOT / "research" / "data" / "vn_daily.parquet"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class StudyConfig:
    """Immutable study configuration matching PREREGISTRATION.md."""

    # Market & sample boundaries
    exchange: str = "HOSE"
    warmup_start: date = date(2012, 1, 1)
    discovery_start: date = date(2014, 1, 1)
    discovery_end: date = date(2021, 12, 31)
    holdout_start: date = date(2022, 1, 1)

    # Point-in-time universe & data cleaning parameters
    hose_price_limit: float = 0.07
    price_limit_tolerance: float = 0.005
    min_history_bars: int = 250
    min_close_vnd: float = 5_000.0
    adv_window: int = 60
    top_n_liquid: int = 150

    # Target return & horizons (open-to-open: open_{t+1+h} / open_{t+1} - 1)
    primary_horizon: int = 5
    exploratory_horizons: tuple[int, ...] = (1, 20)
    ic_method: str = "spearman"

    # Statistical inference & multiple testing
    nw_lags: int = 4
    n_boot: int = 2000
    fdr_q: float = 0.10
    fwer_alpha: float = 0.05
    eb_prior: str = "zero"
    seed: int = 20261110

    # Transaction cost stage (long-only top quintile vs equal-weight universe)
    n_quantiles: int = 5
    rebalance_every: int = 5
    brokerage_fee_one_way: float = 0.0015  # 0.15% per side
    sell_tax_rate: float = 0.0010  # 0.10% on sell proceeds
    slippage_one_way: float = 0.0010  # 0.10% per side
    cost_alpha: float = 0.05

    @property
    def round_trip_cost_rate(self) -> float:
        """Total round-trip cost (buy + sell) per unit of 1-way turnover."""
        # A 1-unit weight replacement (sell 1 unit of old stock, buy 1 unit of new stock)
        # incurs buy brokerage + buy slippage + sell brokerage + sell tax + sell slippage.
        # Expressed per unit of L1 weight change |w_t - w_{t-1}| (where 1 full replacement = 2 L1 units):
        buy_side = self.brokerage_fee_one_way + self.slippage_one_way
        sell_side = self.brokerage_fee_one_way + self.sell_tax_rate + self.slippage_one_way
        return 0.5 * (buy_side + sell_side)


DEFAULT_CONFIG = StudyConfig()
