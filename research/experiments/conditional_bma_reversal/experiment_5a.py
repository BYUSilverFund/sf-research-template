import datetime as dt
import os
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel, run_conditional_bma_loop

load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "amihud_conditional_bma"
signal_name_title = "Amihud Conditional BMA"
IC = 0.05
gamma = 115
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_5")

# BMA parameters
dynamic_window_months = 12
checkpoint_dir = "temp/checkpoints_amihud_bma"

results_folder.mkdir(parents=True, exist_ok=True)
reversal_factors = ["rev_5d", "rev_10d", "rev_20d"]

# Data Ingestion
returns = sfd.load_assets(
    start=start, 
    end=end,
    columns=[
        "date", "barrid", "ticker", "price", "market_cap", "daily_volume",
        "return", "specific_return", "specific_risk", "predicted_beta"
    ],
    in_universe=True,
).with_columns([
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
])

# Feature Engineering
data = returns.sort("date", "barrid").with_columns([
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(5).over("barrid")).sub(1)).alias("rev_5d"),
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(10).over("barrid")).sub(1)).alias("rev_10d"),
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(20).over("barrid")).sub(1)).alias("rev_20d"),
    pl.col("price").sort_by("date").shift(1).over("barrid").alias("prev_price"),
    
    # Calculate Daily Amihud = |Return| / (Price * Volume)
    (pl.col("return").abs().truediv(pl.col("price").mul(pl.col("daily_volume")).add(1))).alias("daily_amihud")
])

# Smooth Amihud with a 20-day rolling median to capture the structural liquidity regime
data = data.with_columns([
    pl.col("daily_amihud").sort_by("date").rolling_median(window_size=20).over("barrid").alias("amihud_illiquidity")
])

# Cross-Sectional Scoring
tradable_universe = data.filter(pl.col("prev_price").gt(price_filter))

scored_factors = (
    tradable_universe.drop_nulls("amihud_illiquidity")
    .with_columns(
        pl.col("amihud_illiquidity")
        .qcut(5, labels=["1_Most_Liquid", "2_Liquid", "3_Mid", "4_Illiquid", "5_Most_Illiquid"], allow_duplicates=True)
        .over("date")
        .cast(pl.String)
        # Alias to "size_bucket" to maintain compatibility with the BMA utility function
        .alias("size_bucket") 
    )
    .select(["date", "barrid", "size_bucket"] + reversal_factors)
    .drop_nulls()
    .with_columns([
        (pl.col(f).sub(pl.col(f).mean().over("date")).truediv(pl.col(f).std().over("date"))).mul(-1).alias(f)
        for f in reversal_factors
    ])
)

# Alignment & Lagging
data = data.drop(reversal_factors).join(scored_factors, on=["date", "barrid"], how="left")

data = data.with_columns([
    pl.col(f).sort_by("date").shift(1).over("barrid").alias(f"{f}_lag")
    for f in reversal_factors
] + [
    pl.col("size_bucket").sort_by("date").shift(1).over("barrid").alias("size_bucket_lag")
])

# BMA Pipeline
trade_data = data.drop_nulls(subset=[f"{f}_lag" for f in reversal_factors] + ["size_bucket_lag"])

daily_factor_returns = trade_data.group_by(["date", pl.col("size_bucket_lag").alias("size_bucket")]).agg([
    (pl.col(f"{f}_lag").mul(pl.col("return"))).mean().alias(f) 
    for f in reversal_factors
]).sort("date")

m_fac_rets = daily_factor_returns.sort("date").group_by_dynamic("date", every="1mo", group_by="size_bucket").agg([
    (((pl.col(f).add(1)).product().sub(1)) * 100).alias(f) 
    for f in reversal_factors
])

timing_ready_df = m_fac_rets.with_columns([
    pl.col(f).shift(-1).over("size_bucket").alias(f"{f}_target")
    for f in reversal_factors
]).drop_nulls()

unique_dates = timing_ready_df.select("date").unique().sort("date").to_series().to_list()
unique_buckets = timing_ready_df.select("size_bucket").unique().sort("size_bucket").to_series().to_list()

# Standard BMA Loop (No priors, no vol targets)
rolling_weights_df = run_conditional_bma_loop(
    timing_ready_df=timing_ready_df, 
    reversal_factors=reversal_factors,
    unique_dates=unique_dates, 
    unique_buckets=unique_buckets,
    dynamic_window_months=dynamic_window_months, 
    checkpoint_dir=checkpoint_dir
).rename({f: f + "_beta" for f in reversal_factors})

# Final Signal Generation
trade_data = trade_data.with_columns(pl.col("date").dt.month_start().alias("month_key"))
weights_with_key = rolling_weights_df.with_columns(pl.col("date").dt.offset_by("1mo").dt.month_start().alias("month_key")).drop("date")

trade_data = trade_data.join(weights_with_key, left_on=["month_key", "size_bucket_lag"], right_on=["month_key", "size_bucket"], how="inner")

alphas = (
    trade_data.with_columns([
        pl.sum_horizontal([pl.col(f).mul(pl.col(f + "_beta")) for f in reversal_factors]).alias(signal_name)
    ])
    .filter(pl.col(signal_name).is_not_null())
    .with_columns(
        ((pl.col(signal_name).sub(pl.col(signal_name).mean().over("date")))
         .truediv(pl.col(signal_name).std().over("date"))
         .mul(IC).mul("specific_risk")).alias("alpha")
    )
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

forward_returns = (
    returns.sort("date", "barrid")
    .select("date", "barrid", pl.col("return").sort_by("date").shift(-1).over("barrid").alias("return"))
    .drop_nulls("return")
)

ics = sfp.generate_alpha_ics(alphas=alphas, rets=forward_returns, method="rank", window=22)

sfp.generate_ic_chart(
    ics=ics, 
    title=f"{signal_name_title} Cumulative IC", 
    ic_type="Rank", 
    file_name=results_folder / "rank_ic_chart.png"
)
sfp.generate_ic_chart(
    ics=ics, 
    title=f"{signal_name_title} Cumulative IC", 
    ic_type="Pearson", 
    file_name=results_folder / "pearson_ic_chart.png"
)

run_backtest_parallel(
    data=alphas, 
    signal_name=signal_name, 
    constraints=constraints, 
    gamma=gamma, 
    n_cpus=n_cpus
)