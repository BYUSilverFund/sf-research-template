import datetime as dt
import os
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel, run_conditional_bma_loop

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "adaptive_inhibited_bma_reversal"
signal_name_title = "Adaptive Inhibited BMA Reversal"
IC = 0.05
gamma = 115
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_4")

# BMA parameters
dynamic_window_months = 12
checkpoint_dir = "temp/checkpoints_adaptive_inhibited_bma"

# Hierarchical Shrinkage Priors
hierarchical_priors = {
    "1_Micro": 0.85,  
    "2_Small": 0.65,
    "3_Mid": 0.45,
    "4_Large": 0.25,
    "5_Mega": 0.10    
}

# Adaptive Volume Inhibition Thresholds
inhibition_thresholds = {
    "1_Micro": 1.2,  
    "2_Small": 1.5,
    "3_Mid": 1.8,
    "4_Large": 2.2,
    "5_Mega": 3.0    
}
vol_slope = 3.0 

# Create folders
results_folder.mkdir(parents=True, exist_ok=True)

# Define custom reversal factors
reversal_factors = ["rev_5d", "rev_10d", "rev_20d"]

# Load continuous asset data
returns = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "market_cap",
        "daily_volume", 
        "return",
        "specific_return",
        "specific_risk",
        "predicted_beta",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Compute trailing returns and relative volume on the continuous dataset
data = returns.sort("date", "barrid").with_columns(
    [
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(5).over("barrid")).sub(1)).alias("rev_5d"),
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(10).over("barrid")).sub(1)).alias("rev_10d"),
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(20).over("barrid")).sub(1)).alias("rev_20d"),
        pl.col("price").sort_by("date").shift(1).over("barrid").alias("prev_price"),
        (
            pl.col("daily_volume")
            .add(1)
            .log()
            .sub(
                pl.col("daily_volume")
                .add(1)
                .log()
                .sort_by("date")
                .rolling_median(window_size=20)
                .over("barrid")
            )
        ).exp().alias("rel_vol")
    ]
)

# Isolate the tradable universe to calculate quintiles and z-scores
tradable_universe = data.filter(
    pl.col("prev_price").gt(price_filter)
)

# Compute size buckets and z-scores only on tradable stocks
scored_factors = (
    tradable_universe.with_columns(
        pl.col("market_cap")
        .qcut(5, labels=["1_Micro", "2_Small", "3_Mid", "4_Large", "5_Mega"], allow_duplicates=True)
        .cast(pl.String)
        .over("date")
        .alias("size_bucket")
    )
    .select(["date", "barrid", "size_bucket", "rel_vol"] + reversal_factors)
    .drop_nulls()
    .with_columns(
        [
            (
                (pl.col(f).sub(pl.col(f).mean().over("date"))).truediv(pl.col(f).std().over("date"))
            ).mul(-1).alias(f)
            for f in reversal_factors
        ]
    )
)

# Join the scored factors back to the continuous dataset
data = data.drop(reversal_factors + ["rel_vol"]).join(
    scored_factors, 
    on=["date", "barrid"], 
    how="left"
)

# Compute the lagged exposures on the continuous dataset
data = data.with_columns(
    [
        pl.col(f).sort_by("date").shift(1).over("barrid").alias(f"{f}_lag")
        for f in reversal_factors
    ] + [
        pl.col("size_bucket").sort_by("date").shift(1).over("barrid").alias("size_bucket_lag"),
        pl.col("rel_vol").sort_by("date").shift(1).over("barrid").alias("rel_vol_lag")
    ]
)

# Filter out untradable rows for factor return calculation
trade_data = data.drop_nulls(
    subset=[f"{f}_lag" for f in reversal_factors] + ["size_bucket_lag"]
)

# Create daily factor returns grouped by date and lagged size bucket
daily_factor_returns = trade_data.group_by(
    ["date", pl.col("size_bucket_lag").alias("size_bucket")]
).agg(
    [
        (pl.col(f"{f}_lag").mul(pl.col("return"))).mean().alias(f)
        for f in reversal_factors
    ]
).sort("date")

# Compound daily to monthly factor returns
m_fac_rets = (
    daily_factor_returns.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="size_bucket")
    .agg(
        [
            (((pl.col(f).add(1)).product().sub(1)) * 100).alias(f)
            for f in reversal_factors
        ]
    )
)

# Volatility Adjustment for BMA Targets
m_fac_rets = m_fac_rets.sort("date", "size_bucket").with_columns(
    [
        pl.col(f)
        .rolling_std(window_size=12, min_periods=3)
        .over("size_bucket")
        .fill_null(strategy="backward")
        .clip(lower_bound=0.01) 
        .alias(f"{f}_vol")
        for f in reversal_factors
    ]
)

# Build timing ready dataframe
timing_ready_df = m_fac_rets.with_columns(
    [
        (pl.col(f).shift(-1).truediv(pl.col(f"{f}_vol"))).over("size_bucket").alias(f"{f}_target") 
        for f in reversal_factors
    ]
).drop_nulls()

unique_dates = (
    timing_ready_df.select("date")
    .unique()
    .sort("date")
    .to_series()
    .to_list()
)

unique_buckets = (
    timing_ready_df.select("size_bucket")
    .unique()
    .sort("size_bucket")
    .to_series()
    .to_list()
)

# Run the Conditional BMA Loop
rolling_weights_df = run_conditional_bma_loop(
    timing_ready_df=timing_ready_df,
    reversal_factors=reversal_factors,
    unique_dates=unique_dates,
    unique_buckets=unique_buckets,
    dynamic_window_months=dynamic_window_months,
    checkpoint_dir=checkpoint_dir,
    null_priors=hierarchical_priors
).rename(
    {f: f + "_beta" for f in reversal_factors}
)

# Align BMA weights to trade data
trade_data = trade_data.with_columns(
    pl.col("date").dt.month_start().alias("month_key")
)

weights_with_key = rolling_weights_df.with_columns(
    pl.col("date").dt.offset_by("1mo").dt.month_start().alias("month_key")
).drop("date")

trade_data = trade_data.join(
    weights_with_key, 
    left_on=["month_key", "size_bucket_lag"], 
    right_on=["month_key", "size_bucket"],
    how="inner"
)

# Map volume inhibition thresholds based on size bucket
trade_data = trade_data.with_columns(
    pl.col("size_bucket_lag")
    .replace_strict(inhibition_thresholds, default=None)
    .cast(pl.Float64)
    .alias("vol_limit")
)

# Compute BMA signal, apply volume inhibition, and compute final alphas
alphas = (
    trade_data.with_columns(
        pl.sum_horizontal(
            [pl.col(f).mul(pl.col(f + "_beta")) for f in reversal_factors]
        ).alias("raw_signal")
    )
    .with_columns(
        (
            pl.col("raw_signal") * (1.0 / (1.0 + (pl.col("rel_vol_lag") / pl.col("vol_limit")).pow(vol_slope)))
        ).alias("inhibited_signal")
    )
    .filter(
        pl.col("inhibited_signal").is_not_null(),
        pl.col("predicted_beta").is_not_null(),
        pl.col("specific_risk").is_not_null()
    )
    .with_columns(
        (
            (pl.col("inhibited_signal").sub(pl.col("inhibited_signal").mean().over("date")))
            .truediv(pl.col("inhibited_signal").std().over("date"))
            .mul(IC)
            .mul("specific_risk")
        ).alias("alpha")
    )
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Get forward returns on the unfiltered contiguous dataset
forward_returns = (
    returns.sort("date", "barrid")
    .select(
        "date", 
        "barrid", 
        pl.col("return").sort_by("date").shift(-1).over("barrid").alias("return")
    )
    .drop_nulls("return")
)

# Get ics
ics = sfp.generate_alpha_ics(
    alphas=alphas, 
    rets=forward_returns, 
    method="rank", 
    window=22
)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"

sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)

sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name_title} Cumulative IC",
    ic_type="Pearson",
    file_name=pearson_chart_path,
)

# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)