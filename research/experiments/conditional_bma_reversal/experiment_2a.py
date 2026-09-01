import datetime as dt
import os
from pathlib import Path
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

# Import from your centralized utils
from research.utils import run_backtest_parallel, run_conditional_bma_loop

load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "hierarchical_bma_reversal"
signal_name_title = "Hierarchical BMA Reversal"
IC = 0.05
gamma = 115 
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_2")

# BMA parameters
dynamic_window_months = 12
checkpoint_dir = "temp/checkpoints_hierarchical_bma"

hierarchical_priors = {
    "1_Micro": 0.85,  # 85% prior belief that signal is noise
    "2_Small": 0.65,
    "3_Mid": 0.45,
    "4_Large": 0.25,
    "5_Mega": 0.10    # 10% prior belief that signal is noise
}

# Folder setup
results_folder.mkdir(parents=True, exist_ok=True)

# Define factors
reversal_factors = ["rev_5d", "rev_10d", "rev_20d"]

# Load data
returns = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date", "barrid", "ticker", "price", 
        "market_cap", "return", "specific_return", 
        "specific_risk", "predicted_beta",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Feature Engineering
data = returns.sort("date", "barrid").with_columns([
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(5).over("barrid")).sub(1)).alias("rev_5d"),
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(10).over("barrid")).sub(1)).alias("rev_10d"),
    (pl.col("price").truediv(pl.col("price").sort_by("date").shift(20).over("barrid")).sub(1)).alias("rev_20d"),
    pl.col("price").sort_by("date").shift(1).over("barrid").alias("prev_price")
])

# Cross-Sectional Scoring
tradable_universe = data.filter(pl.col("prev_price").gt(price_filter))

scored_factors = (
    tradable_universe.with_columns(
        pl.col("market_cap")
        .qcut(5, labels=["1_Micro", "2_Small", "3_Mid", "4_Large", "5_Mega"], allow_duplicates=True)
        .over("date")
        .cast(pl.String)
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

# Prepare BMA Training Data
trade_data = data.drop_nulls(subset=[f"{f}_lag" for f in reversal_factors] + ["size_bucket_lag"])

daily_factor_returns = trade_data.group_by(["date", pl.col("size_bucket_lag").alias("size_bucket")]).agg([
    (pl.col(f"{f}_lag").mul(pl.col("return"))).mean().alias(f)
    for f in reversal_factors
]).sort("date")

m_fac_rets = (
    daily_factor_returns.sort("date")
    .group_by_dynamic("date", every="1mo", group_by="size_bucket")
    .agg([
        (((pl.col(f).add(1)).product().sub(1)) * 100).alias(f)
        for f in reversal_factors
    ])
)

timing_ready_df = m_fac_rets.with_columns([
    pl.col(f).shift(-1).over("size_bucket").alias(f"{f}_target") 
    for f in reversal_factors
]).drop_nulls()

unique_dates = timing_ready_df.select("date").unique().sort("date").to_series().to_list()
unique_buckets = timing_ready_df.select("size_bucket").unique().sort("size_bucket").to_series().to_list()

# Run the Hierarchical BMA Engine
rolling_weights_df = run_conditional_bma_loop(
    timing_ready_df=timing_ready_df,
    reversal_factors=reversal_factors,
    unique_dates=unique_dates,
    unique_buckets=unique_buckets,
    dynamic_window_months=dynamic_window_months,
    checkpoint_dir=checkpoint_dir,
    null_priors=hierarchical_priors
).rename({f: f + "_beta" for f in reversal_factors})

#Signal Generation & Backtest
trade_data = trade_data.with_columns(pl.col("date").dt.month_start().alias("month_key"))

weights_with_key = rolling_weights_df.with_columns(
    pl.col("date").dt.offset_by("1mo").dt.month_start().alias("month_key")
).drop("date")

trade_data = trade_data.join(
    weights_with_key, 
    left_on=["month_key", "size_bucket_lag"], 
    right_on=["month_key", "size_bucket"],
    how="inner"
)

alphas = (
    trade_data.with_columns(
        pl.sum_horizontal([pl.col(f).mul(pl.col(f + "_beta")) for f in reversal_factors]).alias(signal_name)
    )
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
    .select(
        "date", 
        "barrid", 
        pl.col("return").sort_by("date").shift(-1).over("barrid").alias("return")
    )
    .drop_nulls("return")
)

ics = sfp.generate_alpha_ics(
    alphas=alphas, 
    rets=forward_returns, 
    method="rank", 
    window=22
)

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

run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)