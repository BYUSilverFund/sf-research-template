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
signal_name = "master_filtered_bma_reversal"
signal_name_title = "Master Filtered Double Sort BMA"
IC = 0.05
gamma = 115
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_16")
checkpoint_dir = "temp/checkpoints_exp16_master_filtered"
dynamic_window_months = 12
vol_slope = 3.0

results_folder.mkdir(parents=True, exist_ok=True)
reversal_factors = ["rev_5d", "rev_10d", "rev_20d"]

# Matrix Generators for Priors and Thresholds
mcap_labels = ["1_Micro", "2_Small", "3_Mid", "4_Large", "5_Mega"]
amihud_labels = ["1_Liq", "2_SLiq", "3_Mid", "4_Illiq", "5_VIlliq"]

base_priors = {"1_Micro": 0.60, "2_Small": 0.45, "3_Mid": 0.30, "4_Large": 0.15, "5_Mega": 0.05}
illiq_penalty = {"1_Liq": -0.10, "2_SLiq": -0.05, "3_Mid": 0.00, "4_Illiq": 0.15, "5_VIlliq": 0.30}

base_thresholds = {"1_Micro": 1.2, "2_Small": 1.5, "3_Mid": 1.8, "4_Large": 2.2, "5_Mega": 3.0}
illiq_thresh_adj = {"1_Liq": 0.2, "2_SLiq": 0.1, "3_Mid": 0.0, "4_Illiq": -0.1, "5_VIlliq": -0.2}

hierarchical_priors = {}
inhibition_thresholds = {}

for m in mcap_labels:
    for a in amihud_labels:
        bucket = f"{m}_{a}"
        # Constrain priors between 5% and 95%
        hierarchical_priors[bucket] = min(max(base_priors[m] + illiq_penalty[a], 0.05), 0.95)
        # Constrain thresholds to be at least 1.0x
        inhibition_thresholds[bucket] = max(base_thresholds[m] + illiq_thresh_adj[a], 1.0)

# Data Ingestion
returns = (
    sfd.load_assets(
        start=start, 
        end=end,
        columns=[
            "date", "barrid", "ticker", "price", "market_cap", "daily_volume",
            "return", "specific_return", "specific_risk", "predicted_beta"
        ],
        in_universe=True,
    )
    .with_columns([
        pl.col("return").truediv(100),
        pl.col("specific_return").truediv(100),
        pl.col("specific_risk").truediv(100),
    ])
)

# Load exposures for Biolife extraction
biolife_col = "USSLOWL_BIOLIFE"

exposures = sfd.load_exposures(
    start=start, 
    end=end, 
    in_universe=True, 
    columns=["date", "barrid", biolife_col]
).fill_null(0.0)

# Select only date, barrid, and the specific Biolife dummy
biolife_mapping = (
    exposures
    .select([
        "date", 
        "barrid", 
        pl.col(biolife_col).cast(pl.Float64).alias("is_biolife")
    ])
)

# Feature Engineering
data = (
    returns
    .sort("date", "barrid")
    .with_columns([
        (
            pl.col("price")
            .truediv(
                pl.col("price")
                .sort_by("date")
                .shift(5)
                .over("barrid")
            )
            .sub(1)
        ).alias("rev_5d"),
        
        (
            pl.col("price")
            .truediv(
                pl.col("price")
                .sort_by("date")
                .shift(10)
                .over("barrid")
            )
            .sub(1)
        ).alias("rev_10d"),
        
        (
            pl.col("price")
            .truediv(
                pl.col("price")
                .sort_by("date")
                .shift(20)
                .over("barrid")
            )
            .sub(1)
        ).alias("rev_20d"),
        
        pl.col("price")
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias("prev_price"),
        
        # Log-Amihud for the Double Sort
        (
            pl.col("return")
            .abs()
            .add(1e-6)
            .log()
            .sub(
                (
                    pl.col("price")
                    .mul(pl.col("daily_volume"))
                    .add(1)
                ).log()
            )
        ).alias("daily_log_amihud"),
        
        # Relative Log Volume for the Inhibition Circuit Breaker
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
        )
        .exp()
        .alias("rel_vol")
    ])
)

# Smooth Log-Amihud
data = (
    data
    .with_columns([
        pl.col("daily_log_amihud")
        .sort_by("date")
        .rolling_median(window_size=20)
        .over("barrid")
        .alias("amihud_illiquidity")
    ])
)

# Cross-Sectional Double Sorting
tradable_universe = (
    data
    .filter(pl.col("prev_price").gt(price_filter))
)

scored_factors = (
    tradable_universe
    .drop_nulls(["amihud_illiquidity", "market_cap"])
    .with_columns([
        pl.col("market_cap")
        .qcut(5, labels=mcap_labels, allow_duplicates=True)
        .over("date")
        .cast(pl.String)
        .alias("mcap_q")
    ])
    .with_columns([
        pl.col("amihud_illiquidity")
        .qcut(5, labels=amihud_labels, allow_duplicates=True)
        .over(["date", "mcap_q"])
        .cast(pl.String)
        .alias("amihud_q")
    ])
    .with_columns([
        pl.concat_str([
            pl.col("mcap_q"), 
            pl.lit("_"), 
            pl.col("amihud_q")
        ]).alias("size_bucket")
    ])
    .select(["date", "barrid", "size_bucket", "rel_vol"] + reversal_factors)
    .drop_nulls()
    .with_columns([
        (
            pl.col(f)
            .sub(pl.col(f).mean().over("date"))
            .truediv(pl.col(f).std().over("date"))
        )
        .mul(-1)
        .alias(f)
        for f in reversal_factors
    ])
)

# Alignment & Lagging
data = (
    data
    .drop(reversal_factors + ["rel_vol"])
    .join(scored_factors, on=["date", "barrid"], how="left")
)

data = (
    data
    .with_columns([
        pl.col(f)
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias(f"{f}_lag")
        for f in reversal_factors
    ] + [
        pl.col("size_bucket")
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias("size_bucket_lag"),
        
        pl.col("rel_vol")
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias("rel_vol_lag")
    ])
)

# BMA Pipeline
trade_data = (
    data
    .drop_nulls(subset=[f"{f}_lag" for f in reversal_factors] + ["size_bucket_lag"])
)

daily_factor_returns = (
    trade_data
    .group_by(["date", pl.col("size_bucket_lag").alias("size_bucket")])
    .agg([
        (
            pl.col(f"{f}_lag")
            .mul(pl.col("return"))
        )
        .mean()
        .alias(f) 
        for f in reversal_factors
    ])
    .sort("date")
)

m_fac_rets = (
    daily_factor_returns
    .sort("date")
    .group_by_dynamic("date", every="1mo", group_by="size_bucket")
    .agg([
        (
            (
                pl.col(f)
                .add(1)
            )
            .product()
            .sub(1)
        )
        .mul(100)
        .alias(f) 
        for f in reversal_factors
    ])
)

timing_ready_df = (
    m_fac_rets
    .with_columns([
        pl.col(f)
        .shift(-1)
        .over("size_bucket")
        .alias(f"{f}_target") 
        for f in reversal_factors
    ])
    .drop_nulls()
)

unique_dates = (
    timing_ready_df
    .select("date")
    .unique()
    .sort("date")
    .to_series()
    .to_list()
)

unique_buckets = (
    timing_ready_df
    .select("size_bucket")
    .unique()
    .sort("size_bucket")
    .to_series()
    .to_list()
)

# Run the Conditional BMA Loop with Hierarchical Priors
rolling_weights_df = (
    run_conditional_bma_loop(
        timing_ready_df=timing_ready_df, 
        reversal_factors=reversal_factors,
        unique_dates=unique_dates, 
        unique_buckets=unique_buckets,
        dynamic_window_months=dynamic_window_months, 
        checkpoint_dir=checkpoint_dir,
        null_priors=hierarchical_priors
    )
    .rename({f: f + "_beta" for f in reversal_factors})
)

# Final Signal Generation
trade_data = (
    trade_data
    .with_columns(
        pl.col("date")
        .dt.month_start()
        .alias("month_key")
    )
)

weights_with_key = (
    rolling_weights_df
    .with_columns(
        pl.col("date")
        .dt.offset_by("1mo")
        .dt.month_start()
        .alias("month_key")
    )
    .drop("date")
)

trade_data = (
    trade_data
    .join(
        weights_with_key, 
        left_on=["month_key", "size_bucket_lag"], 
        right_on=["month_key", "size_bucket"], 
        how="inner"
    )
)

# Map inhibition thresholds to the dataframe
trade_data = (
    trade_data
    .with_columns(
        pl.col("size_bucket_lag")
        .replace_strict(inhibition_thresholds, default=None)
        .cast(pl.Float64)
        .alias("vol_limit")
    )
)

alphas = (
    trade_data
    .with_columns([
        pl.sum_horizontal([
            pl.col(f).mul(pl.col(f + "_beta")) 
            for f in reversal_factors
        ]).alias("raw_signal")
    ])
    .with_columns([
        (
            pl.col("raw_signal") 
            * (
                1 / (
                    1 + (
                        pl.col("rel_vol_lag") / pl.col("vol_limit")
                    ).pow(vol_slope)
                )
            )
        ).alias("inhibited_signal")
    ])
    .filter(
        pl.col("inhibited_signal").is_not_null()
    )
    .with_columns([
        (
            (
                pl.col("inhibited_signal")
                .sub(
                    pl.col("inhibited_signal")
                    .mean()
                    .over("date")
                )
            )
            .truediv(
                pl.col("inhibited_signal")
                .std()
                .over("date")
            )
            .mul(IC)
            .mul(pl.col("specific_risk"))
        ).alias("alpha")
    ])
    # ADDED: Safely join just the 1-to-1 Biolife flag
    .join(biolife_mapping, on=["date", "barrid"], how="left")
    .with_columns([
        # If it is a Biolife stock (1.0) and the signal is a Buy (> 0), flatten it to 0.0
        pl.when((pl.col("is_biolife") == 1.0) & (pl.col("alpha") > 0))
        .then(0.0)
        .otherwise(pl.col("alpha"))
        .alias("alpha")
    ])
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

forward_returns = (
    returns
    .sort("date", "barrid")
    .select(
        "date", 
        "barrid", 
        pl.col("return")
        .sort_by("date")
        .shift(-2)
        .over("barrid")
        .alias("return")
    )
    .drop_nulls("return")
)

ics = sfp.generate_alpha_ics(alphas=alphas, rets=forward_returns, method="rank", window=22)

sfp.generate_ic_chart(
    ics=ics, 
    title=f"{signal_name_title} Rank IC", 
    ic_type="Rank", 
    file_name=results_folder / "rank_ic_chart.png"
)

sfp.generate_ic_chart(
    ics=ics, 
    title=f"{signal_name_title} Pearson IC", 
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