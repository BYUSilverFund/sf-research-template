import datetime as dt
import os
from pathlib import Path

import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from research.utils import run_backtest_parallel, run_conditional_bma_loop

load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "master_hmm_bma_reversal"
signal_name_title = "Master Double Sort BMA (HMM Macro Regime)"
IC = 0.05
gamma = 115
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_17")
checkpoint_dir = "temp/checkpoints_exp17_master_hmm"
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
    .select(["date", "barrid", "size_bucket", "rel_vol", "specific_risk"] + reversal_factors)
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
    .drop(reversal_factors + ["rel_vol", "specific_risk"])
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
        .alias("rel_vol_lag"),
        
        pl.col("specific_risk")
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias("specific_risk_lag")
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
        .sort_by("date")
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

# Final Signal Generation Map
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

# Define HMM Emissions
daily_macro = (
    trade_data
    .group_by("date")
    .agg([
        pl.col("specific_risk_lag").mean().alias("avg_specific_risk"),
        pl.col("rel_vol_lag").mean().alias("avg_rel_vol")
    ])
    .sort("date")
    .drop_nulls()
)

X_macro = daily_macro.select(["avg_specific_risk", "avg_rel_vol"]).to_numpy()

# Expanding Window HMM
train_window = 504 
prob_safe_regime = np.full(len(X_macro), np.nan)

current_model = None
current_scalar = None
safe_state_idx = 0

for t in range(train_window, len(X_macro)):
    # Monthly Refit: Training on data strictly BEFORE today
    if t % 21 == 0 or current_model is None:
        X_train = X_macro[:t] 
        
        # Fit the scaler on training data to prevent lookahead
        current_scaler = StandardScaler()
        X_train_scaled = current_scaler.fit_transform(X_train)
        
        # Fit the model on the scaled data
        current_model = GaussianHMM(
            n_components=2, 
            covariance_type="full", 
            random_state=42, 
            n_iter=100,
            tol=1e-2
        )
        current_model.fit(X_train_scaled)
        
        # Identify Safe State (Low Risk State)
        risk_mean_0 = current_model.means_[0, 0]
        risk_mean_1 = current_model.means_[1, 0]
        safe_state_idx = 0 if risk_mean_0 < risk_mean_1 else 1
        
    X_up_to_today_scaled = current_scaler.transform(X_macro[:t+1])
    
    # Isolate the Forward pass by taking only the last prediction [-1]
    daily_proba = current_model.predict_proba(X_up_to_today_scaled)
    prob_safe_regime[t] = daily_proba[-1, safe_state_idx]

# Map back to daily_macro
daily_macro = daily_macro.with_columns(
    pl.Series("prob_mr", prob_safe_regime).fill_null(1.0)
).select(["date", "prob_mr"])

# Final alpha generation with HMM scaling
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
        ).alias("alpha_pre_hmm")
    ])
    .join(daily_macro, on="date", how="left")
    .with_columns([
        (pl.col("alpha_pre_hmm") * pl.col("prob_mr")).alias("alpha")
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