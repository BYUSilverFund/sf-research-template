import datetime as dt
import os
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from sf_backtester import BacktestDynamicConfig, BacktestDynamicRunner, SlurmConfig

load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "orthogonalized_std_reversal"
signal_name_title = "Residual Std Reversal"
target_signal_name = "reversal" 
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]

results_folder = Path("results/conditional_bma_reversal/experiment_11") 
results_folder.mkdir(parents=True, exist_ok=True)
temp_folder = Path("temp")
temp_folder.mkdir(parents=True, exist_ok=True)

# Data Loading for Alphas
alphas_7 = pl.read_parquet("temp/alphas_7.parquet")

std_rev_alphas = (
    pl.read_parquet("temp/std_rev_alphas.parquet")
    .filter(pl.col("signal_name") == target_signal_name)
    .select([
        "date", 
        "barrid", 
        pl.col("alpha").alias("std_alpha")
    ])
)

# Data Loading for ICs
returns = (
    sfd.load_assets(
        start=start, 
        end=end,
        columns=["date", "barrid", "return"],
        in_universe=True,
    )
    .with_columns([
        pl.col("return").truediv(100)
    ])
)

forward_returns = (
    returns
    .sort("date", "barrid")
    .select(
        "date", 
        "barrid", 
        pl.col("return")
        .sort_by("date")
        .shift(-1)
        .over("barrid")
        .alias("return")
    )
    .drop_nulls("return")
)

# Alignment
joined_alphas = (
    alphas_7
    .join(
        std_rev_alphas,
        on=["date", "barrid"],
        how="inner"
    )
    .drop_nulls(["alpha", "std_alpha"])
)

# Cross-Sectional Regression
# Y = Std Reversal Alpha, X = Exp 7 Alpha (Master BMA)
orthogonalized = (
    joined_alphas
    .with_columns([
        # Calculate Beta: Cov(X,Y) / Var(X)
        (
            pl.cov("alpha", "std_alpha").over("date") 
            / pl.col("alpha").var().over("date")
        ).alias("beta")
    ])
    .with_columns([
        # Calculate Intercept: Mean(Y) - Beta * Mean(X)
        (
            pl.col("std_alpha").mean().over("date") 
            - pl.col("beta").mul(pl.col("alpha").mean().over("date"))
        ).alias("intercept")
    ])
    .with_columns([
        # Calculate Residual: Y - (Beta * X + Intercept)
        (
            pl.col("std_alpha") 
            - (
                pl.col("beta").mul(pl.col("alpha")) 
                + pl.col("intercept")
            )
        ).alias("residual_alpha")
    ])
)

# Final Alpha Formatting
final_alphas = (
    orthogonalized
    .select([
        "date", 
        "barrid", 
        pl.col("residual_alpha").alias("alpha"), 
        "predicted_beta"
    ])
    .sort("date", "barrid")
)

alpha_path = temp_folder / "alphas_11.parquet"
final_alphas.write_parquet(alpha_path)

# Generate IC Charts
ics = sfp.generate_alpha_ics(alphas=final_alphas, rets=forward_returns, method="rank", window=22)

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

# Run Backtest on Residuals
slurm_config = SlurmConfig(n_cpus=8, mem="32G", time="03:00:00")
backtest_config = BacktestDynamicConfig(
    signal_name=signal_name,
    data_path=str(alpha_path),
    initial_gamma=50,
    target_active_risk=0.05,
    active_weights=True,
    project_root="/home/grich27/Projects/sf-research-bayesian-reversal",
    byu_email="grich27@byu.edu",
    constraints=["ZeroBeta", "ZeroInvestment"],
    slurm=slurm_config
)

BacktestDynamicRunner(backtest_config).submit()