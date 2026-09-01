import datetime as dt
import os
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "orthogonalized_master_bma"
signal_name_title = "Residual Master BMA"
target_signal_name = "reversal" 
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/conditional_bma_reversal/experiment_8")

results_folder.mkdir(parents=True, exist_ok=True)

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
)

# Cross-Sectional Regression
# Y = Exp 7 Alpha, X = Std Reversal Alpha
orthogonalized = (
    joined_alphas
    .with_columns([
        # Calculate Beta: Cov(X,Y) / Var(X)
        (
            pl.cov("std_alpha", "alpha").over("date") 
            / pl.col("std_alpha").var().over("date")
        ).alias("beta")
    ])
    .with_columns([
        # Calculate Intercept: Mean(Y) - Beta * Mean(X)
        (
            pl.col("alpha").mean().over("date") 
            - pl.col("beta").mul(pl.col("std_alpha").mean().over("date"))
        ).alias("intercept")
    ])
    .with_columns([
        # Calculate Residual: Y - (Beta * X + Intercept)
        (
            pl.col("alpha") 
            - (
                pl.col("beta").mul(pl.col("std_alpha")) 
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
run_backtest_parallel(
    data=final_alphas, 
    signal_name=signal_name, 
    constraints=constraints, 
    gamma=gamma, 
    n_cpus=n_cpus
)