import datetime as dt
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "raw_bma_reversal"
signal_name_title = "Raw BMA Reversal"
IC = 0.05
gamma = 140
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_1")

# Pure BMA parameters
dynamic_window_months = 12
checkpoint_dir = "temp/checkpoints_raw_reversal_bma"

# Create folders
results_folder.mkdir(parents=True, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Define custom reversal factors
reversal_factors = ["rev_5d", "rev_10d", "rev_20d"]

# Load asset data
returns = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
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

# Compute trailing price returns for reversal and previous price
data = returns.sort("date", "barrid").with_columns(
    [
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(5).over("barrid")).sub(1)).alias("rev_5d"),
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(10).over("barrid")).sub(1)).alias("rev_10d"),
        (pl.col("price").truediv(pl.col("price").sort_by("date").shift(20).over("barrid")).sub(1)).alias("rev_20d"),
        pl.col("price").sort_by("date").shift(1).over("barrid").alias("prev_price")
    ]
)

# Apply price filter prior to scoring to avoid normalization drift
data = data.filter(
    pl.col("prev_price").gt(price_filter)
)

# Cross sectional Z score for the factors
factors = data.select(
    ["date", "barrid"] + reversal_factors
).drop_nulls().with_columns(
    [
        (
            (pl.col(f).sub(pl.col(f).mean().over("date"))).truediv(pl.col(f).std().over("date"))
        ).mul(-1).alias(f)
        for f in reversal_factors
    ]
)

# Merge back to asset data
data = data.join(
    factors, 
    on=["date", "barrid"], 
    how="inner"
)

data = data.with_columns(
    [
        pl.col(f).sort_by("date").shift(1).over("barrid").alias(f"{f}_lag")
        for f in reversal_factors
    ]
).drop_nulls(
    subset=[f"{f}_lag" for f in reversal_factors]
)

# Create daily factor returns using lagged signals for BMA training
daily_factor_returns = data.group_by("date").agg(
    [
        (pl.col(f"{f}_lag").mul(pl.col("return"))).mean().alias(f)
        for f in reversal_factors
    ]
).sort("date")

# Compound daily to monthly factor returns
m_fac_rets = (
    daily_factor_returns.sort("date")
    .group_by_dynamic("date", every="1mo")
    .agg(
        [
            (((pl.col(f).add(1)).product().sub(1)) * 100).alias(f)
            for f in reversal_factors
        ]
    )
)

# Build timing ready dataframe
timing_ready_df = m_fac_rets.with_columns(
    [
        pl.col(f).shift(-1).alias(f"{f}_target") 
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

all_combos = []
for k in range(1, len(reversal_factors) + 1):
    all_combos.extend(list(combinations(reversal_factors, k)))

# Helper function to perform OLS and calculate BICs with stable SSR
def get_bics_stable(df_X, y, combos):
    n = len(df_X)
    bics = []
    params_list = []
    
    y_vec = y.to_numpy()
        
    for subset in combos:
        X_cols = ["const"] + list(subset)
        X = df_X.select(X_cols).to_numpy()
        
        try:
            # Explicit residual calculation to avoid collinear cancellation
            beta = np.linalg.solve(X.T @ X, X.T @ y_vec)
            residuals = y_vec - (X @ beta)
            ssr = max(np.vdot(residuals, residuals), 1e-10)
        except np.linalg.LinAlgError:
            beta, ssr_list, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
            ssr = ssr_list[0] if len(ssr_list) > 0 else 1e-10
            
        bic = np.log(n) * len(X_cols) + n * np.log(ssr / n)
        bics.append(bic)
        params_list.append(dict(zip(X_cols, beta)))
        
    return np.array(bics), params_list

# Walk forward loop for raw likelihood factor timing
rolling_results = []

for i in range(dynamic_window_months, len(unique_dates)):
    current_date = unique_dates[i]
    checkpoint_path = f"{checkpoint_dir}/{current_date}.parquet"

    if os.path.exists(checkpoint_path):
        saved_forecasts = pl.read_parquet(checkpoint_path).to_dicts()[0]
        rolling_results.append(saved_forecasts)
        continue

    train_dates = unique_dates[i - dynamic_window_months : i]

    df_train = timing_ready_df.filter(
        pl.col("date").is_in(train_dates)
    ).with_columns(
        pl.lit(1.0).alias("const")
    )

    month_forecasts = {"date": current_date}
    latest_X = timing_ready_df.filter(
        pl.col("date") == current_date
    ).row(0, named=True)

    for target_factor in reversal_factors:
        target_col = f"{target_factor}_target"
        y_dyn = df_train.get_column(target_col)
        
        bics, params = get_bics_stable(df_train, y_dyn, all_combos)
        
        bics_adj = bics - np.min(bics)
        probs = np.exp(-0.5 * bics_adj)
        probs /= probs.sum()

        expected_return = 0.0
        for idx, subset in enumerate(all_combos):
            m_params = params[idx]
            model_forecast = m_params.get("const", 0.0)
            
            for f in subset:
                model_forecast += m_params[f] * latest_X[f]

            expected_return += model_forecast * probs[idx]

        month_forecasts[target_factor] = expected_return

    pl.DataFrame([month_forecasts]).write_parquet(checkpoint_path)
    rolling_results.append(month_forecasts)

rolling_weights_df = pl.DataFrame(rolling_results).rename(
    {f: f + "_beta" for f in reversal_factors}
)

# Compute signal using the corrected monthly alignment
data = data.with_columns(
    pl.col("date").dt.month_start().alias("month_key")
)

weights_with_key = rolling_weights_df.with_columns(
    pl.col("date").dt.offset_by("1mo").dt.month_start().alias("month_key")
).drop("date")

data = data.join(
    weights_with_key, 
    on="month_key", 
    how="inner"
)

signals = data.with_columns(
    pl.sum_horizontal(
        [pl.col(f).mul(pl.col(f + "_beta")) for f in reversal_factors]
    ).alias(signal_name)
)

# Compute scores with the price filter already applied at the start
scores = signals.filter(
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null()
).select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    (
        (pl.col(signal_name).sub(pl.col(signal_name).mean().over("date")))
        .truediv(pl.col(signal_name).std().over("date"))
    ).alias("score"),
)

# Compute alphas
alphas = (
    scores.with_columns(
        pl.col("score").mul(IC).mul("specific_risk").alias("alpha")
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

# Merge alphas and forward returns
merged = alphas.join(
    other=forward_returns, 
    on=["date", "barrid"], 
    how="inner"
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