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

# Script parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "shrinkage_bma_reversal"
signal_name_title = "Shrinkage BMA Reversal"
IC = 0.05
gamma = 140  
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_4")

# Hierarchical bayesian parameters
prior_window_months = 60
dynamic_window_months = 12
decay_factor = 0.90 
ema_alpha = 0.50 
null_prior_weight = 0.50
checkpoint_dir = Path("temp/checkpoints_shrinkage_bma")

# Create folders
results_folder.mkdir(
    parents=True, 
    exist_ok=True
)
checkpoint_dir.mkdir(
    parents=True, 
    exist_ok=True
)

# Define custom reversal features
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

# Compute features on full dataset to maintain chronological continuity
data = returns.sort("date", "barrid").with_columns(
    [
        (
            pl.col("price")
            .truediv(pl.col("price").sort_by("date").shift(5).over("barrid"))
            .sub(1)
        ).alias("rev_5d"),
        (
            pl.col("price")
            .truediv(pl.col("price").sort_by("date").shift(10).over("barrid"))
            .sub(1)
        ).alias("rev_10d"),
        (
            pl.col("price")
            .truediv(pl.col("price").sort_by("date").shift(20).over("barrid"))
            .sub(1)
        ).alias("rev_20d"),
        pl.col("price")
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias("prev_price")
    ]
)

# Apply universe filter
data = data.filter(
    pl.col("prev_price").gt(price_filter)
)

# Cross sectional score and invert
factors = data.select(
    ["date", "barrid"] + reversal_factors
).drop_nulls().with_columns(
    [
        (
            (pl.col(f).sub(pl.col(f).mean().over("date")))
            .truediv(pl.col(f).std().over("date"))
        ).mul(-1).alias(f)
        for f in reversal_factors
    ]
)

data = data.join(
    factors, 
    on=["date", "barrid"], 
    how="inner"
)

# Lag signals to prevent lookahead bias
data = data.with_columns(
    [
        pl.col(f)
        .sort_by("date")
        .shift(1)
        .over("barrid")
        .alias(f"{f}_lag")
        for f in reversal_factors
    ]
).drop_nulls(
    subset=[f"{f}_lag" for f in reversal_factors]
)

# Compute daily factor returns
daily_factor_returns = data.group_by("date").agg(
    [
        (pl.col(f"{f}_lag").mul(pl.col("return"))).mean().alias(f) 
        for f in reversal_factors
    ]
).sort("date")

# Compound to monthly returns
m_fac_rets = (
    daily_factor_returns.group_by_dynamic("date", every="1mo")
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

# Add null model for shrinkage prior baseline
all_combos = [()]
for k in range(1, len(reversal_factors) + 1):
    all_combos.extend(list(combinations(reversal_factors, k)))

# Calculate static shrinkage priors once
shrinkage_priors = np.zeros(len(all_combos))
for idx, subset in enumerate(all_combos):
    if len(subset) == 0:
        shrinkage_priors[idx] = null_prior_weight
    else:
        shrinkage_priors[idx] = (1.0 - null_prior_weight) / (len(all_combos) - 1)

# Stable regression for long term prior formulation
def get_bics_stable(df_X, y, combos):
    n = len(df_X)
    bics = []
    
    y_vec = y.to_numpy()
        
    for subset in combos:
        X_cols = ["const"] + list(subset)
        X = df_X.select(X_cols).to_numpy()
        
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y_vec)
            residuals = y_vec - (X @ beta)
            ssr = max(np.vdot(residuals, residuals), 1e-10)
        except np.linalg.LinAlgError:
            beta, ssr_list, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
            ssr = ssr_list[0] if len(ssr_list) > 0 else 1e-10
            
        bic = np.log(n) * len(X_cols) + n * np.log(ssr / n)
        bics.append(bic)
        
    return np.array(bics)

# Weighted regression for short term likelihood formulation
def get_bics_weighted(df_X, y, combos):
    n = len(df_X)
    bics = []
    params_list = []
    
    weights = df_X.get_column("obs_weights").to_numpy()
    
    # Normalize importance weights so they sum to total row count
    weights = weights * (n / weights.sum())
    
    sqrt_W = np.sqrt(weights)
    y_raw = y.to_numpy()
    y_w = y_raw * sqrt_W
    
    for subset in combos:
        X_cols = ["const"] + list(subset)
        X_orig = df_X.select(X_cols).to_numpy()
        X_w = df_X.select([pl.col(c) * sqrt_W for c in X_cols]).to_numpy()
        
        try:
            beta = np.linalg.solve(X_w.T @ X_w, X_w.T @ y_w)
            res = y_raw - (X_orig @ beta)
            ssr = max(np.vdot(res, weights * res), 1e-10)
        except np.linalg.LinAlgError:
            beta, ssr_list, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            ssr = ssr_list[0] if len(ssr_list) > 0 else 1e-10
            
        # Use row count as the degree of freedom proxy for penalty calculation
        bic = np.log(n) * len(X_cols) + n * np.log(ssr / n)
        bics.append(bic)
        params_list.append(dict(zip(X_cols, beta)))
        
    return np.array(bics), params_list

# Walk forward loop integrating hierarchical updates
rolling_results = []
prev_forecasts = {f: 0.0 for f in reversal_factors}

existing_checkpoints = sorted(list(checkpoint_dir.glob("*.parquet")))
if existing_checkpoints:
    latest_checkpoint = existing_checkpoints[-1]
    last_saved = pl.read_parquet(latest_checkpoint).to_dicts()[0]
    for f in reversal_factors:
        prev_forecasts[f] = last_saved[f]

# Index starts at long term window length to ensure valid prior distribution
for i in range(prior_window_months, len(unique_dates)):
    current_date = unique_dates[i]
    checkpoint_path = checkpoint_dir / f"{current_date}.parquet"
    
    if os.path.exists(checkpoint_path):
        saved = pl.read_parquet(checkpoint_path).to_dicts()[0]
        rolling_results.append(saved)
        
        for f in reversal_factors: 
            prev_forecasts[f] = saved[f]
        continue

    # Slice windows strictly separated to prevent double counting
    prior_dates = unique_dates[i - prior_window_months : i - dynamic_window_months]
    likelihood_dates = unique_dates[i - dynamic_window_months : i]
    
    # Formulate long term prior dataframe
    df_prior = timing_ready_df.filter(
        pl.col("date").is_in(prior_dates)
    ).with_columns(
        pl.lit(1.0).alias("const")
    )

    # Formulate short term likelihood dataframe
    weight_map = {
        d: (decay_factor ** (len(likelihood_dates) - 1 - idx)) 
        for idx, d in enumerate(likelihood_dates)
    }
    
    df_likelihood = timing_ready_df.filter(
        pl.col("date").is_in(likelihood_dates)
    ).with_columns(
        [
            pl.col("date").replace(weight_map).cast(pl.Float64).alias("obs_weights"),
            pl.lit(1.0).alias("const")
        ]
    )

    latest_X = timing_ready_df.filter(
        pl.col("date") == current_date
    ).row(0, named=True)
    
    month_forecasts = {"date": current_date}

    for target_factor in reversal_factors:
        target_col = f"{target_factor}_target"
        
        # Calculate unweighted long term prior probabilities
        y_prior = df_prior.get_column(target_col)
        prior_bics = get_bics_stable(df_prior, y_prior, all_combos)
        prior_probs = np.exp(-0.5 * (prior_bics - np.min(prior_bics)))
        
        # Apply shrinkage prior favoring the null model
        prior_probs = prior_probs * shrinkage_priors
        prior_probs /= prior_probs.sum()

        # Calculate weighted short term likelihood probabilities
        y_likelihood = df_likelihood.get_column(target_col)
        likelihood_bics, params = get_bics_weighted(df_likelihood, y_likelihood, all_combos)
        likelihood_probs = np.exp(-0.5 * (likelihood_bics - np.min(likelihood_bics)))
        likelihood_probs /= likelihood_probs.sum()

        # Bayes update combining prior and likelihood into posterior probability
        posterior_probs = prior_probs * likelihood_probs
        
        # Catch edge case of vanishing probabilities
        if posterior_probs.sum() == 0:
            posterior_probs = likelihood_probs
        else:
            posterior_probs /= posterior_probs.sum()

        expected_return = 0.0
        for idx, subset in enumerate(all_combos):
            m_params = params[idx]
            model_forecast = m_params.get("const", 0.0)
            
            for f in subset:
                model_forecast += m_params[f] * latest_X[f]

            expected_return += model_forecast * posterior_probs[idx]

        # Apply exponential moving average to posterior expectation
        smoothed = (ema_alpha * expected_return) + ((1 - ema_alpha) * prev_forecasts[target_factor])
        prev_forecasts[target_factor] = smoothed
        month_forecasts[target_factor] = smoothed

    pl.DataFrame([month_forecasts]).write_parquet(checkpoint_path)
    rolling_results.append(month_forecasts)

# Merge and backtest
rolling_weights_df = pl.DataFrame(rolling_results).rename(
    {f: f + "_beta" for f in reversal_factors}
)

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

alphas = signals.filter(
    pl.col(signal_name).is_not_null()
).select(
    [
        "date", 
        "barrid", 
        "predicted_beta", 
        "specific_risk",
        (
            (pl.col(signal_name).sub(pl.col(signal_name).mean().over("date")))
            .truediv(pl.col(signal_name).std().over("date"))
            .mul(IC)
            .mul(pl.col("specific_risk"))
        ).alias("alpha")
    ]
)

# Forward returns calculation on continuous dataset
forward_returns = returns.sort(
    "date", 
    "barrid"
).select(
    "date", 
    "barrid", 
    pl.col("return").sort_by("date").shift(-1).over("barrid").alias("return")
).drop_nulls(
    "return"
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