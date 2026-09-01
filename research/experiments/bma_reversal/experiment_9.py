import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import sf_quant.data as sfd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
TARGET_SIGNAL_NAME = "reversal" 
BMA_PATH = Path("temp/bma_reversal_alphas.parquet")
PORTFOLIO_PATH = Path("temp/portfolio_alphas.parquet")
RESULTS_DIR = Path("results/experiment_9")

# Ensure output directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load and pivot portfolio alphas
std_alphas = (
    pl.read_parquet(PORTFOLIO_PATH)
    .filter(pl.col("signal_name") == TARGET_SIGNAL_NAME)
    .select([
        "date",
        "barrid",
        pl.col("alpha").alias("std_alpha")
    ])
)

# Load BMA alphas
bma_alphas = (
    pl.read_parquet(BMA_PATH)
    .select([
        "date",
        "barrid",
        pl.col("alpha").alias("exp8_alpha")
    ])
)

# Load forward returns for target variable
forward_returns = (
    sfd.load_assets(
        start=dt.date(1996, 1, 1), 
        end=dt.date(2024, 12, 31), 
        columns=["date", "barrid", "return"], 
        in_universe=True
    )
    .with_columns(pl.col("return").truediv(100))
    .sort("date", "barrid")
    .select([
        "date", 
        "barrid",
        pl.col("return")
        .sort_by("date")
        .shift(-1)
        .over("barrid")
        .alias("fwd_ret")
    ])
)

# Merge and daily z-score
merged = (
    bma_alphas
    .join(std_alphas, on=["date", "barrid"], how="inner")
    .join(forward_returns, on=["date", "barrid"], how="inner")
    .with_columns([
        ((pl.col("exp8_alpha") - pl.col("exp8_alpha").mean().over("date")) / 
          pl.col("exp8_alpha").std().over("date")).alias("z_exp8"),
        ((pl.col("std_alpha") - pl.col("std_alpha").mean().over("date")) / 
          pl.col("std_alpha").std().over("date")).alias("z_std")
    ])
    .drop_nulls()
)

# Analysis function
def get_stats(df_subset, label):
    
    # Daily Spearman Rank Correlation
    avg_corr = (
        df_subset.group_by("date")
        .agg(pl.corr("z_exp8", "z_std", method="spearman").alias("corr"))
        .select(pl.col("corr").mean())
        .item() 
    )

    # Fama-MacBeth Regression via Partition Iteration
    b_exp8_list, b_std_list = [], []
    
    for (date_key,), group_df in df_subset.sort("date").group_by("date", maintain_order=True):
        if len(group_df) > 20:
            y = group_df["fwd_ret"].to_numpy()
            X = sm.add_constant(group_df[["z_exp8", "z_std"]].to_numpy())
            
            try:
                params = sm.OLS(y, X).fit().params
                b_exp8_list.append(params[1])
                b_std_list.append(params[2])
            except np.linalg.LinAlgError:
                continue

    # Calculate standard and newey-west t-stats
    b_exp8_arr = np.array(b_exp8_list)
    b_std_arr = np.array(b_std_list)
    n = len(b_exp8_arr)
    
    # Standard error t-stats
    t_exp8 = b_exp8_arr.mean() / (b_exp8_arr.std() / np.sqrt(n))
    t_std = b_std_arr.mean() / (b_std_arr.std() / np.sqrt(n))
    
    # Newey-West t-stats via intercept-only regression
    nw_model_exp8 = sm.OLS(b_exp8_arr, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    nw_model_std = sm.OLS(b_std_arr, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    
    nw_t_exp8 = nw_model_exp8.tvalues[0]
    nw_t_std = nw_model_std.tvalues[0]

    # Print to terminal
    print(f"{label:<15} | Corr: {avg_corr:.4f} | Exp8 T: {t_exp8:.2f} (NW: {nw_t_exp8:.2f}) | Std T: {t_std:.2f} (NW: {nw_t_std:.2f})")

    # Return as dictionary for table creation
    return {
        "Period": label,
        "Signal Corr": avg_corr,
        "BMA Rev T-Stat": t_exp8,
        "BMA Rev NW T-Stat": nw_t_exp8,
        "Rev T-Stat": t_std,
        "Rev NW T-Stat": nw_t_std
    }

# Execute sub-period tests
cutoff_10y = merged["date"].max() - dt.timedelta(days=365 * 10)

print(f"{'Period':<15} | {'Signal Corr':<12} | {'Exp8 T-Stat':<22} | {'Std T-Stat':<22}")
print("-" * 85)

results_data = [
    get_stats(merged, "Full Sample"),
    get_stats(merged.filter(pl.col("date") >= cutoff_10y), "Last 10 Years")
]

# Create DataFrame
results_df = pl.DataFrame(results_data)

# Format the data for clean visualization
pd_df = results_df.to_pandas()
pd_df["Signal Corr"] = pd_df["Signal Corr"].apply(lambda x: f"{x:.4f}")
pd_df["BMA Rev T-Stat"] = pd_df["BMA Rev T-Stat"].apply(lambda x: f"{x:.2f}")
pd_df["BMA Rev NW T-Stat"] = pd_df["BMA Rev NW T-Stat"].apply(lambda x: f"{x:.2f}")
pd_df["Rev T-Stat"] = pd_df["Rev T-Stat"].apply(lambda x: f"{x:.2f}")
pd_df["Rev NW T-Stat"] = pd_df["Rev NW T-Stat"].apply(lambda x: f"{x:.2f}")

# Render and save as PNG using Matplotlib
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')

# Create the table
table = ax.table(
    cellText=pd_df.values, 
    colLabels=pd_df.columns, 
    loc='center', 
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Format header colors
for i in range(len(pd_df.columns)):
    table[(0, i)].set_facecolor("#c6c6c6")
    table[(0, i)].set_text_props(color="black", weight="bold")
    table[(0, i)].set_fontsize(9)

output_path = RESULTS_DIR / "fama_macbeth_results.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()