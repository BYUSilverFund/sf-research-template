import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sf_quant.data as sfd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parameters
end = dt.date(2024, 12, 31)
signal_name = "fast_decay_volume_bma"
gamma = 90
results_folder = Path("results/experiment_10")
weights_path = Path(f"weights/{signal_name}/{gamma}/2024.parquet")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load pre-computed weights for the last day
weights = (
    pl.read_parquet(weights_path)
    .filter(pl.col("date") == end)
    .rename({"weight": "total_weight"})
)

# Load benchmark weights
benchmark_weights = (
    sfd.load_benchmark(start=end, end=end)
    .filter(pl.col("date") == end)
    .rename({"weight": "bmk_weight"})
)

# Load ticker mapping
tickers = sfd.load_assets_by_date(
    date_=end, 
    in_universe=True, 
    columns=["barrid", "ticker"]
)

# Compute active weights and benchmark percentage change
all_weights = (
    benchmark_weights
    .join(other=weights, on=["barrid"], how="left")
    .join(other=tickers, on=["barrid"], how="left")
    .with_columns(pl.col("total_weight").fill_null(0.0))
    .select(
        "date",
        "ticker",
        "barrid",
        "total_weight",
        "bmk_weight",
        pl.col("total_weight").sub(pl.col("bmk_weight")).alias("active_weight"),
    )
    .with_columns(
        pl.col("total_weight").truediv(pl.col("bmk_weight")).sub(1).alias("pct_change_bmk")
    )
    .sort("total_weight", descending=True)
)

# Calculate Active Risk
active_weights_np = all_weights.sort("barrid")["active_weight"].to_numpy()
barrids = all_weights.sort("barrid")["barrid"].to_list()

covariance_matrix = sfd.construct_covariance_matrix(date_=end, barrids=barrids)
covariance_matrix_np = covariance_matrix.drop("barrid").to_numpy()

active_risk = np.sqrt(active_weights_np @ covariance_matrix_np @ active_weights_np.T)
print(f"Active Risk: {active_risk * 100:.2f}%")

# Format data for Matplotlib table
top_10_df = all_weights.head(10).to_pandas()

# Drop barrid for cleaner presentation
top_10_df = top_10_df[["date", "ticker", "total_weight", "bmk_weight", "active_weight", "pct_change_bmk"]]

# Format the columns
top_10_df["date"] = top_10_df["date"].astype(str)
top_10_df["total_weight"] = top_10_df["total_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["bmk_weight"] = top_10_df["bmk_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["active_weight"] = top_10_df["active_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["pct_change_bmk"] = top_10_df["pct_change_bmk"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) and not np.isinf(x) else "N/A")

# Rename columns for the plot
top_10_df.columns = ["Date", "Ticker", "Total Weight", "Benchmark Weight", "Active Weight", "Benchmark Percent Change"]

# Render and save the Matplotlib table fig
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(
    cellText=top_10_df.values, 
    colLabels=top_10_df.columns, 
    loc='center', 
    cellLoc='center'
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Format header colors
for i in range(len(top_10_df.columns)):
    header_cell = table[(0, i)]
    header_cell.set_facecolor("#c6c6c6")
    header_cell.set_text_props(color="black", weight="bold")
    
    if i == len(top_10_df.columns) - 1:
        header_cell.set_fontsize(8)
    else:
        header_cell.set_fontsize(10)

plt.title("BMA Fast Decayed Log-Volume Portfolio (Top 10 Holdings)", weight="bold", pad=20)

table_path = results_folder / "portfolio.png"
plt.savefig(table_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Portfolio table successfully saved to {table_path}")