import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
from dotenv import load_dotenv

load_dotenv()

# Parameters
end = dt.date(2024, 12, 31)
results_folder = Path("results/conditional_bma_reversal/experiment_9")
results_folder.mkdir(parents=True, exist_ok=True)
gamma = 50

alphas_path = Path("temp/alphas_7.parquet") 

constraints = [
    sfo.FullInvestment(),
    sfo.LongOnly(),
    sfo.NoBuyingOnMargin(),
    sfo.UnitBeta(),
]

# Data Loading & Prep
alphas = pl.read_parquet(alphas_path).filter(pl.col("date") == end)

alphas_np = alphas["alpha"].to_numpy()
betas_np = alphas["predicted_beta"].to_numpy()
barrids = alphas["barrid"].to_list()

# Get factor model components
factor_exposures, factor_covariance, specific_risk = sfd.construct_factor_model_components(
    date_=end, 
    barrids=barrids
)

# Optimization
weights = sfo.mve_optimizer(
    ids=barrids, 
    alphas=alphas_np, 
    factor_exposures=factor_exposures,
    factor_covariance=factor_covariance,
    specific_risk=specific_risk,
    constraints=constraints, 
    gamma=gamma, 
    betas=betas_np
)

# Calculate and Print Active Risk
bmk_df = sfd.load_benchmark(start=end, end=end).filter(pl.col("date") == end)
bmk_dict = dict(zip(bmk_df["barrid"].to_list(), bmk_df["weight"].to_list()))
w_bmk = np.array([bmk_dict.get(b, 0.0) for b in barrids])
w_active = weights["weight"].to_numpy() - w_bmk

f_risk = (w_active @ factor_exposures) @ factor_covariance @ (w_active @ factor_exposures).T
s_risk = np.sum((w_active**2) * specific_risk)
active_risk = np.sqrt(f_risk + s_risk)

print(f"Active Risk: {active_risk * 100:.2f}%")

# Table Formatting
benchmark_weights = sfd.load_benchmark(start=end, end=end).filter(pl.col("date") == end)
tickers = sfd.load_assets_by_date(date_=end, in_universe=True, columns=["barrid", "ticker"])

all_weights = (
    benchmark_weights.rename({"weight": "bmk_weight"})
    .join(weights.rename({"weight": "total_weight"}), on="barrid", how="left")
    .join(tickers, on="barrid", how="left")
    .with_columns([
        pl.col("total_weight").fill_null(0.0),
        (pl.col("total_weight") - pl.col("bmk_weight")).alias("active_weight"),
        (pl.col("total_weight") / pl.col("bmk_weight") - 1).alias("pct_change_bmk")
    ])
    .sort("total_weight", descending=True).head(10)
)

# Render with Matplotlib
top_10_df = all_weights.to_pandas()
top_10_df = top_10_df[["date", "ticker", "total_weight", "bmk_weight", "active_weight", "pct_change_bmk"]]

top_10_df["date"] = top_10_df["date"].astype(str)
top_10_df["total_weight"] = top_10_df["total_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["bmk_weight"] = top_10_df["bmk_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["active_weight"] = top_10_df["active_weight"].apply(lambda x: f"{x*100:.2f}%")
top_10_df["pct_change_bmk"] = top_10_df["pct_change_bmk"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
top_10_df.columns = ["Date", "Ticker", "Total Weight", "Benchmark Weight", "Active Weight", "Benchmark Percent Change"]

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=top_10_df.values, colLabels=top_10_df.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

for i in range(len(top_10_df.columns)):
    header_cell = table[(0, i)]
    header_cell.set_facecolor("#c6c6c6")
    header_cell.set_text_props(color="black", weight="bold")
    header_cell.set_fontsize(8 if i == len(top_10_df.columns) - 1 else 10)

plt.title("Enhanced BMA Reversal Sample Portfolio (Total)", weight="bold", pad=20)
save_path = results_folder / "enhanced_bma_rev_sample_port.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()