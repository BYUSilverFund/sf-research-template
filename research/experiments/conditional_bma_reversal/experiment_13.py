import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import sf_quant.data as sfd
import sf_quant.optimizer as sfo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(2024, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "barra_reversal_volume_clipped"
results_folder = Path("results/conditional_bma_reversal/experiment_13")
IC = 0.05
n_cpus = 8
constraints = [
    sfo.FullInvestment(),
    sfo.LongOnly(),
    sfo.NoBuyingOnMargin(),
    sfo.UnitBeta(),
]

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Get data
data = sfd.load_assets(
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
        "daily_volume",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# Compute signal
signals = data.sort("barrid", "date").with_columns(
    pl.col("specific_return")
    .ewm_mean(span=5, min_samples=5)
    .mul(-1)
    .shift(1)
    .over("barrid")
    .alias(signal_name)
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col(signal_name).is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
    pl.col("daily_volume").is_not_null(),
)

# Compute scores
scores = filtered.select(
    "date",
    "barrid",
    "price",
    "predicted_beta",
    "specific_risk",
    "daily_volume",
    pl.col(signal_name)
    .sub(pl.col(signal_name).mean())
    .truediv(pl.col(signal_name).std())
    .over("date")
    .alias("score"),
)

# Windsorize scores
scores = scores.with_columns(pl.col("score").clip(lower_bound=-2.0, upper_bound=2.0))

volume_scores = (
    scores.sort(["barrid", "date"])
    .with_columns(dollar_volume=pl.col("daily_volume").mul(pl.col("price")).log1p())
    .with_columns(
        # Mean can be calculated on Day 1
        dollar_volume_mean=pl.col("dollar_volume")
        .rolling_mean(window_size=252, min_samples=1)
        .over("barrid"),
        # Std Dev requires min_samples=2.
        # It will still produce a null on Day 1.
        dollar_volume_std=pl.col("dollar_volume")
        .rolling_std(window_size=252, min_samples=2)
        .over("barrid"),
    )
    .with_columns(
        volume_score=(
            (pl.col("dollar_volume") - pl.col("dollar_volume_mean"))
            /
            # fill the Day 1 null std with 1.0 (or any non-zero) to avoid division by null
            pl.col("dollar_volume_std").fill_null(1.0).clip(lower_bound=0.0001)
        )
        .fill_null(0.0)  # Catch any remaining edge cases
        .alias("volume_score")
    )
)

# Convert just the score column to Pandas/Numpy for plotting
score_data = volume_scores["volume_score"].to_numpy()

# Filter universe to just be last day
volume_scores = volume_scores.filter(pl.col("date").eq(end))

# Compute alphas with conditional logic: Set alpha to 0 if reversal is high with strong volume
alphas = (
    volume_scores.with_columns(
        # grinold and kahn alpha
        gk_alpha=pl.col("score") * IC * pl.col("specific_risk")
    )
    .with_columns(
        # Set alpha to 0 if both score == 2.0 and volume_score > 2.0
        alpha=pl.when((pl.col("score").eq(2.0)) & (pl.col("volume_score") > 2.0))
        .then(0.0)
        .otherwise(pl.col("gk_alpha"))
    )
    .select("date", "barrid", "alpha", "predicted_beta")
    .sort("date", "barrid")
)

# Set up arrays
alphas_np = alphas["alpha"].to_numpy()
betas_np = alphas["predicted_beta"].to_numpy()
barrids = alphas["barrid"].to_list()

# Get factor model components
factor_exposures, factor_covariance, specific_risk = sfd.construct_factor_model_components(
    date_=end, 
    barrids=barrids
)

# Load benchmark early to calculate active risk during optimization
benchmark_weights = sfd.load_benchmark(start=end, end=end).filter(pl.col("date").eq(end))
bmk_dict = dict(zip(benchmark_weights["barrid"].to_list(), benchmark_weights["weight"].to_list()))
w_bmk = np.array([bmk_dict.get(b, 0.0) for b in barrids])

# Optimization: Iteratively find Gamma to hit 5% Active Risk
target_risk = 0.05
current_gamma = 15.0  # Starting guess

print("Optimizing to target 5.00% Active Risk...")
for iteration in range(10):
    weights = sfo.mve_optimizer(
        ids=barrids, 
        alphas=alphas_np, 
        factor_exposures=factor_exposures,
        factor_covariance=factor_covariance,
        specific_risk=specific_risk,
        constraints=constraints, 
        gamma=current_gamma, 
        betas=betas_np
    )

    # Calculate resulting Active Risk
    w_total = weights["weight"].to_numpy()
    w_active = w_total - w_bmk
    
    # Risk calculation using the factor model components
    f_risk = (w_active @ factor_exposures) @ factor_covariance @ (w_active @ factor_exposures).T
    s_risk = np.sum((w_active**2) * specific_risk)
    actual_risk = np.sqrt(f_risk + s_risk)

    # Stop if within 0.1% of our 5% target
    if abs(actual_risk - target_risk) < 0.001: 
        break
        
    # Adjust Gamma proportionally
    current_gamma = current_gamma * (actual_risk / target_risk)

print(f"Final Active Risk: {actual_risk*100:.2f}% at Gamma: {current_gamma:.2f}")

# Get ticker mapping
tickers = sfd.load_assets_by_date(
    date_=end, in_universe=True, columns=["barrid", "ticker"]
)

# Compute active weights and pct_change_bmk
all_weights = (
    benchmark_weights.rename({"weight": "bmk_weight"})
    .join(other=weights.rename({"weight": "total_weight"}), on=["barrid"], how="left")
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
        pl.when(pl.col("bmk_weight") > 0)
        .then(pl.col("total_weight").truediv("bmk_weight").sub(1))
        .otherwise(pl.lit(None))
        .alias("pct_change_bmk")
    )
    .sort("total_weight", descending=True)
)

# Render with Matplotlib
top_10_df = all_weights.head(10).to_pandas()
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

plt.title("Volume Conditioned Barra Reversal Portfolio (Total)", weight="bold", pad=20)
table_path = results_folder / "vol_cond_barra_rev_sample_port.png"
plt.savefig(table_path, bbox_inches='tight', dpi=300)
plt.close()