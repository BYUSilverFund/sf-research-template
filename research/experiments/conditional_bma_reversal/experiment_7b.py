import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import sf_quant.data as sfd
import statsmodels.formula.api as smf

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
signal_name = "master_bma_reversal"
gamma = 115
results_folder = Path("results/conditional_bma_reversal/experiment_7")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)


# Helper function to save a DataFrame as a stylized Matplotlib table image
def save_table_image(df: pd.DataFrame, title: str, filepath: Path):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4 + 1.5))
    ax.axis("off")
    ax.axis("tight")
    ax.set_title(title, fontweight="bold", fontsize=12, pad=15)

    table = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Stylize header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


# Load MVO weights
weights = pl.read_parquet(f"weights/{signal_name}/{gamma}/*.parquet")

# Get returns
returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
    )
)

# Compute portfolio returns
portfolio_returns = (
    weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
    .sort("date")
)

# Compute cumulative log returns
cumulative_returns = portfolio_returns.select(
    "date", pl.col("return").log1p().cum_sum().mul(100).alias("cumulative_return")
)

# Plot cumulative log returns using Matplotlib
cum_rets_pd = cumulative_returns.to_pandas()

plt.figure(figsize=(10, 5))
plt.plot(
    cum_rets_pd["date"],
    cum_rets_pd["cumulative_return"],
    color="#1f77b4",
    linewidth=1.5,
)
plt.title("MVO Backtest Results (Active)", fontweight="bold", pad=15)
plt.ylabel("Cumulative Log Return (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

chart_path = results_folder / "cumulative_returns.png"
plt.savefig(chart_path, dpi=300, bbox_inches="tight")
plt.close()

# Create summary table
summary = portfolio_returns.select(
    pl.col("return").mean().mul(252).alias("mean_return"),
    pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
).with_columns(pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe"))

# Format for Matplotlib table
summary_pd = summary.to_pandas()
summary_pd["mean_return"] = summary_pd["mean_return"].apply(lambda x: f"{x:.2%}")
summary_pd["volatility"] = summary_pd["volatility"].apply(lambda x: f"{x:.2%}")
summary_pd["sharpe"] = summary_pd["sharpe"].apply(lambda x: f"{x:.2f}")
summary_pd.columns = ["Mean Return", "Volatility", "Sharpe"]

table_path = results_folder / "summary_table.png"
save_table_image(summary_pd, "MVO Backtest Results", table_path)

# Fama french regression
ff5 = (
    sfd.load_fama_french(start=start, end=end)
    .sort("date")
    .with_columns(pl.exclude("date").shift(-1))
)

regression_data = (
    portfolio_returns.join(other=ff5, on="date", how="left")
    .drop_nulls("return")
    .with_columns(pl.col("return").sub("rf").alias("return_rf"))
    .with_columns(pl.exclude("date").mul(100))
)

formula = "return_rf ~ mkt_rf + smb + hml + rmw + cma"
model = smf.ols(formula, regression_data.to_pandas())
results = model.fit()

regression_summary = pl.DataFrame(
    {
        "variable": results.params.index,
        "coefficient": results.params.values,
        "tstat": results.tvalues.values,
    }
)

# Format for Matplotlib table
reg_pd = regression_summary.to_pandas()
reg_pd["coefficient"] = reg_pd["coefficient"].apply(lambda x: f"{x:.4f}")
reg_pd["tstat"] = reg_pd["tstat"].apply(lambda x: f"{x:.4f}")
reg_pd.columns = ["Variable", "Coefficient", "T-stat"]

reg_table_path = results_folder / "regression_table.png"
save_table_image(reg_pd, "MVO Regression (Daily %)", reg_table_path)
