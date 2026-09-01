import datetime as dt
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from sf_backtester import BacktestDynamicConfig, BacktestDynamicRunner, SlurmConfig

start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)

temp_folder = Path("temp")
temp_folder.mkdir(parents=True, exist_ok=True)
signal_name_title = "Averaged Stratified Rev"
results_folder = Path("results/conditional_bma_reversal/experiment_10")

results_folder.mkdir(parents=True, exist_ok=True)

df = sfd.load_assets(
    start=start, 
    end=end, 
    columns=["date", "barrid", "market_cap", "daily_volume", "price", "predicted_beta"], 
    in_universe=True
)

processed_df = (
    df.sort("date", "barrid")
    .with_columns([
        ((pl.col("price") / pl.col("price").shift(1).over("barrid")) - 1).alias("ret_1d_raw"),
        ((pl.col("price") / pl.col("price").shift(5).over("barrid")) - 1).alias("ret_5d_raw"),
        ((pl.col("price") / pl.col("price").shift(10).over("barrid")) - 1).alias("ret_10d_raw"),
        ((pl.col("price") / pl.col("price").shift(20).over("barrid")) - 1).alias("ret_20d_raw"),
        (pl.col("price") * pl.col("daily_volume")).alias("dollar_vol")
    ])
    .with_columns([
        pl.when(pl.col("dollar_vol") > 0)
        .then(pl.col("ret_1d_raw").abs() / pl.col("dollar_vol").log1p())
        .otherwise(None)
        .alias("amihud_raw")
    ])
    .with_columns([
        pl.col("amihud_raw").rolling_mean(20).over("barrid").alias("amihud"),
        pl.col("ret_5d_raw").shift(1).over("barrid").alias("ret_5d"),
        pl.col("ret_10d_raw").shift(1).over("barrid").alias("ret_10d"),
        pl.col("ret_20d_raw").shift(1).over("barrid").alias("ret_20d"),
    ])
    .filter(
        pl.col("ret_5d").is_not_null() & 
        pl.col("ret_10d").is_not_null() & 
        pl.col("ret_20d").is_not_null() & 
        pl.col("amihud").is_finite() &
        pl.col("amihud").is_not_null()
    )
)

final_alphas = (
    processed_df.with_columns([
        pl.col("market_cap").qcut(5).over("date").to_physical().alias("cap_q"),
    ])
    .with_columns([
        pl.col("amihud").qcut(5).over(["date", "cap_q"]).to_physical().alias("amihud_q")
    ])
    .with_columns([
        pl.concat_str([
            pl.col("cap_q").cast(pl.String), 
            pl.lit("_"), 
            pl.col("amihud_q").cast(pl.String)
        ]).alias("size_bucket")
    ])
    .with_columns([
        (
            ((pl.col(f) - pl.col(f).mean().over(["date", "size_bucket"])) / 
             pl.col(f).std().over(["date", "size_bucket"])) * -1
        ).alias(f"z_{f}")
        for f in ["ret_5d", "ret_10d", "ret_20d"]
    ])
    .with_columns([
        ((pl.col("z_ret_5d") + pl.col("z_ret_10d") + pl.col("z_ret_20d")) / 3).alias("alpha")
    ])
    .select(["date", "barrid", "alpha", "predicted_beta"])
    .drop_nulls()
    .sort("date", "barrid")
)

alpha_path = temp_folder / "exp10_stratified_avg_alphas.parquet"
final_alphas.write_parquet(alpha_path)

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

slurm_config = SlurmConfig(n_cpus=8, mem="32G", time="03:00:00")
backtest_config = BacktestDynamicConfig(
    signal_name="stratified_avg_reversal",
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