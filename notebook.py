import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import great_tables as gt
    import marimo as mo
    import polars as pl
    import sf_quant.data as sfd

    return alt, gt, mo, pl, sfd


@app.cell
def _(alt):
    alt.data_transformers.enable("vegafusion")
    return


@app.cell
def _(mo):
    start = mo.ui.date(
        value="1996-01-01",
        start="1996-01-01",
        stop="2024-12-31",
    )

    end = mo.ui.date(
        value="2024-12-31",
        start="1996-01-01",
        stop="2024-12-31",
    )

    signal_names = mo.ui.multiselect(
        value=[
            "vol_scaled_momentum",
            "vol_scaled_capm_momentum",
            "vol_scaled_ff3_momentum",
            "vol_scaled_ff5_momentum",
            "vol_scaled_barra_momentum",
        ],
        options=[
            "momentum",
            "vol_scaled_momentum",
            "capm_momentum",
            "vol_scaled_capm_momentum",
            "ff3_momentum",
            "vol_scaled_ff3_momentum",
            "ff5_momentum",
            "vol_scaled_ff5_momentum",
            "barra_momentum",
            "vol_scaled_barra_momentum",
        ],
        label="Select signals",
    )

    mo.vstack([start, end, signal_names])
    return end, signal_names, start


@app.cell
def _(end, pl, signal_names, start):
    gammas = [
        {
            "momentum": 50,
            "vol_scaled_momentum": 50,
            "barra_momentum": 50,
            "vol_scaled_barra_momentum": 43,
            "ff3_momentum": 60,
            "vol_scaled_ff3_momentum": 60,
            "ff5_momentum": 60,
            "vol_scaled_ff5_momentum": 60,
            "capm_momentum": 60,
            "vol_scaled_capm_momentum": 60,
        }[signal_name]
        for signal_name in signal_names.value
    ]

    weights_list = []
    for signal, gamma in zip(signal_names.value, gammas):
        signal_weights = (
            pl.read_parquet(f"weights/{signal}/{gamma}/*.parquet")
            .filter(pl.col("date").is_between(start.value, end.value))
            .with_columns(pl.lit(signal).alias("signal"))
        )
        weights_list.append(signal_weights)

    weights = pl.concat(weights_list)
    return (weights,)


@app.cell
def _(end, pl, sfd, start):
    # Get returns
    returns = (
        sfd.load_assets(
            start=start.value,
            end=end.value,
            columns=["date", "barrid", "return"],
            in_universe=True,
        )
        .sort("date", "barrid")
        .select(
            "date",
            "barrid",
            pl.col("return")
            .truediv(100)
            .shift(-1)
            .over("barrid")
            .alias("forward_return"),
        )
    )
    return (returns,)


@app.cell
def _(pl, returns, weights):
    # Compute portfolio returns
    portfolio_returns = (
        weights.join(other=returns, on=["date", "barrid"], how="left")
        .group_by("date", "signal")
        .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("return"))
        .sort("date", "signal")
    )
    return (portfolio_returns,)


@app.cell
def _(pl, portfolio_returns):
    # Compute cumulative log returns
    cumulative_returns = portfolio_returns.select(
        "date",
        "signal",
        pl.col("return")
        .log1p()
        .cum_sum()
        .mul(100)
        .over("signal")
        .alias("cumulative_return"),
    )
    return (cumulative_returns,)


@app.cell
def _(alt, cumulative_returns):
    # Plot cumulative log returns
    chart = (
        alt.Chart(cumulative_returns, title="MVO Backtest Results (Active)")
        .mark_line()
        .encode(
            x=alt.X("date", title=""),
            y=alt.Y("cumulative_return", title="Cumulative Log Return (%)"),
            color=alt.Color("signal", title="Signal"),
        )
        .properties(width=800, height=400)
    )
    chart
    return


@app.cell
def _(gt, pl, portfolio_returns):
    # Create summary table
    summary = (
        portfolio_returns.group_by("signal")
        .agg(
            pl.col("return").mean().mul(252).alias("mean_return"),
            pl.col("return").std().mul(pl.lit(252).sqrt()).alias("volatility"),
        )
        .with_columns(
            pl.col("mean_return").truediv(pl.col("volatility")).alias("sharpe")
        )
    )

    table = (
        gt.GT(summary.sort("signal"))
        .tab_header(title="MVO Backtest Results (Active)")
        .cols_label(
            signal="Signal",
            mean_return="Mean Return",
            volatility="Volatility",
            sharpe="Sharpe",
        )
        .fmt_percent(["mean_return", "volatility"], decimals=2)
        .fmt_number("sharpe", decimals=2)
        .opt_stylize(style=4, color="gray")
    )

    table
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
