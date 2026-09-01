import argparse
import datetime as dt

import polars as pl
import sf_quant.backtester as sfb
import sf_quant.optimizer as sfo


def get_constraints_from_names(constraint_names: list[str]) -> list:
    """Convert constraint names to sfo.constraints objects."""
    constraint_map = {
        "ZeroBeta": sfo.constraints.ZeroBeta,
        "ZeroInvestment": sfo.constraints.ZeroInvestment,
    }

    return [constraint_map[name]() for name in constraint_names]


def run_backtest_by_year(
    df: pl.LazyFrame,
    gamma: float,
    year: int,
    output_dir: str,
    n_cpus: int,
    constraints: list[str],
) -> None:
    year_start = dt.date(year, 1, 1)
    year_end = dt.date(year, 12, 31)

    filtered = (
        df.filter(pl.col("date").is_between(year_start, year_end))
        .select(["date", "barrid", "alpha", "predicted_beta"])
        .collect()
    )

    constraint_objects = get_constraints_from_names(constraints)

    weights = sfb.backtest_parallel(
        data=filtered, constraints=constraint_objects, gamma=gamma, n_cpus=n_cpus
    )

    weights.write_parquet(f"{output_dir}/{year}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run signal weighting on a parquet dataset."
    )

    parser.add_argument("--data_path", help="Path to parquet file containing the data")
    parser.add_argument("--gamma", type=float, help="Gamma parameter for MVO")
    parser.add_argument("--year", type=int, help="Year to process")
    parser.add_argument("--output_dir", help="Directory to write output parquet file")
    parser.add_argument("--n_cpus", type=int, help="Number of cpus to use")
    parser.add_argument("--constraints", nargs="+", help="List of constraint names")

    args = parser.parse_args()

    # Load parquet into polars DataFrame
    df = pl.scan_parquet(args.data_path)

    # Run the signal weights calculation
    run_backtest_by_year(
        df=df,
        gamma=args.gamma,
        year=args.year,
        output_dir=args.output_dir,
        n_cpus=args.n_cpus,
        constraints=args.constraints,
    )
