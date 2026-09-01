import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import sf_quant.data as sfd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup and helper function
target_date = dt.date(2024, 12, 31)
alphas_path = Path("temp/alphas_7.parquet") 
results_dir = Path("results/conditional_bma_reversal/experiment_15")
results_dir.mkdir(parents=True, exist_ok=True)

def save_table_image(pl_df, title, filename, footer_text=None):
    """Converts a Polars DataFrame to a styled Matplotlib table image."""
    # Convert to pandas for easier plotting
    df = pl_df.to_pandas()
    
    # Format the alpha column to 5 decimal places for clean viewing
    if "alpha" in df.columns:
        df["alpha"] = df["alpha"].apply(lambda x: f"{float(x):.5f}")
        
    # Format return column as percentage for the final table
    for col in df.columns:
        if "return" in col.lower():
            df[col] = df[col].apply(lambda x: f"{float(x)*100:.2f}%" if pd.notnull(x) else "N/A")
            
    # Dynamic figure height based on number of rows and presence of footer
    fig_height = max(5, 0.4 * len(df) + (2.5 if footer_text else 1.5))
    fig, ax = plt.subplots(figsize=(12, fig_height)) 
    
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=[c.replace("_", " ").title() for c in df.columns],
        cellLoc='center',
        loc='center'
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Format headers
    for i in range(len(df.columns)):
        header_cell = table[(0, i)]
        header_cell.set_facecolor("#c6c6c6")
        header_cell.set_text_props(color="black", weight="bold")
            
    plt.title(title, weight='bold', size=14, pad=20)
    
    # Render the footer box if provided
    if footer_text:
        plt.figtext(
            0.5, 0.08, footer_text, 
            ha='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='grey')
        )
        
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Load data and map industries
alphas = pl.read_parquet(alphas_path).filter(pl.col("date") == target_date)
barrids = alphas["barrid"].to_list()
all_exposures = sfd.load_exposures_by_date(date_=target_date)

# Filter to in universe and fill all 'null' values with 0.0 so math works
exposures = all_exposures.filter(pl.col("barrid").is_in(barrids)).fill_null(0.0)

# Industry mapping with barra exposures
potential_industry_cols = [
    col for col in exposures.columns 
    if col not in ["barrid", "date"] and exposures[col].max() == 1.0 and exposures[col].min() == 0.0
]

# Unpivot the binary matrix to create a 1-to-1 mapping
industry_mapping = (
    exposures.unpivot(
        index="barrid", 
        on=potential_industry_cols, 
        variable_name="industry", 
        value_name="is_member"
    )
    .filter(pl.col("is_member") == 1.0)
    .with_columns(
        pl.col("industry").str.replace("USSLOWL_", "").str.to_titlecase()
    )
    .select(["barrid", "industry"])
)

# Merge alphas + industry mapping + metadata
case_study_df = (
    alphas.join(industry_mapping, on="barrid", how="left")
    .join(sfd.load_assets_by_date(date_=target_date, columns=["barrid", "ticker", "name"], in_universe=True), on="barrid")
)


# Generate ex ante tables
# Biotech case study 
biotech_case_studies = (
    case_study_df.filter(pl.col("industry").str.contains("(?i)biolife|pharm|hlth"))
    .sort("alpha", descending=True)
    .head(10)
    .select(["barrid", "ticker", "industry", "alpha"])
)

save_table_image(
    pl_df=biotech_case_studies.drop("barrid"), 
    title="Top 10 Overbought Biotech & Pharma Scores", 
    filename=results_dir / "top_10_biotech.png"
)

# Top 20 overall
top_20_overall = (
    case_study_df
    .drop_nulls(subset=["industry"])
    .sort("alpha", descending=True)
    .head(20)
    .select(["barrid", "ticker", "industry", "alpha"])
)

save_table_image(
    pl_df=top_20_overall.drop("barrid"), 
    title="Top 20 Oversold Scores (Overall)", 
    filename=results_dir / "top_20_overall.png"
)


# Ex-post forward returns
# Month following signal date for analysis period
start_jan = dt.date(2025, 1, 1)
end_jan = dt.date(2025, 1, 21)

# Combine barrids to query database once
target_barrids = list(set(top_20_overall["barrid"].to_list() + biotech_case_studies["barrid"].to_list()))

# Pull returns for combined basket
realized_performance = (
    sfd.load_assets(
        start=start_jan,
        end=end_jan,
        columns=["barrid", "date", "return"]
    )
    .filter(pl.col("barrid").is_in(target_barrids))
    .with_columns((pl.col("return").truediv(100)).alias("return"))
    .drop_nulls(subset=["return"])
)

# Calculate cumulative return for each ticker
cumulative_jan_ret = (
    realized_performance
    .group_by("barrid")
    .agg(((pl.col("return") + 1).product() - 1).alias("realized_21d_return"))
)

# Overall Bridge Table and Hit Rate
bridge_table_overall = top_20_overall.join(cumulative_jan_ret, on="barrid", how="left")
hits_overall = (bridge_table_overall["realized_21d_return"] > 0).sum()
total_overall = len(bridge_table_overall)
hit_rate_overall = hits_overall / total_overall if total_overall > 0 else 0
footer_overall = f"Overall Hit Rate: {hit_rate_overall:.0%} ({hits_overall}/{total_overall} successful reversals)"

save_table_image(
    pl_df=bridge_table_overall.select(["ticker", "industry", "alpha", "realized_21d_return"]), 
    title="Realized 21-Day Returns for Top 20 Longs", 
    filename=results_dir / "realized_returns_top_20.png",
    footer_text=footer_overall
)

# Biotech Bridge Table and Hit Rate
bridge_table_bio = biotech_case_studies.join(cumulative_jan_ret, on="barrid", how="left")
hits_bio = (bridge_table_bio["realized_21d_return"] > 0).sum()
total_bio = len(bridge_table_bio)
hit_rate_bio = hits_bio / total_bio if total_bio > 0 else 0
footer_bio = f"Biotech Hit Rate: {hit_rate_bio:.0%} ({hits_bio}/{total_bio} successful reversals)"

save_table_image(
    pl_df=bridge_table_bio.select(["ticker", "industry", "alpha", "realized_21d_return"]), 
    title="Realized 21-Day Returns for Top 10 Biotech Longs", 
    filename=results_dir / "realized_returns_top_10_biotech.png",
    footer_text=footer_bio
)