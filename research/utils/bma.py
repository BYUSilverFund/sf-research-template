import os
from itertools import combinations
import numpy as np
import polars as pl

def get_bics_stable(df_X, y, combos):
    n = len(df_X)
    bics = []
    params_list = []
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
        params_list.append(dict(zip(X_cols, beta)))
    return np.array(bics), params_list

def run_conditional_bma_loop(
    timing_ready_df, 
    reversal_factors, 
    unique_dates, 
    unique_buckets, 
    dynamic_window_months, 
    checkpoint_dir,
    null_priors=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_combos = [()]
    for k in range(1, len(reversal_factors) + 1):
        all_combos.extend(list(combinations(reversal_factors, k)))

    rolling_results = []

    for i in range(dynamic_window_months, len(unique_dates)):
        current_date = unique_dates[i]
        checkpoint_path = os.path.join(checkpoint_dir, f"{current_date}.parquet")

        if os.path.exists(checkpoint_path):
            saved_forecasts = pl.read_parquet(checkpoint_path).with_columns(pl.col("size_bucket").cast(pl.String)).to_dicts()
            rolling_results.extend(saved_forecasts)
            continue

        train_dates = unique_dates[i - dynamic_window_months : i]
        date_forecasts = []

        for bucket in unique_buckets:
            df_train = timing_ready_df.filter(
                (pl.col("date").is_in(train_dates)) & 
                (pl.col("size_bucket") == bucket)
            ).with_columns(pl.lit(1.0).alias("const"))

            latest_X_df = timing_ready_df.filter(
                (pl.col("date") == current_date) & 
                (pl.col("size_bucket") == bucket)
            )
            
            if len(latest_X_df) == 0:
                continue
                
            latest_X = latest_X_df.row(0, named=True)
            bucket_forecasts = {"date": current_date, "size_bucket": str(bucket)}

            if null_priors and bucket in null_priors:
                p_null = null_priors[bucket]
            else:
                p_null = 1.0 / len(all_combos)
            
            p_other = (1.0 - p_null) / (len(all_combos) - 1)
            
            prior_array = np.array([
                p_null if len(subset) == 0 else p_other
                for subset in all_combos
            ])

            for target_factor in reversal_factors:
                target_col = f"{target_factor}_target"
                y_dyn = df_train.get_column(target_col)
                
                bics, params = get_bics_stable(df_train, y_dyn, all_combos)
                
                # Turn BICs into Likelihoods
                bics_adj = bics - np.min(bics)
                likelihoods = np.exp(-0.5 * bics_adj)
                
                # NEW: Calculate Posterior Probs = Likelihood * Prior
                unnormalized_probs = likelihoods * prior_array
                
                if unnormalized_probs.sum() == 0:
                    probs = prior_array
                else:
                    probs = unnormalized_probs / unnormalized_probs.sum()

                expected_return = 0.0
                for idx, subset in enumerate(all_combos):
                    m_params = params[idx]
                    model_forecast = m_params.get("const", 0.0)
                    for f in subset:
                        model_forecast += m_params[f] * latest_X[f]
                    expected_return += model_forecast * probs[idx]

                bucket_forecasts[target_factor] = expected_return
            
            date_forecasts.append(bucket_forecasts)

        if date_forecasts:
            pl.DataFrame(date_forecasts).write_parquet(checkpoint_path)
            rolling_results.extend(date_forecasts)

    return pl.DataFrame(rolling_results)