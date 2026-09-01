# Silver Fund Bayesian Reversal Repository

## Set Up

Set up your Python virtual environment using `uv`.
```bash
uv sync
```

Source your Python virtual environment.
```bash
source .venv/bin/activate
```

Set up your environment variables in a `.env` file. You can follow the example found in `.env.example`.

Set up pre-commit by running:
```bash
prek install
```

Now all of your files will be formatted on commit (you will need to re-commit after the formatting).

## Experiments
## BMA Reversal
1. Baseline BMA Reversal (3-Horizons)
2. Dynamic Weighting and Recency Decay
3. Hierarchical Bayesian Update (Prior vs. Likelihood)
4. Bayesian Shrinkage via the Null Model
5. Information Filtering (Volume Inhibition)
6. Winsorized Volume-Inhibited BMA
7. Alpha Smoothing (EWMA Decay)
8. Signal Acceleration & Risk Scaling
9. Correlation and Bivariate Regression Test (BMA Rev. vs Enhanced Rev.)
10. Sample Portfolio Active Risk & Attribution

## Conditional BMA Reversal
1. Market Cap Conditional BMA
2. Hierarchical Bayesian Shrinkage BMA
3. Volatility Adjusted BMA
4. Adaptive Volume Inhibition BMA
5. Liquidity Conditional BMA
6. Double Conditional BMA
7. Enhanced Double Conditional BMA
8. Orthogonality Test (Enhanced BMA Alphas regressed on Vol Conditioned Barra alphas)
9. Enhanced BMA Sample Portfolio
10. Equal-Weight Average HL's
11. Orthogonality Test (Vol Conditioned Barra Alphas regressed on Enhanced BMA Alphas)
12. Standard Reversal Sample Portfolio
13. Vol Conditioned Barra Sample Portfolio
14. Negative Alphas Case Study
15. Positive Alphas Case Study