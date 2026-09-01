# Research Report

**Bayesian Reversal**<br>
**Grant Rich & Josh Oldroyd**<br>

---

## 1. Summary

**Research/Development Direction:** Quantitative equity research focusing on mean-reversion alpha generation in the post-2020 market regime.
**Core Idea:** A Conditional Bayesian Model Averaging (BMA) reversal strategy. Traditional single-factor reversal models fail in the modern market due to the decoupling of market capitalization and liquidity. This system uses a "double-conditioned" approach—filtering price stretch (Alpha) by industry and active liquidity relative to size—to avoid microstructure traps.
**Key Conclusions:** The double-conditioned BMA model is an elite generator of short-alpha, particularly in high-beta/high-noise sectors (e.g., Biotech). It successfully identifies "blow-off tops" and statistical exhaustion. However, the long side remains vulnerable to "Value Traps" where extreme positive alpha often signals fundamental distress (a falling knife) rather than a temporary liquidity stretch.

### Key Metrics
*(Based on 12/31/2024 to 01/21/2025 Case Study)*

| Metric | Value | Notes |
|------|------|------|
| Biotech Short Hit Rate | 90% (9/10) | Exceptional at identifying exhausted momentum in high-noise names (e.g., IGMS -72.01%). |
| Overall Short Hit Rate | 70% (14/20) | Consistently fades unsupported retail/momentum stretches across the broader universe. |
| Overall Long Hit Rate | 50% (10/20) | Shows the difficulty of the long book; highly susceptible to fundamental regime shifts. |
| Biotech Long Hit Rate | 30% (3/10) | Severe "Value Trap" risk. (e.g., BLUE yielded -2.40% despite a massive 2.00 Alpha). |

---

## 2. Data Requirements

**Sources**
- Barra data

**Rate of Availability**
- Daily (EOD).

**Inputs Required**
- Daily Close Price
- Daily Trading Volume
- Market Capitalization
- Industry Group Designation

**Preprocessing**
- Calculation of baseline reversal signal (Alpha) over a standard trailing window (e.g., 21-day).
- Bayesian Model Averaging (BMA) to establish consensus conviction on the price "stretch."
- Normalization of active trading volume relative to the asset's market capitalization.

---

## 3. Approach / System Design

**Economic Intuition:**
The post-2020 market regime is characterized by high interest rate volatility and retail-driven reflexivity. In this environment, size and liquidity have decoupled. Single-factor reversal signals fail because they fall for two specific traps:
1.  **"Liquidity Holes" (Large-Cap Trap):** Large-cap stocks that gap down on thin volume. Simple models see a mean-reversion opportunity, but the lack of active liquidity means there is no energy for a "snap-back."
2.  **"Liquid Monsters" (Small-Cap Trap):** Small-cap stocks that rip on hyper-liquid retail FOMO. Simple models short the parabolic move, resulting in a catastrophic short squeeze.

**System Design:**
We built a **Double-Conditioned BMA Model**. 
- **Step 1 (The Stretch):** The BMA calculates the raw Alpha to identify overbought/oversold candidates based on price deviation.
- **Step 2 (The Filter):** The signal is conditioned on active liquidity relative to market capitalization. It lowers conviction on Large-Cap "Liquidity Holes" (gap downs on thin volume) and avoids shorting Small-Cap "Liquid Monsters" by recognizing hyper-volume as a momentum breakout rather than a mean-reverting stretch.

---

## 4. Code Structure

The project is organized to separate core mathematical utilities from specific experiment implementations and outputs. Dependency management is handled via `uv`.

```text
sf-research-bayesian-reversal/
├── research/
│   ├── experiments/
│   │   ├── bma_reversal/             # Standard BMA implementation
│   │   ├── conditional_bma_reversal/ # Double-conditioned BMA implementation
│   │   └── utils/                    # Core mathematical and logic modules
│   │       ├── backtest.py           # Historical backtesting engine
│   │       ├── bma.py                # Bayesian Model Averaging calculations
│   │       └── mvo.py                # Mean-Variance Optimization logic
│   └── results/
│       ├── bma_reversal/             # Outputs for standard BMA
│       └── conditional_bma_reversal/ # Outputs for conditional BMA
├── pyproject.toml                    # uv project configuration
├── uv.lock                           # uv dependency lockfile
├── README.md                         # Project setup
└── REPORT.md                         # Research documentation
```

---

## 5. Results / Evaluation

The most successful signal has been the enhanced (double conditional) BMA reversal conditioned on market cap and liquidity. This signal is still subject to fundamental failings within industries such Biotech, something that needs to be addressed. 

## 6. Performance Discussion

Strengths: Elite predictive power on the short side, specifically in high-beta/high-volatility industries like Biotech. The double-conditioning successfully filters out "Liquid Monsters," saving the portfolio from retail-driven short squeezes.

Weaknesses: Poor predictive power on the long side. Catching a "falling knife" is mathematically difficult in the current macro regime.

Sensitivity: The model is highly sensitive to the scaling parameters used to define "normal" liquidity relative to market cap. Rapid macro shifts (e.g., sudden Fed rate changes) can temporarily disguise fundamental shifts as technical noise.

## 7. Limitations

Known issues: The long book suffers from "Value Traps." A massively beaten-down stock often reflects legitimate bankruptcy or clinical failure risk, which the raw math interprets as a "discount to mean."

Missing features: The signal currently lacks a fundamental or credit-risk overlay for the long side.

Risks: Continued institutionalization of complex options strategies (e.g., 0DTEs) could alter the definition of "active liquidity," requiring recalibration of the volume condition.

## 8. Future Work

Fundamental Filter Integration: Develop a secondary credit-risk or fundamental overlay specifically for positive Alpha (Long) signals to differentiate between temporary oversold conditions and permanent distress.

Dynamic Liquidity Thresholds: Optimize the liquidity/volume thresholds dynamically based on industry groupings rather than a static universe-wide calculation.

Regime-Specific Backtesting: Run the exact pipeline through the 2012-2020 ZIRP environment to formally quantify the factor decay of standard reversal versus the BMA signal.