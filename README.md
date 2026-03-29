# VaR-Edelweiss-Flexicap-Mutual-Fund-Project
This repository acts as a code storage to my independent VaR (Value at Risk) Project, required in the field of Risk Management and Broader Finance.

---

## Overview

This project implements and compares three industry-standard Value at Risk (VaR) methodologies on the Edelweiss Flexi Cap Fund — a real, publicly disclosed Indian mutual fund portfolio. The analysis covers VaR estimation, model backtesting against 2024 market data, stress testing under historical crisis scenarios, and CVaR (Expected Shortfall) calculation.

The project is structured as a working paper and is intended to demonstrate practical risk management skills in a real-world Indian equity context.

---

## Key Findings

| Method | 1-Day 95% VaR | CVaR |
|---|---|---|
| Historical Simulation | -1.41% (₹28,200) | -2.28% (₹45,600) |
| Parametric (Normal) | -1.47% (₹29,353) | -1.85% (₹37,000) |
| Monte Carlo (10,000 sims) | -1.48% (₹29,600) | -1.86% (₹37,200) |

*Based on a hypothetical portfolio value of ₹20,00,000*

**Backtesting (2024):** 15 exceptions over 246 trading days (6.1%) — Basel Red Zone  
**Worst day identified:** June 4, 2024 (Indian General Election results) — single day loss of -6.29%, representing 4.7x the VaR estimate

---

## Methodology

### Portfolio
- Fund: Edelweiss Flexi Cap Fund (top 25 holdings)
- Holdings source: AMC monthly portfolio disclosure (AMFI)
- Data: 3 years of daily NSE closing prices (2022–2025) via Yahoo Finance
- Portfolio value assumed: ₹20,00,000 (hypothetical)

### VaR Methods

**1. Historical Simulation**
Uses actual historical returns directly. Finds the 5th percentile of the empirical return distribution without assuming any particular distribution shape.

**2. Parametric (Variance-Covariance)**
Assumes returns follow a normal distribution. VaR is derived analytically from the portfolio mean and standard deviation using the formula:
```
VaR = norm.ppf(1 - confidence, mean, std) × Portfolio Value
```

**3. Monte Carlo Simulation**
Generates 10,000 hypothetical daily returns using the portfolio's statistical parameters. VaR is estimated from the simulated distribution. Includes a 252-day path simulation showing the range of portfolio outcomes over one year.

### Backtesting
Model trained on 2022–2023 data and tested against 2024 actual returns. Exceptions (days where actual losses exceeded VaR) counted and evaluated against Basel II Traffic Light standards.

### Stress Testing
Portfolio shocked with three historical crisis scenarios:
- COVID Crash (March 23, 2020): -13.13%
- Indian General Election Shock (June 4, 2024): -6.29%
- 2008 Global Financial Crisis (October 2008): -11.96%

### CVaR (Expected Shortfall)
Calculated as the average loss conditional on returns falling below the VaR threshold — answering the question VaR cannot: *"When it breaks, how bad does it actually get?"*

---

## Repository Structure

```
var-analysis-edelweiss-flexi-cap/
│
├── README.md                  ← This file
├── var_analysis.py            ← Full Python script
├── requirements.txt           ← Required libraries
├── data/
│   ├── Edelweiss_Flexi_Cap_Fund_.xlsx   ← Portfolio holdings (source)
│   └── edelweiss_prices.csv             ← Historical price data
└── charts/
    ├── 01_historical_var.png
    ├── 02_parametric_var.png
    ├── 03_montecarlo_distribution.png
    ├── 04_montecarlo_paths.png
    ├── 05_var_comparison.png
    ├── 06_backtesting_2024.png
    ├── 07_stress_testing.png
    └── 08_cvar.png
```

---

## Requirements

```
yfinance
pandas
numpy
scipy
matplotlib
openpyxl
```

Install all dependencies:
```bash
pip install yfinance pandas numpy scipy matplotlib openpyxl
```

---

## How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/var-analysis-edelweiss-flexi-cap.git
cd var-analysis-edelweiss-flexi-cap
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the analysis
```bash
python var_analysis.py
```

All charts will be generated and displayed sequentially. Results will be printed to console.

---

## Results Summary

### All Three VaR Methods Converge
The narrow spread across methods (-1.41% to -1.48%) indicates the Edelweiss Flexi Cap Fund's return distribution closely approximates normality under normal market conditions. This convergence suggests low model risk for day-to-day risk management.

### Fat Tails in Actual Returns
The actual return distribution exhibits excess kurtosis — extreme loss events occur more frequently than the normal distribution predicts. This is visible in the parametric VaR comparison chart where actual returns extend further left than the theoretical normal curve.

### Backtesting — Basel Red Zone
The model produced 15 exceptions in 2024 against an expected 12–13, placing it in Basel's Red Zone. This reflects the model being calibrated on a relatively calm 2022–2023 period and encountering elevated volatility in 2024, particularly the June 4th election shock.

### Stress Testing — VaR Vastly Underestimates Crisis Losses
Under all three crisis scenarios, losses ranged from 4.7x to 9.7x the VaR estimate — underscoring that VaR is a measure of normal market risk, not crisis risk. Stress testing must complement VaR in any robust risk framework.

### CVaR Reveals Hidden Tail Depth
The CVaR of -2.28% (Historical) versus VaR of -1.41% reveals that when the threshold is breached, average losses are 62% larger than the VaR itself. This supports Basel III's preference for CVaR as the primary regulatory risk measure.

---

## Limitations

- Analysis uses top 25 holdings (covering ~56% of fund AUM). Remaining holdings are excluded and weights are normalised.
- Portfolio value is hypothetical (₹20,00,000). Actual fund AUM is significantly larger.
- Monte Carlo simulation assumes normally distributed returns — subject to the same fat-tail limitations as the Parametric method.
- Historical Simulation is backward-looking and cannot anticipate structural breaks or novel events.
- Correlation structure between stocks is implicitly captured through historical returns but is not modelled explicitly.

---

## References

- Basel Committee on Banking Supervision (2019). *Minimum Capital Requirements for Market Risk.*
- Jorion, P. (2006). *Value at Risk: The New Benchmark for Managing Financial Risk.* McGraw-Hill.
- AMFI India — Edelweiss Flexi Cap Fund Monthly Portfolio Disclosure
- NSE India — Historical price data via Yahoo Finance API

---

## Author

**Aritra Tarafdar**  
BA Honours (English - Major + Economics - Minor)   

[LinkedIn](#) | [SSRN Working Paper](#)

---

## License

MIT License — free to use, adapt, and build upon with attribution.
