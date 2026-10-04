# AAPL Next-Day Forecasting, Model Comparison & Profitability Back-test

Forecasting Apple Inc. (NASDAQ: **AAPL**) next-day returns with six models, turning the forecasts into a long-only trading strategy, and back-testing it against Buy & Hold and an SMA 50/200 crossover on an untouched 2023–2026 test period. The repo also includes a week-by-week forward forecast for October 2026.

> **Disclaimer:** This is an academic back-test on historical data. Nothing here is financial or investment advice.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Methodology](#methodology)
- [October 2026 Forecast](#october-2026-forecast)
- [Figures](#figures)
- [Limitations](#limitations)

---

## Project Overview

| Item | Choice |
|---|---|
| Domain | Stock market analysis, prediction and profitability |
| Instrument | Apple Inc. (AAPL) |
| Data | Yahoo Finance via `yfinance`, 22 Sep 1986 to 11 Sep 2026 (10,070 daily records) |
| Forecast level | Daily, 1 trading day ahead |
| Target | Next-day log return `r(t+1) = ln(C(t+1) / C(t))`, converted back to a closing price `C(t) * exp(r_hat)` |
| Features | 23 engineered features (returns, lags, SMA gaps, MACD, RSI, volatility, ranges, volume) |
| Models | Naive random walk, ARIMA(1,0,1), Ridge, Random Forest, XGBoost, SVR (RBF), Ensemble (RF + XGB + ARIMA) |
| Validation | Chronological split + yearly walk-forward retraining on the test period |

**Why predict returns instead of prices?** Prices are non-stationary and range from about $0.10 to $330+. Tree models cannot extrapolate beyond the range seen in training, and a price-level regression mostly learns "tomorrow is about today". Log returns are approximately stationary (ADF test, p < 1e-29), so a model trained on 2000–2019 still applies in 2023–2026.

---

## Key Results

### Forecast accuracy (test set, 925 trading days)

| Model | MAE ($) | RMSE ($) | MAPE (%) | R² (price) | R² (return) | Direction acc. (%) |
|---|---|---|---|---|---|---|
| Naive (Random Walk) | 2.462 | 3.631 | 1.127 | 0.9939 | -0.0044 | n/a |
| Ridge | 2.514 | 3.683 | 1.150 | 0.9937 | -0.0243 | 50.4 |
| Random Forest | 2.465 | 3.641 | 1.129 | 0.9938 | -0.0089 | 53.4 |
| XGBoost | 2.465 | 3.636 | 1.129 | 0.9938 | 0.0003 | 53.4 |
| SVR | 2.653 | 3.862 | 1.214 | 0.9931 | -0.1173 | 52.1 |
| ARIMA(1,0,1) | 2.458 | 3.638 | 1.125 | 0.9938 | -0.0060 | 53.3 |
| **Ensemble** | **2.455** | **3.628** | **1.124** | **0.9939** | **0.0024** | **53.7** |

Price-level R² of about 0.99 is **not** evidence of skill, because the naive random walk scores the same. The meaningful measures are R² on returns (about 0 for every model) and directional accuracy (barely above 50%).

### Trading back-test (test set, $100,000 start, 0.1% cost per trade)

| Strategy | τ (%) | Final ($) | Return (%) | Ann. return (%) | Sharpe | Max drawdown (%) | Trades | Win rate (%) |
|---|---|---|---|---|---|---|---|---|
| **Buy & Hold** | n/a | 265,505 | 165.5 | 30.5 | 1.16 | -33.4 | 1 | n/a |
| SMA 50/200 crossover | n/a | 129,037 | 29.0 | 7.2 | 0.44 | -30.0 | 5 | 50.0 |
| Ridge | 0.30 | 206,347 | 106.3 | 21.8 | 0.95 | -28.4 | 23 | 54.5 |
| Random Forest | 0.05 | 247,705 | 147.7 | 28.0 | 1.11 | -31.9 | 19 | 77.8 |
| XGBoost | 0.10 | 215,860 | 115.9 | 23.3 | 0.97 | -28.6 | 35 | 52.9 |
| SVR | 0.30 | 185,218 | 85.2 | 18.3 | 0.87 | -25.1 | 179 | 60.7 |
| ARIMA(1,0,1) | 0.00 | 180,910 | 80.9 | 17.5 | 0.77 | -36.1 | 88 | 68.2 |
| **Ensemble (main model)** | 0.00 | 231,100 | 131.1 | 25.6 | 1.03 | -31.9 | 39 | 68.4 |

### Ensemble profit by period

| Period | # Periods | Profitable | Loss-making | % Profitable |
|---|---|---|---|---|
| Daily | 925 | 484 | 426 | 52.3 |
| Weekly | 193 | 115 | 77 | 59.6 |
| Monthly | 45 | 29 | 16 | 64.4 |

| Year | Ensemble return (%) | Buy & Hold return (%) | Ensemble profit ($) |
|---|---|---|---|
| 2023 | 56.06 | 54.64 | 56,061 |
| 2024 | 24.86 | 30.71 | 38,799 |
| 2025 | 5.25 | 9.05 | 10,230 |
| 2026 (to 10 Sep) | 12.68 | 20.45 | 26,010 |

### Takeaways
1. **Next-day AAPL returns are almost unpredictable** from past prices and volume. Learned models are at best marginally better than a random walk, which is consistent with the weak-form efficient market hypothesis.
2. **The Ensemble was profitable in every test year**, but **Buy & Hold had the highest total return** because the model strategies sit in cash during part of a strongly rising market.
3. The timing strategies' real benefit is a **shallower drawdown** in some cases (for example SVR at -25.1% against -33.4% for Buy & Hold), at the cost of lower returns.
4. The strategy that looked best on validation was not necessarily best on test, a typical sign of how noisy daily stock signals are.

---

## Repository Structure

```
.
├── README.md
├── Apple_Forecast.ipynb                 # Main notebook: features, models, back-test, Oct 2026 forecast
├── data/
│   └── Apple_daily.csv                  # Daily OHLCV from Yahoo Finance (1986-09-22 to 2026-09-11)
├── figures/
│   ├── actual_vs_predicted.png
│   ├── equity_curves.png
│   ├── feature_importance.png
│   ├── profit_by_period.png
│   └── trade_signals.png
├── results/
│   ├── backtest_results_test.csv        # Strategy performance on the test set
│   ├── forecast_metrics_test.csv        # Forecast error metrics on the test set
│   ├── profit_daily_ensemble.csv
│   ├── profit_weekly_ensemble.csv
│   ├── profit_monthly_ensemble.csv
│   ├── profit_yearly_ensemble.csv
│   ├── AAPL_october_2026_daily_forecast.csv
│   └── AAPL_october_2026_weekly_forecast.csv
└── report/
    └── AAPL_Forecasting_Report.docx     # Full written project report
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create an environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn xgboost statsmodels jupyter
```

### 3. Point the notebook at the data
The notebook was written on Kaggle and locates the data with a `/kaggle/input/...` search. To run it locally, replace that cell with:

```python
import pandas as pd

DATA_PATH = "data/Apple_daily.csv"

raw = pd.read_csv(DATA_PATH)
raw["Date"] = (
    pd.to_datetime(raw["Date"], utc=True)
      .dt.tz_convert(None)
      .dt.normalize()
)
raw = raw.sort_values("Date").set_index("Date")
```

### 4. Run
```bash
jupyter notebook Apple_Forecast.ipynb
```
Run all cells top to bottom. The notebook saves figures to `figures/` and writes the result CSVs to the working directory (move them into `results/` if you want the layout above).

### Refreshing the data (optional)
```python
import yfinance as yf   # pip install yfinance
yf.Ticker("AAPL").history(period="max").to_csv("data/Apple_daily.csv")
```

---

## Methodology

### Data split (chronological, no shuffling)

| Set | Period | Rows | Purpose |
|---|---|---|---|
| Train | 2000-01-03 to 2019-12-31 | 5,031 | Fit models |
| Validation | 2020-01-02 to 2022-12-30 | 756 | Compare models, tune trading threshold τ |
| Test | 2023-01-03 to 2026-09-10 | 925 | Final out-of-sample evaluation and back-test |

Pre-2000 data is excluded from training: sub-$1 prices were quoted in coarse tick steps that distort returns, and the market regime was very different.

### Features (23, all computed from data up to day *t* only)
Returns (1/5/10/20-day), return lags (1/2/3/5/10), SMA gaps (5/10/20/50), MACD line/signal/diff, RSI(14), 10- and 20-day volatility, high-low and open-close ranges, volume change, and volume relative to its 20-day average. Price-level quantities are normalised by the current close.

### Models

| Model | Configuration |
|---|---|
| Naive | Predicted return of 0, so tomorrow's close equals today's close |
| ARIMA(1,0,1) | On log returns; fit on train, filtered through test with no leakage |
| Ridge | `StandardScaler` + `Ridge(alpha=10)` |
| Random Forest | 400 trees, `max_depth=6`, `min_samples_leaf=50` |
| XGBoost | 400 trees, `max_depth=3`, `learning_rate=0.02`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=20`, `reg_lambda=5` |
| SVR | RBF kernel, `C=0.1`, `epsilon=0.01`, trained on the latest 2,500 rows |
| Ensemble | Equal-weight mean of RF, XGBoost and ARIMA forecasts |

Every model is refit before each test year on all data up to 31 December of the previous year (yearly walk-forward retraining).

### Trading strategy
- Long-only, all-in / all-out, no leverage, $100,000 starting capital, **0.1% cost per trade**.
- **BUY** if not holding and forecast return > +τ. **SELL** if holding and forecast return < -τ. Otherwise **HOLD**.
- τ is chosen per model on the validation set from {0, 0.05%, 0.1%, 0.2%, 0.3%} and frozen for the test period.
- Baselines: **Buy & Hold** and **SMA 50/200 crossover**.

---

## October 2026 Forecast

The Ensemble is refit on all labelled data through **2026-09-11** (the last row of the dataset) and rolled forward recursively through 30 Oct 2026. Missing future OHLCV values are filled with simple assumptions: next-day open equals the previous close, the 20-day median intraday range sets high/low, and the 20-day median volume is carried forward.

| Week | Trading days | Start price ($) | End price ($) | Predicted return (%) | Projected portfolio ($) |
|---|---|---|---|---|---|
| Oct 01 - Oct 02 | 2 | 337.29 | 337.98 | 0.21 | 100,205 |
| Oct 05 - Oct 09 | 5 | 337.98 | 339.75 | 0.52 | 100,730 |
| Oct 12 - Oct 16 | 5 | 339.75 | 341.49 | 0.51 | 101,245 |
| Oct 19 - Oct 23 | 5 | 341.49 | 343.23 | 0.51 | 101,761 |
| Oct 26 - Oct 30 | 5 | 343.23 | 344.98 | 0.51 | 102,279 |

Base case: about **+2.28%** for October (about $2,279 on $100,000). This is a model estimate made as of 11 Sep 2026 with large uncertainty, and it should not be read as a reliable prediction.

---

## Figures

### Actual vs predicted next-day close
![Actual vs predicted](figures/actual_vs_predicted.png)

### Equity curves
![Equity curves](figures/equity_curves.png)

### Feature importance
![Feature importance](figures/feature_importance.png)

### Profit by period
![Profit by period](figures/profit_by_period.png)

### Trade signals
![Trade signals](figures/trade_signals.png)

---

## Limitations
- Execution at the same-day close is idealised and ignores slippage.
- Only one stock is analysed, and no fundamental, news or sentiment data is used.
- The October 2026 forecast uses synthetic future OHLCV values and recursive predictions, so errors compound over the horizon.
- Possible extensions: sentiment features, direct weekly/monthly models, up/down classification with probability-based position sizing, and LSTM/GRU sequence models.

---

*Academic back-test only; not financial advice.*
