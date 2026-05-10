# StockVision_AI

Nexus is an end-to-end Python ML workflow forecasting AI stock trends from 11k+ records. It generates 40+ technical indicators and uses Mutual Information to train a Random Forest model for next-day predictions. The automated system scales easily.

## 🗂️ Project Structure

```
daproj.ipynb        # Main analysis notebook
raw_df.csv          # Raw stock price dataset (required input)
```

---

## 🔄 Pipeline Overview

The notebook is organized into three phases:

### Phase 1 — Data Cleaning & Ingestion
- Loads raw CSV stock data (`raw_df.csv`)
- Parses and normalizes the `Date` column to timezone-naive `YYYY-MM-DD`
- Drops unnecessary columns (e.g., `Stock Splits`)
- Validates data integrity (checks for `High < Low` and negative volume)

### Phase 2 — Feature Engineering
- Sorts data by `Ticker` and `Date`, grouped per company
- Adds time context features: `day_of_week`, `month`, `quarter`, `year`
- Computes a `daily_return` and flags dividend days (`is_dividend_day`)
- Builds rolling technical indicators:
  - **SMA 20 / SMA 50 / SMA 200** — short, medium, and long-term trend
  - **Bollinger Bands** — upper/lower bounds for detecting overbought/oversold conditions
  - **20-day Standard Deviation** — price volatility
  - **SMA Ratio** — momentum signal (fast vs. slow average)
  - **Bollinger Band Position** — normalized price location within the band
  - **Historical Volatility** — rolling std of daily returns
  - **Weekly & Monthly Returns** — `pct_change` over 5 and 21 trading days
  - **Daily Swing %** — intraday high-to-low range as a risk proxy
  - **Macro Trend Distance** — current price relative to the 200-day average
- Defines a binary classification **target**: will tomorrow's close be higher than today's?

### Phase 3 — Analytics & Investment Decision
- Plots **cumulative returns** across all tickers (base $1 investment)
- Generates a **feature correlation heatmap** against the prediction target
- Produces a per-stock **3-panel dashboard**:
  1. Long-term macro trend (price vs. SMA 200)
  2. Daily risk & violent intraday swings
  3. Historical monthly return distribution

---

## 🧰 Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

| Library       | Purpose                              |
|---------------|--------------------------------------|
| `numpy`       | Numerical computations               |
| `pandas`      | Data loading, cleaning, engineering  |
| `matplotlib`  | Plotting charts and dashboards       |
| `seaborn`     | Correlation heatmaps, histograms     |
| `scikit-learn`| (Available for modeling extension)   |

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Place your raw stock data file as `raw_df.csv` in the project root. The file should contain columns including: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Ticker`.

3. Launch the notebook:
   ```bash
   jupyter notebook daproj.ipynb
   ```

4. Run all cells top to bottom. Each phase builds on the previous one.

---

## 📊 Expected Dataset Format

| Column      | Type     | Description                        |
|-------------|----------|------------------------------------|
| `Date`      | datetime | Trading date (any timezone)        |
| `Open`      | float    | Opening price                      |
| `High`      | float    | Daily high price                   |
| `Low`       | float    | Daily low price                    |
| `Close`     | float    | Closing price                      |
| `Volume`    | int      | Number of shares traded            |
| `Dividends` | float    | Dividend amount (0 if none)        |
| `Ticker`    | string   | Stock symbol (e.g., `AMD`, `NVDA`) |

---

## 📌 Notes

- All rolling indicators are computed **per ticker** using `groupby` to prevent data leakage across companies.
- `NaN` rows (introduced by rolling windows and `shift`) are dropped before modeling.
- The notebook is designed to be extended — the engineered features are ready to be fed into a classifier (e.g., `RandomForestClassifier`, `XGBoost`).

---

## 📄 License

This project is for educational and research purposes.
