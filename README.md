# Social Media Attention and Stock Mispricing

Code for the bachelor thesis *Social Media Attention and Stock Mispricing: Evidence from r/WallStreetBets* (BSc Economics & Business Economics, Financial Economics, Erasmus School of Economics).

The project tests whether coordinated retail attention on r/WallStreetBets drives stock prices away from fundamental value, and whether those prices subsequently correct. It uses a market-model event study (MacKinlay, 1997) to measure cumulative abnormal returns (CARs) over a short event window and buy-and-hold abnormal returns (BHARs) over a longer post-event window, and it adds a cross-sectional regression and a set of robustness checks.

## Method in brief

1. **Ticker selection.** All r/WallStreetBets post titles are scanned for dollar-sign mentions (e.g. `$GME`). A post counts once per ticker, and only the dollar-sign form is matched. Tickers with at least 500 mentions over the sample period are kept, and tokens that are not U.S. common equity in CRSP (ETFs, crypto, delisted or foreign issuers) are excluded. This yields a final universe of seven tickers: AMC, BB, GME, NOK, PLTR, SPCE, TSLA.
2. **Event identification.** A stock-day is an elevated-attention event if its mention count exceeds the rolling 90th percentile of that stock's own mentions over the prior 250 trading days (Hasso et al., 2022). Consecutive days are grouped into one event, and a minimum gap of 65 trading days is enforced.
3. **Abnormal returns.** For each event the market model is estimated by OLS over the window `[t-260, t-11]`, and CARs `[0, +5]` and BHARs `[+6, +60]` are computed relative to the CRSP value-weighted market return.
4. **Cross-sectional regression.** The event-level CARs and BHARs are regressed on event-day attention, pre-event volatility, and firm size, using heteroskedasticity-robust (White) standard errors.
5. **Robustness.** The short-run result is re-checked with a calendar-time portfolio (to remove cross-sectional dependence) and a bootstrap (to handle return skewness).

## Repository structure

| File | Description |
| --- | --- |
| `wsb_ticker_selection.py` | Builds the ticker universe from the Reddit data. |
| `wsb_pipeline_final.py` | Runs the full event study, cross-sectional regressions, and robustness checks, and exports the results and the event-time figure. |
| `requirements.txt` | Python dependencies. |

## Data

The data files are not included in this repository because of their size and licensing.

- **Reddit data:** `r_wallstreetbets_posts.csv` from the [Kaggle r/WallStreetBets dataset](https://www.kaggle.com/datasets/unanimad/reddit-rwallstreetbets) (Fontes, 2021).
- **Stock returns:** `crsp_returns.csv` — CRSP Daily Stock File via WRDS, with columns `ticker, date, ret, prc, shrout` for the seven tickers, 2019-01-01 to 2021-03-01.
- **Market return:** `crsp_market.csv` — CRSP Daily Stock File Indexes via WRDS, with the value-weighted market return `vwretd`.

Place all three CSV files in the repository root before running.

## Usage

```bash
pip install -r requirements.txt

python wsb_ticker_selection.py    # builds candidate_tickers.csv
python wsb_pipeline_final.py      # runs the event study and robustness checks
```

The pipeline writes `thesis_results.csv`, `thesis_results.xlsx`, `regression_results.csv`, `robustness_results.csv`, and `figure1_eventtime.png`.

## Main results

| Measure | Window | N | Mean | t-statistic | p-value |
| --- | --- | --- | --- | --- | --- |
| CAR | [0, +5] | 19 | +9.58% | +2.083 | 0.052 |
| BHAR | [+6, +60] | 14 | +15.56% | +1.266 | 0.228 |

The positive CAR is consistent with short-run, sentiment-driven price inflation, and it remains significant under a calendar-time portfolio (p = 0.053) and a bootstrap that accounts for return skewness (p = 0.032). The BHAR shows no significant reversal within the 60-day window under any method, which is consistent with short-squeeze dynamics sustaining elevated prices.

## References

- Fontes, R. (2021). *Reddit – r/WallStreetBets* [Dataset]. Kaggle.
- Hasso, T., Müller, D., Pelster, M., & Warkulat, S. (2022). Who participated in the GameStop frenzy? *Finance Research Letters, 45*, 102140.
- MacKinlay, A. C. (1997). Event studies in economics and finance. *Journal of Economic Literature, 35*(1), 13–39.
