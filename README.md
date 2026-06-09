# Social Media Attention and Stock Mispricing: Evidence from r/WallStreetBets

Replication code and derived data for the bachelor thesis by Raul Hardeman
(Erasmus School of Economics, BSc Economics & Business Economics, June 2026).

The thesis runs a market-model event study (MacKinlay, 1997) on elevated-attention
events from r/WallStreetBets between January 2020 and February 2021, covering seven
tickers: AMC, BB, GME, NOK, PLTR, SPCE, TSLA.

## Repository contents

| File | Description |
| --- | --- |
| `wsb_ticker_selection.py` | Screens the ticker universe from the raw Reddit posts (dollar-sign mentions, at least 500, U.S. common equity only). Writes `candidate_tickers.csv`. |
| `wsb_event_study.py` | Builds daily mention counts, identifies elevated-attention events, runs the event study (CAR [0,+5], BHAR [+6,+60]), the robustness checks (calendar-time portfolio, bootstrap, excluding GameStop) and draws Figures 1 and 2. |
| `wsb_mentions_daily.csv` | Derived snapshot: distinct posts per ticker-day (901 rows, weekends included). |
| `wsb_events.csv` | Derived snapshot: the 21 identified events. |
| `candidate_tickers.csv` | Output of the universe screen, with exclusion reasons. |
| `requirements.txt` | Python dependencies. |

## Data you need to obtain yourself

**1. Raw Reddit posts (optional).** The raw file `r_wallstreetbets_posts.csv` comes
from the Kaggle dataset "Reddit WallStreetBets Posts" (Fontes, 2021). It is about
220 MB and holds 1.12 million posts, of which 864,273 fall in the sample window.
It is only needed to rebuild the mention counts from scratch or to rerun the
universe screen. The event study runs without it: if the raw file is absent,
`wsb_event_study.py` loads the included snapshot `wsb_mentions_daily.csv` instead.
Both routes give identical results.

**2. CRSP return data (required).** The CRSP files are licensed through WRDS and are
not redistributed in this repository. Download them yourself with a WRDS account:

* `crsp_returns.csv`: CRSP Daily Stock File, 2019-01-01 to 2021-03-01, for the
  tickers AMC, BB, GME, IPOA, NOK, PLTR, SPCE and TSLA, with at least the columns
  `date`, `TICKER` and `RET`. SPCE traded as IPOA before its 2019 merger; the code
  renames IPOA to SPCE.
* `crsp_market.csv`: CRSP Index File (daily), same period, with the column `vwretd`
  (value-weighted market return including dividends).

Column names are lowercased by the code, so capitalisation does not matter.
**Do not commit the CRSP files to a public repository.** A suitable `.gitignore`:

```
r_wallstreetbets_posts.csv
crsp_returns.csv
crsp_market.csv
__pycache__/
```

## How to run

```
pip install -r requirements.txt
python wsb_ticker_selection.py   # optional, needs the raw Kaggle file
python wsb_event_study.py        # uses the raw file if present, else the snapshot
```

`wsb_event_study.py` prints Tables 1 to 5 and writes `thesis_results.csv` (the
per-event results), `figure1_eventtime.png`, `figure2_decomposition.png` and
refreshed copies of the two snapshot CSVs.

## Reproducibility notes

* **End-to-end check.** Rebuilding everything from the raw Kaggle file reproduces
  `candidate_tickers.csv`, `wsb_mentions_daily.csv`, `wsb_events.csv` and
  `thesis_results.csv` exactly, byte for byte.
* **Two mention totals.** `wsb_ticker_selection.py` counts mention occurrences
  (GME: 15,679). The event study counts distinct posts per ticker-day
  (GME: 15,344). Both are correct; they answer different questions. The thesis
  reports the post counts.
* **Weekday grid.** Weekend mentions are present in `wsb_mentions_daily.csv` and
  in the reported totals, but events are identified on a Monday to Friday grid,
  so weekend mentions do not enter the threshold or the event definition. When a
  flagged day falls on a market holiday, the first trading day after it serves
  as event day 0.
* **Gap rule.** The minimum spacing of 65 days between same-ticker events is
  measured in weekdays, which include market holidays.
* **Seed.** The bootstrap uses a fixed seed (42), so the reported p-values and
  confidence intervals reproduce exactly.
* **Verified environment.** Results were verified with numpy 2.4.4, pandas 3.0.2,
  scipy 1.17.1 and matplotlib 3.10.8. Other recent versions should work.
* **Headline numbers to expect.** Mean CAR [0,+5]: +9.58% (t = 2.083, p = 0.052),
  excluding GME: +12.81% (p = 0.032). Mean BHAR [+6,+60]: +15.56% (p = 0.228).
  Sample: 19 CAR events, 14 BHAR events.
