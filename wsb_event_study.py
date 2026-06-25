"""
Social Media Attention and Stock Mispricing: Evidence from r/WallStreetBets.
Bachelor thesis, Raul Hardeman (Erasmus School of Economics, 2026).

Market-model event study (MacKinlay, 1997) on elevated-attention events from
r/WallStreetBets between January 2020 and February 2021, for seven tickers:
AMC, BB, GME, NOK, PLTR, SPCE, TSLA.

The script runs the full analysis in one pass:

  1. Build daily ticker mention counts from the raw Reddit posts (or load the
     derived snapshot) and flag days above a rolling 90th-percentile threshold
     (Hasso et al., 2022) as elevated-attention events.
  2. Estimate the market model on a pre-event window and compute the cumulative
     abnormal return CAR[0, +5] and the buy-and-hold abnormal return BHAR[+6, +60]
     for every event.
  3. Report the descriptive statistics, the correlation matrix and the mean
     CARs and BHARs with one-sample t-tests (Tables 1-5).
  4. Test the short-run result against cross-sectional dependence (calendar-time
     portfolio, Fama 1998), return skewness (bootstrap, Lyon, Barber & Tsai 1999),
     and the normality assumption (standardised cross-sectional test, Boehmer et
     al. 1991; generalised sign test, Cowan 1992), and re-estimate the means over
     alternative event windows (Tables 6-8).
  5. Repeat the headline analysis on the sample excluding GameStop (Table 9).
  6. Draw Figure 1 (cumulative average abnormal return) and Figure 2 (per-ticker
     CAR and BHAR, with and without GameStop), and print the worked example
     reported in Appendix A.

Inputs:  r_wallstreetbets_posts.csv (raw Kaggle posts; see README) or, if that file
         is absent, the derived snapshot wsb_mentions_daily.csv;
         crsp_returns.csv and crsp_market.csv (CRSP via WRDS, not redistributed).
Outputs: thesis_results.csv (per-event results), wsb_mentions_daily.csv and
         wsb_events.csv (derived snapshots), figure1_eventtime.png and
         figure2_decomposition.png.
Usage:   python wsb_event_study.py
"""

import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Files
REDDIT_CSV     = "r_wallstreetbets_posts.csv"   # raw Kaggle posts (optional)
CRSP_CSV       = "crsp_returns.csv"             # CRSP daily stock file (WRDS)
MARKET_CSV     = "crsp_market.csv"              # CRSP value-weighted index (WRDS)
MENTIONS_CSV   = "wsb_mentions_daily.csv"       # derived snapshot / fallback input
EVENTS_CSV     = "wsb_events.csv"               # derived snapshot
RESULTS_CSV    = "thesis_results.csv"           # per-event results
FIG_EVENTTIME  = "figure1_eventtime.png"
FIG_DECOMP     = "figure2_decomposition.png"

# Sample and universe
SAMPLE_START = "2020-01-01"
SAMPLE_END   = "2021-02-16"
TICKERS       = ["AMC", "BB", "GME", "NOK", "PLTR", "SPCE", "TSLA"]
TICKER_RENAME = {"IPOA": "SPCE"}   # SPCE traded as IPOA in CRSP before its 2019 merger
DROP_TICKER   = "GME"              # excluded in the robustness check (Table 9)

# Event identification (Hasso et al., 2022)
PERCENTILE   = 0.90   # rolling percentile threshold
LOOKBACK     = 250    # lookback window (weekdays)
MIN_OBS      = 30     # min prior observations before computing the threshold
MIN_GAP_DAYS = 65     # min spacing between same-ticker events (weekdays)

# Market model (MacKinlay, 1997)
EST_WINDOW  = 250     # estimation window length (trading days)
EST_GAP     = 10      # gap between estimation window and event date
MIN_EST_OBS = 100     # min observations to estimate the model
CAR_END     = 5       # CAR window [0, +5]
BHAR_START  = 6       # BHAR window [+6, +60]
BHAR_END    = 60

# Inference
N_BOOTSTRAP = 10000   # bootstrap resamples
BOOT_SEED   = 42      # bootstrap RNG seed

# Alternative windows reported in Table 8
CAR_WINDOWS  = {"[0,+1]": (0, 1), "[0,+3]": (0, 3), "[0,+5] (baseline)": (0, 5),
                "[0,+10]": (0, 10), "[-1,+1]": (-1, 1), "[-5,+5]": (-5, 5)}
BHAR_WINDOWS = {"[+6,+30]": (6, 30), "[+6,+60] (baseline)": (6, 60),
                "[+6,+90]": (6, 90), "[+6,+120]": (6, 120), "[+21,+60]": (21, 60)}
EXAMPLE_TICKER = "BB"   # event used for the worked example in Appendix A



# Data loading and event identification
def build_mentions(reddit_csv):
    """Daily mentions per ticker = number of distinct posts naming it ($TICKER)."""
    df = pd.read_csv(reddit_csv, low_memory=False)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    df = df.dropna(subset=["date", "title"])

    ticker_re = re.compile(r"\$(" + "|".join(map(re.escape, TICKERS)) + r")\b")
    work = df[["date", "title"]].copy()
    work["day"] = work["date"].dt.normalize()
    work["tickers"] = work["title"].astype(str).str.upper().apply(
        lambda s: sorted({m.group(1) for m in ticker_re.finditer(s)})
    )
    work = work[work["tickers"].map(len) > 0].explode("tickers")
    mentions = (work.groupby(["day", "tickers"]).size()
                    .reset_index(name="mention_count")
                    .rename(columns={"day": "date", "tickers": "ticker"}))
    mentions["date"] = pd.to_datetime(mentions["date"])
    return mentions.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_mentions():
    """Build mentions from the raw Reddit file if present, else load the snapshot."""
    if Path(REDDIT_CSV).exists():
        mentions = build_mentions(REDDIT_CSV)
        mentions.to_csv(MENTIONS_CSV, index=False, date_format="%Y-%m-%d")
        print(f"Built mentions from {REDDIT_CSV}; snapshot saved to {MENTIONS_CSV}")
        return mentions
    if Path(MENTIONS_CSV).exists():
        mentions = pd.read_csv(MENTIONS_CSV, parse_dates=["date"])
        print(f"{REDDIT_CSV} not found; loaded snapshot {MENTIONS_CSV} "
              f"({len(mentions)} ticker-day rows)")
        return mentions.sort_values(["date", "ticker"]).reset_index(drop=True)
    sys.exit(f"Error: neither {REDDIT_CSV} nor {MENTIONS_CSV} found. See README.md.")


def identify_events(mentions):
    """Flag days above the rolling 90th percentile, then enforce the min-gap rule.

    shift(1) makes the threshold use only prior days, avoiding look-ahead bias.
    Only the first day of each cluster is kept, and same-ticker events must lie
    at least MIN_GAP_DAYS weekday rows apart.
    """
    all_dates = pd.bdate_range(mentions["date"].min(), mentions["date"].max())
    idx = pd.MultiIndex.from_product([all_dates, TICKERS], names=["date", "ticker"])
    panel = (mentions.set_index(["date", "ticker"])["mention_count"]
                     .reindex(idx, fill_value=0).reset_index())
    panel["date"] = pd.to_datetime(panel["date"])

    out = []
    for ticker in TICKERS:
        sub = panel[panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        sub["p90"] = (sub["mention_count"].shift(1)
                                          .rolling(LOOKBACK, min_periods=MIN_OBS)
                                          .quantile(PERCENTILE))
        above = sub["mention_count"] > sub["p90"]
        last_pos, flags = None, []
        for pos, is_high in enumerate(above):
            keep = bool(is_high) and (last_pos is None or pos - last_pos >= MIN_GAP_DAYS)
            flags.append(keep)
            if keep:
                last_pos = pos
        sub["elevated"] = np.array(flags, dtype=int)
        out.append(sub[sub["elevated"] == 1])

    events = pd.concat(out, ignore_index=True)
    events["event_id"] = range(1, len(events) + 1)
    return events


def load_crsp():
    """Load CRSP daily stock returns and the value-weighted market return."""
    for f in (CRSP_CSV, MARKET_CSV):
        if not Path(f).exists():
            sys.exit(f"Error: {f} not found. Download from WRDS (CRSP daily files).")

    crsp = pd.read_csv(CRSP_CSV, low_memory=False)
    crsp.columns = crsp.columns.str.lower().str.strip()
    crsp["date"] = pd.to_datetime(crsp["date"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp["ticker"] = crsp["ticker"].replace(TICKER_RENAME)
    crsp = crsp.dropna(subset=["ret", "date"])

    mkt = pd.read_csv(MARKET_CSV, low_memory=False)
    mkt.columns = mkt.columns.str.lower().str.strip()
    mkt["date"] = pd.to_datetime(mkt["date"])
    mkt = (mkt.drop_duplicates("date")[["date", "vwretd"]]
              .rename(columns={"vwretd": "mkt_ret"}))

    crsp = crsp.merge(mkt, on="date", how="left").dropna(subset=["ret", "mkt_ret"])
    return crsp.sort_values(["ticker", "date"]).reset_index(drop=True)

# Event study
def _locate(crsp, ticker, event_date):
    """Return (indexed stock frame, date list, row position of the event day)."""
    stock = crsp[crsp["ticker"] == ticker].set_index("date").sort_index()
    dates = list(stock.index)
    on_or_after = [d for d in dates if d >= pd.to_datetime(event_date)]
    pos = dates.index(on_or_after[0]) if on_or_after else None
    return stock, dates, pos


def _market_model(stock, pos):
    """OLS market model on [pos-EST_GAP-EST_WINDOW, pos-EST_GAP). Returns alpha, beta."""
    est_end, est_start = pos - EST_GAP, pos - EST_GAP - EST_WINDOW
    if est_start < 0 or est_end - est_start < MIN_EST_OBS:
        return None
    est = stock.iloc[est_start:est_end]
    if len(est) < MIN_EST_OBS:
        return None
    x = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
    (alpha, beta), *_ = np.linalg.lstsq(x, est["ret"].values, rcond=None)
    return float(alpha), float(beta), len(est)


def compute_results(events, crsp):
    """Estimate the market model per event and compute CAR[0,+5] and BHAR[+6,+60]."""
    rows = []
    for _, e in events.iterrows():
        ticker = e["ticker"]
        stock, dates, pos = _locate(crsp, ticker, e["date"])
        if pos is None:
            continue
        model = _market_model(stock, pos)
        if model is None or pos + CAR_END + 1 > len(dates):
            continue
        alpha, beta, n_est = model

        car_seg = stock.iloc[pos:pos + CAR_END + 1]
        car = float((car_seg["ret"] - (alpha + beta * car_seg["mkt_ret"])).sum())

        if pos + BHAR_END + 1 <= len(dates):
            post = stock.iloc[pos + BHAR_START:pos + BHAR_END + 1]
            bhar = float((1 + post["ret"]).prod() - (1 + post["mkt_ret"]).prod())
        else:
            bhar = np.nan

        vol_w = stock.iloc[max(0, pos - 60):pos]
        vol = float(vol_w["ret"].std()) if len(vol_w) >= 20 else np.nan

        rows.append({"event_id": e["event_id"], "ticker": ticker,
                     "event_date": dates[pos], "alpha": round(alpha, 6),
                     "beta": round(beta, 4), "n_est_days": n_est,
                     "car_0_5": round(car, 6),
                     "bhar_6_60": round(bhar, 6) if not np.isnan(bhar) else np.nan,
                     "volatility": round(vol, 6) if not np.isnan(vol) else np.nan})
    return pd.DataFrame(rows)


def car_over(row, crsp, start, end):
    """CAR over an arbitrary window [start, end] using the event's stored alpha/beta."""
    stock, dates, pos = _locate(crsp, row["ticker"], row["event_date"])
    if pos is None or pos + start < 0 or pos + end + 1 > len(dates):
        return np.nan
    seg = stock.iloc[pos + start:pos + end + 1]
    return float((seg["ret"] - (row["alpha"] + row["beta"] * seg["mkt_ret"])).sum())


def bhar_over(row, crsp, start, end):
    """BHAR over an arbitrary window [start, end]."""
    stock, dates, pos = _locate(crsp, row["ticker"], row["event_date"])
    if pos is None or pos + start < 0 or pos + end + 1 > len(dates):
        return np.nan
    seg = stock.iloc[pos + start:pos + end + 1]
    return float((1 + seg["ret"]).prod() - (1 + seg["mkt_ret"]).prod())

# Statistics
def ttest(series):
    """One-sample t-test against zero. Returns (N, mean, t, p) with NaNs if N < 2."""
    s = series.dropna()
    if len(s) < 2:
        return len(s), (float(s.mean()) if len(s) else np.nan), np.nan, np.nan
    t, p = stats.ttest_1samp(s, 0)
    return len(s), float(s.mean()), float(t), float(p)


def calendar_time_portfolio(res, crsp):
    """Mean daily AR of an equally weighted portfolio of stocks inside [0, +5].

    Events sharing a date are merged into one observation, so overlapping events
    are not double-counted (Fama, 1998).
    """
    recs = []
    for _, e in res.iterrows():
        stock, dates, pos = _locate(crsp, e["ticker"], e["event_date"])
        if pos is None:
            continue
        for d in range(0, CAR_END + 1):
            p = pos + d
            if 0 <= p < len(dates):
                r = stock.iloc[p]
                recs.append({"date": dates[p],
                             "ar": r["ret"] - (e["alpha"] + e["beta"] * r["mkt_ret"])})
    if not recs:
        return np.nan, np.nan, np.nan, 0
    port = pd.DataFrame(recs).groupby("date")["ar"].mean()
    t, p = stats.ttest_1samp(port, 0)
    return float(port.mean()), float(t), float(p), len(port)


def bootstrap_mean(series):
    """Two-sided bootstrap p-value and 95% percentile CI for the mean (Lyon et al., 1999)."""
    x = series.dropna().to_numpy()
    if len(x) < 2:
        return np.nan, np.nan, np.nan, len(x)
    m = x.mean()
    rng = np.random.default_rng(BOOT_SEED)
    null = rng.choice(x - m, size=(N_BOOTSTRAP, len(x))).mean(axis=1)
    p = float((np.abs(null) >= abs(m)).mean())
    boot = rng.choice(x, size=(N_BOOTSTRAP, len(x))).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(m), p, (float(lo), float(hi)), len(x)


def _standardised_car_and_sign(row, crsp):
    """Standardised CAR and sign(CAR) for one event (used in Tables for §5.3)."""
    stock, dates, pos = _locate(crsp, row["ticker"], row["event_date"])
    if pos is None:
        return np.nan, np.nan
    est_start = pos - EST_GAP - EST_WINDOW
    if est_start < 0:
        return np.nan, np.nan
    est = stock.iloc[est_start:pos - EST_GAP]
    sigma = (est["ret"] - (row["alpha"] + row["beta"] * est["mkt_ret"])).std(ddof=2)
    car = car_over(row, crsp, 0, CAR_END)
    se = sigma * np.sqrt(CAR_END + 1)
    scar = car / se if se > 0 else np.nan
    return scar, np.sign(car)


def standardised_cross_sectional(subset, crsp):
    """Standardised cross-sectional test (Boehmer, Musumeci & Poulsen, 1991)."""
    scars = np.array([s for s, _ in (_standardised_car_and_sign(r, crsp)
                                     for _, r in subset.iterrows()) if not np.isnan(s)])
    n = len(scars)
    if n < 2:
        return n, np.nan, np.nan
    t = scars.mean() / (scars.std(ddof=1) / np.sqrt(n))
    return n, float(t), float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))


def generalised_sign(subset, crsp):
    """Generalised sign test (Cowan, 1992), benchmarked on each event's own model."""
    signs, fracs = [], []
    for _, r in subset.iterrows():
        scar, sgn = _standardised_car_and_sign(r, crsp)
        if np.isnan(sgn):
            continue
        stock, dates, pos = _locate(crsp, r["ticker"], r["event_date"])
        est = stock.iloc[pos - EST_GAP - EST_WINDOW:pos - EST_GAP]
        est_ar = est["ret"] - (r["alpha"] + r["beta"] * est["mkt_ret"])
        signs.append(sgn)
        fracs.append(float((est_ar > 0).mean()))
    signs = np.array(signs)
    n = len(signs)
    if n < 2:
        return n, np.nan, np.nan, np.nan
    n_pos = int((signs > 0).sum())
    phat = np.mean(fracs)
    z = (n_pos - n * phat) / np.sqrt(n * phat * (1 - phat))
    return n, n_pos, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))

# Reporting
def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _print_ttest_row(label, series):
    n, mean, t, p = ttest(series)
    if np.isnan(t):
        print(f"  {label:<14} N={n:>2}  (insufficient)")
    else:
        print(f"  {label:<14} N={n:>2}  mean={mean*100:+7.2f}%  "
              f"t={t:+6.3f}{_stars(p):<3} p={p:.4f}")


def report_descriptives(res, mentions):
    print("\nTable 1 - Descriptive statistics of daily mention counts per ticker")
    desc = (mentions.groupby("ticker")["mention_count"]
            .agg(Days="count", Total="sum", Mean="mean", SD="std",
                 Min="min", Median="median", Max="max").reindex(TICKERS))
    print(desc.round(2).to_string())

    print("\nTable 2 - Descriptive statistics of the event sample")
    sample = pd.DataFrame({
        "CAR [0,+5] (%)": res["car_0_5"].dropna() * 100,
        "BHAR [+6,+60] (%)": res["bhar_6_60"].dropna() * 100,
        "Return Volatility": res["volatility"].dropna(),
    })
    stats_tbl = sample.agg(["count", "mean", "median", "std"]).T
    stats_tbl["P10"] = [sample[c].quantile(.10) for c in sample]
    stats_tbl["P90"] = [sample[c].quantile(.90) for c in sample]
    print(stats_tbl.round(4).to_string())

    print("\nTable 3 - Correlation matrix of event-level variables")
    cv = res[["car_0_5", "bhar_6_60", "volatility", "beta"]].copy()
    cv.columns = ["CAR[0,+5]", "BHAR[+6,+60]", "Volatility", "Beta"]
    print(cv.corr().round(3).to_string())


def report_mean_abnormal_returns(res):
    print("\nTable 4 - Mean CAR [0, +5] by ticker")
    _print_ttest_row("Full sample", res["car_0_5"])
    for t in TICKERS:
        _print_ttest_row(t, res.loc[res["ticker"] == t, "car_0_5"])

    print("\nTable 5 - Mean BHAR [+6, +60] by ticker")
    _print_ttest_row("Full sample", res["bhar_6_60"])
    for t in TICKERS:
        _print_ttest_row(t, res.loc[res["ticker"] == t, "bhar_6_60"])


def report_robustness(res, crsp, label):
    print(f"\nRobustness ({label}) - dependence- and skew-robust inference")
    m, t, p, n = calendar_time_portfolio(res, crsp)
    print(f"  Calendar-time portfolio [0,+{CAR_END}]: daily mean AR={m*100:+.3f}%  "
          f"t={t:+.3f}  p={p:.4f}  (obs={n})")
    for col, name in [("car_0_5", "CAR [0,+5]"), ("bhar_6_60", "BHAR [+6,+60]")]:
        m, p, ci, n = bootstrap_mean(res[col])
        print(f"  Bootstrap {name}: mean={m*100:+.2f}%  p={p:.4f}  "
              f"95% CI=[{ci[0]*100:+.2f}%, {ci[1]*100:+.2f}%]  (n={n})")


def report_alternative_tests(res, crsp):
    print("\nTable 6 - Robustness of the short-run result (full sample)")
    report_robustness(res, crsp, "full sample")

    print("\nTable 7 - Alternative significance tests for the mean CAR [0, +5]")
    groups = [("Full sample", res)] + [(t, res[res["ticker"] == t]) for t in TICKERS]
    for name, sub in groups:
        n_s, t_s, p_s = standardised_cross_sectional(sub, crsp)
        n_g, n_pos, z, p_g = generalised_sign(sub, crsp)
        if n_s < 2:
            print(f"  {name:<14} (insufficient)")
        else:
            print(f"  {name:<14} SCAR t={t_s:+6.2f}{_stars(p_s):<3} p={p_s:.3f}   "
                  f"sign {n_pos}/{n_g}  z={z:+5.2f}{_stars(p_g):<3} p={p_g:.3f}")

    print("\nTable 8 - Mean abnormal returns over alternative event windows")
    for title, windows, func in [("CAR", CAR_WINDOWS, car_over),
                                 ("BHAR", BHAR_WINDOWS, bhar_over)]:
        print(f"  {title} windows:")
        for name, (a, b) in windows.items():
            vals = res.apply(lambda r: func(r, crsp, a, b), axis=1)
            n, mean, t, p = ttest(vals)
            print(f"    {name:<20} N={n:>2}  mean={mean*100:+7.2f}%  "
                  f"t={t:+5.2f}{_stars(p):<3} p={p:.4f}")


def report_excluding_gme(res, crsp):
    print("\nTable 9 - Results excluding GameStop")
    res_x = res[res["ticker"] != DROP_TICKER]
    _print_ttest_row("CAR full", res["car_0_5"])
    _print_ttest_row("CAR ex-GME", res_x["car_0_5"])
    _print_ttest_row("BHAR full", res["bhar_6_60"])
    _print_ttest_row("BHAR ex-GME", res_x["bhar_6_60"])
    report_robustness(res_x, crsp, "excluding GME")


def report_worked_example(res, crsp):
    """Worked single-event AR/CAR table reported in Appendix A."""
    ex = res[res["ticker"] == EXAMPLE_TICKER].iloc[0]
    stock, dates, pos = _locate(crsp, EXAMPLE_TICKER, ex["event_date"])
    seg = stock.iloc[pos:pos + CAR_END + 1][["ret", "mkt_ret"]].copy()
    seg["expected"] = ex["alpha"] + ex["beta"] * seg["mkt_ret"]
    seg["abnormal"] = seg["ret"] - seg["expected"]
    seg.insert(0, "day", range(0, CAR_END + 1))
    print(f"\nAppendix A - worked example: {EXAMPLE_TICKER}, day 0 = "
          f"{ex['event_date'].date()}, alpha={ex['alpha']:.6f}, beta={ex['beta']:.4f}")
    print(seg.round(4).to_string())
    print(f"  Sum of abnormal returns = CAR[0,+5] = {ex['car_0_5']*100:+.2f}%")

# Figures
def plot_event_time(res, crsp):
    """Figure 1: cumulative average abnormal return from day -5 to +60."""
    if not HAS_MATPLOTLIB:
        return
    window = range(-5, BHAR_END + 1)
    ar_rows = []
    for _, e in res.iterrows():
        stock, dates, pos = _locate(crsp, e["ticker"], e["event_date"])
        ar = {}
        for d in window:
            p = pos + d
            ar[d] = (stock.iloc[p]["ret"] - (e["alpha"] + e["beta"] * stock.iloc[p]["mkt_ret"])
                     if 0 <= p < len(dates) else np.nan)
        ar_rows.append(ar)
    ar_df = pd.DataFrame(ar_rows)
    cum = ar_df.mean().cumsum()
    se = (ar_df.std() / np.sqrt(ar_df.notna().sum())).cumsum()
    days = list(window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, cum * 100, color="black", linewidth=2, label="Cumulative AAR")
    ax.fill_between(days, (cum - 1.96 * se) * 100, (cum + 1.96 * se) * 100,
                    alpha=0.15, color="black", label="95% confidence band")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Event date")
    ax.axvline(CAR_END, color="gray", linestyle=":", linewidth=1, label="End of CAR window")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Event time (trading days)")
    ax.set_ylabel("Cumulative average abnormal return (%)")
    ax.set_title("Cumulative Average Abnormal Returns Around Elevated-Attention Events")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_EVENTTIME, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_decomposition(res):
    """Figure 2: per-ticker mean CAR and BHAR, GameStop highlighted."""
    if not HAS_MATPLOTLIB:
        return
    order = [t for t in TICKERS if t != "PLTR" and (res["ticker"] == t).any()]
    grey, grey_e, red, red_e, ink = "#9a9a9a", "#6f6f6f", "#b23a48", "#7d2832", "#1a1a1a"

    def panel(ax, col, title, ylab):
        means, ses = [], []
        for t in order:
            s = res.loc[res["ticker"] == t, col].dropna()
            means.append(s.mean() * 100)
            ses.append(s.std() / np.sqrt(len(s)) * 100 if len(s) >= 2 else np.nan)
        full = res[col].dropna().mean() * 100
        ex = res.loc[res["ticker"] != DROP_TICKER, col].dropna().mean() * 100
        xs = np.arange(len(order))
        colors = [red if t == DROP_TICKER else grey for t in order]
        edges = [red_e if t == DROP_TICKER else grey_e for t in order]
        ax.bar(xs, means, color=colors, edgecolor=edges, linewidth=1.1, width=0.66, zorder=3)
        for xi, m, se, ec in zip(xs, means, ses, edges):
            if not np.isnan(se):
                ax.errorbar(xi, m, yerr=se, fmt="none", ecolor=ec,
                            elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        ax.axhline(0, color=ink, linewidth=0.9, zorder=2)
        ax.axhline(full, color="#555", linewidth=1.3, zorder=2,
                   label=f"Full-sample mean ({full:+.2f}%)")
        ax.axhline(ex, color=ink, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2,
                   label=f"Excl. {DROP_TICKER} mean ({ex:+.2f}%)")
        ax.set_xticks(xs)
        ax.set_xticklabels(order, fontsize=10.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.tick_params(axis="y", labelsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888")
        ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.legend(fontsize=8.6, frameon=False, loc="upper left")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    panel(a1, "car_0_5", "Mean CAR [0, +5] by ticker", "Cumulative abnormal return (%)")
    panel(a2, "bhar_6_60", "Mean BHAR [+6, +60] by ticker", "Buy-and-hold abnormal return (%)")
    fig.tight_layout()
    fig.savefig(FIG_DECOMP, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)




def main():
    crsp = load_crsp()
    mentions = load_mentions()

    events = identify_events(mentions)
    events.to_csv(EVENTS_CSV, index=False, date_format="%Y-%m-%d")

    res = compute_results(events, crsp)
    res.to_csv(RESULTS_CSV, index=False)

    report_descriptives(res, mentions)          # Tables 1-3
    report_mean_abnormal_returns(res)           # Tables 4-5
    report_alternative_tests(res, crsp)         # Tables 6-8
    report_excluding_gme(res, crsp)             # Table 9
    report_worked_example(res, crsp)            # Appendix A

    plot_event_time(res, crsp)                  # Figure 1
    plot_decomposition(res)                     # Figure 2

    print(f"\nSample: {res['car_0_5'].notna().sum()} CAR events, "
          f"{res['bhar_6_60'].notna().sum()} BHAR events")
    print(f"Saved: {RESULTS_CSV}, {EVENTS_CSV}, {MENTIONS_CSV}, "
          f"{FIG_EVENTTIME}, {FIG_DECOMP}")


if __name__ == "__main__":
    main()
