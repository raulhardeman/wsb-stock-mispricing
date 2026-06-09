"""
WSB event study — Bachelor thesis: Social Media Attention and Stock Mispricing.

Market-model event study (MacKinlay, 1997) on elevated-attention events from
r/WallStreetBets. The script:
  1. counts daily dollar-sign ticker mentions and flags days above a rolling
     90th-percentile threshold (Hasso et al., 2022) as attention events;
  2. estimates the market model on a pre-event window and computes CARs over
     [0, +5] and BHARs over [+6, +60];
  3. reports descriptive statistics and one-sample t-tests (Tables 1-3);
  4. checks the short-run result against cross-sectional dependence
     (calendar-time portfolio, Fama 1998) and return skewness (bootstrap,
     Lyon, Barber & Tsai 1999);
  5. repeats the headline analysis on the sample excluding GameStop (Table 5);
  6. draws Figure 1 (cumulative average abnormal return) and Figure 2
     (per-ticker CAR/BHAR with and without GameStop).

Inputs:  r_wallstreetbets_posts.csv (raw Reddit posts, Kaggle; see README) or, if the
         raw file is absent, the derived snapshot wsb_mentions_daily.csv;
         crsp_returns.csv and crsp_market.csv (CRSP via WRDS, not redistributed).
Outputs: wsb_mentions_daily.csv and wsb_events.csv (derived snapshots, written when
         built from the raw file), thesis_results.csv, figure1_eventtime.png,
         figure2_decomposition.png
Usage:   python wsb_event_study.py

Dependencies: numpy, pandas, scipy (matplotlib optional, for the figures).

Reproducibility notes:
  * Mention counts here are distinct posts per ticker-day. The universe screen in
    wsb_ticker_selection.py counts mention occurrences, so its totals (e.g. GME
    15,679) sit slightly above the post counts reported in the thesis (GME 15,344).
    Both totals are correct; they answer different questions.
  * identify_events() places the daily counts on a Monday-to-Friday grid
    (pd.bdate_range). Weekend mentions appear in wsb_mentions_daily.csv but do
    not enter the threshold or the event definition. A flagged day that falls
    on a market holiday takes the next CRSP trading day as event day 0.
  * The minimum spacing between same-ticker events (MIN_GAP_DAYS) is measured in
    rows of that weekday grid, which includes market holidays.
  * The bootstrap uses a fixed seed (BOOT_SEED), so all reported p-values and
    confidence intervals reproduce exactly.
"""

import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Input and output files.
REDDIT_CSV    = "r_wallstreetbets_posts.csv"
CRSP_CSV      = "crsp_returns.csv"
MARKET_CSV    = "crsp_market.csv"
MENTIONS_CSV  = "wsb_mentions_daily.csv"   # derived snapshot, fallback input
EVENTS_CSV    = "wsb_events.csv"           # derived snapshot
OUT_CSV       = "thesis_results.csv"
OUT_FIG_EVT   = "figure1_eventtime.png"
OUT_FIG_DECOMP = "figure2_decomposition.png"

SAMPLE_START = "2020-01-01"
SAMPLE_END   = "2021-02-16"

# Universe from wsb_ticker_selection.py (>= 500 dollar-sign mentions, equity only).
TICKERS       = ["AMC", "BB", "GME", "NOK", "PLTR", "SPCE", "TSLA"]
TICKER_RENAME = {"IPOA": "SPCE"}   # SPCE traded as IPOA in CRSP before the 2019 merger
DROP_TICKER   = "GME"              # ticker excluded in the robustness check (Section 5.4)

# Event identification (Hasso et al., 2022).
PERCENTILE   = 0.90   # rolling percentile threshold
LOOKBACK     = 250    # lookback window (trading days)
MIN_OBS      = 30     # min observations before computing the threshold
MIN_GAP_DAYS = 65     # min spacing between events of the same ticker (trading days)

# Market model (MacKinlay, 1997).
EST_WINDOW   = 250    # estimation window length
EST_GAP      = 10     # gap between estimation window and event date
MIN_EST_OBS  = 100    # min observations to estimate the model
CAR_END      = 5      # CAR window [0, +5]
BHAR_START   = 6      # BHAR window [+6, +60]
BHAR_END     = 60

# Robustness (calendar-time portfolio and bootstrap).
N_BOOTSTRAP  = 10000  # bootstrap resamples
BOOT_SEED    = 42     # bootstrap RNG seed (reproducibility)


# --------------------------------------------------------------------------- #
# Data loading and event identification
# --------------------------------------------------------------------------- #
def load_reddit():
    if not Path(REDDIT_CSV).exists():
        sys.exit(f"Error: {REDDIT_CSV} not found.")
    df = pd.read_csv(REDDIT_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    return df.dropna(subset=["date", "title"])


def build_mentions(df):
    """Count daily mentions per ticker as the number of distinct posts that mention it.

    Only dollar-sign mentions ($GME) are matched, and a post counts once per ticker
    regardless of how many times that ticker appears in its title.
    """
    ticker_re = re.compile(r"\$(" + "|".join(map(re.escape, TICKERS)) + r")\b")

    counts = defaultdict(int)
    for _, row in df.iterrows():
        day = row["date"].date()
        tickers_in_post = {m.group(1) for m in ticker_re.finditer(str(row["title"]).upper())}
        for ticker in tickers_in_post:
            counts[(day, ticker)] += 1

    mentions = pd.DataFrame(
        [{"date": d, "ticker": t, "mention_count": c} for (d, t), c in counts.items()]
    )
    mentions["date"] = pd.to_datetime(mentions["date"])
    return mentions.sort_values(["date", "ticker"]).reset_index(drop=True)


def load_or_build_mentions():
    """Return the daily mention counts, preferring the raw Reddit file.

    If r_wallstreetbets_posts.csv is present, mentions are built from it and the
    snapshot wsb_mentions_daily.csv is (re)written. If the raw file is absent,
    the snapshot is loaded instead, so the event study reproduces without the
    raw Kaggle file (about 220 MB). Both routes give identical results.
    """
    if Path(REDDIT_CSV).exists():
        mentions = build_mentions(load_reddit())
        mentions.to_csv(MENTIONS_CSV, index=False, date_format="%Y-%m-%d")
        print(f"Built mentions from {REDDIT_CSV}; snapshot saved to {MENTIONS_CSV}")
        return mentions
    if Path(MENTIONS_CSV).exists():
        mentions = pd.read_csv(MENTIONS_CSV, parse_dates=["date"])
        print(f"{REDDIT_CSV} not found; loaded snapshot {MENTIONS_CSV} "
              f"({len(mentions)} ticker-day rows)")
        return mentions.sort_values(["date", "ticker"]).reset_index(drop=True)
    sys.exit(f"Error: neither {REDDIT_CSV} nor {MENTIONS_CSV} found. "
             f"See README.md for how to obtain the data.")


def identify_events(mentions):
    """Flag stock-days above the rolling 90th percentile, then enforce the min-gap rule.

    shift(1) ensures the threshold uses only prior days, avoiding look-ahead bias.
    """
    all_dates = pd.bdate_range(mentions["date"].min(), mentions["date"].max())
    idx = pd.MultiIndex.from_product([all_dates, TICKERS], names=["date", "ticker"])
    panel = (mentions.set_index(["date", "ticker"])["mention_count"]
                     .reindex(idx, fill_value=0).reset_index())
    panel["date"] = pd.to_datetime(panel["date"])

    out = []
    for ticker in TICKERS:
        sub = panel[panel["ticker"] == ticker].sort_values("date").copy()
        sub["p90"] = (sub["mention_count"].shift(1)
                                          .rolling(LOOKBACK, min_periods=MIN_OBS)
                                          .quantile(PERCENTILE))
        sub["elevated"] = (sub["mention_count"] > sub["p90"]).astype(int)

        # Keep only the first day of each cluster, spaced >= MIN_GAP_DAYS *trading*
        # days apart. Spacing is measured in row positions, not calendar days.
        sub = sub.reset_index(drop=True)
        last_pos, flags = None, []
        for pos, row in sub.iterrows():
            if row["elevated"] and (last_pos is None or (pos - last_pos) >= MIN_GAP_DAYS):
                flags.append(1)
                last_pos = pos
            else:
                flags.append(0)
        sub["elevated"] = flags
        out.append(sub)

    panel = pd.concat(out, ignore_index=True)
    events = panel[panel["elevated"] == 1].copy()
    events["event_id"] = range(1, len(events) + 1)
    return events


def load_crsp():
    """Load CRSP daily returns + value-weighted market return; relabel IPOA to SPCE."""
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


# --------------------------------------------------------------------------- #
# Event-study core
# --------------------------------------------------------------------------- #
def compute_abnormal_returns(events, crsp):
    """Estimate the market model per event and compute CAR[0,+5] and BHAR[+6,+60].

    Market model: AR_it = R_it - (alpha_i + beta_i * R_Mt), estimated by OLS over
    [t-260, t-11]. The 10-day gap keeps pre-event drift out of the parameters.
    """
    rows = []
    for _, event in events.iterrows():
        ticker = event["ticker"]
        stock = crsp[crsp["ticker"] == ticker].set_index("date").sort_index()
        dates = list(stock.index)

        future = [d for d in dates if d >= pd.to_datetime(event["date"])]
        if not future:
            continue
        pos = dates.index(future[0])

        est_end, est_start = pos - EST_GAP, pos - EST_GAP - EST_WINDOW
        if est_start < 0 or (est_end - est_start) < MIN_EST_OBS:
            continue
        est = stock.iloc[est_start:est_end]
        if len(est) < MIN_EST_OBS:
            continue

        # OLS with intercept: the column of ones estimates alpha.
        x = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
        coef, *_ = np.linalg.lstsq(x, est["ret"].values, rcond=None)
        alpha, beta = coef

        if pos + CAR_END + 1 > len(dates):
            continue
        ev = stock.iloc[pos:pos + CAR_END + 1]
        car = float((ev["ret"] - (alpha + beta * ev["mkt_ret"])).sum())

        # BHAR compounds (buy-and-hold) rather than summing daily abnormal returns.
        if pos + BHAR_END + 1 <= len(dates):
            post = stock.iloc[pos + BHAR_START:pos + BHAR_END + 1]
            bhar = float((1 + post["ret"]).prod() - (1 + post["mkt_ret"]).prod())
        else:
            bhar = np.nan

        vol_w = stock.iloc[max(0, pos - 60):pos]
        volatility = float(vol_w["ret"].std()) if len(vol_w) >= 20 else np.nan

        rows.append({
            "event_id": event["event_id"], "ticker": ticker,
            "event_date": future[0], "alpha": round(alpha, 6), "beta": round(beta, 4),
            "n_est_days": len(est), "car_0_5": round(car, 6),
            "bhar_6_60": round(bhar, 6) if not np.isnan(bhar) else np.nan,
            "volatility": round(volatility, 6) if not np.isnan(volatility) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def _ttest(series, label):
    """One-sample t-test of H0: mean = 0, printed in a fixed-width row."""
    s = series.dropna()
    if len(s) < 2:
        print(f"  {label:<17} N={len(s):>2}  (insufficient)")
        return
    t, p = stats.ttest_1samp(s, 0)
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<17} N={len(s):>2}  mean={s.mean()*100:+.2f}%  t={t:+.3f}{stars:<3} p={p:.4f}")


def _calendar_time_portfolio(res, crsp, day_start=0, day_end=CAR_END):
    """Equal-weighted calendar-time portfolio over the [day_start, day_end] window.

    Averaging same-date abnormal returns into a single portfolio observation removes
    the cross-sectional correlation that inflates the simple cross-sectional t-test
    (Fama, 1998). Operates on whatever event set is passed, so the same function
    serves both the full sample and the excluding-GME sample.
    Returns (mean_ar, t_stat, p_value, n_trading_days).
    """
    recs = []
    for _, e in res.iterrows():
        stock = crsp[crsp["ticker"] == e["ticker"]].set_index("date").sort_index()
        dates = list(stock.index)
        try:
            pos = dates.index(pd.to_datetime(e["event_date"]))
        except ValueError:
            continue
        for d in range(day_start, day_end + 1):
            p = pos + d
            if 0 <= p < len(dates):
                row = stock.iloc[p]
                recs.append({"date": dates[p],
                             "ar": row["ret"] - (e["alpha"] + e["beta"] * row["mkt_ret"])})
    if not recs:
        return np.nan, np.nan, np.nan, 0
    port = pd.DataFrame(recs).groupby("date")["ar"].mean()
    t, p = stats.ttest_1samp(port, 0)
    return float(port.mean()), float(t), float(p), len(port)


def _bootstrap_mean(series, n_boot=N_BOOTSTRAP, seed=BOOT_SEED):
    """Skew-robust test of H0: mean = 0 via a bootstrap of the centred sample,
    plus a percentile 95% confidence interval (Lyon, Barber & Tsai, 1999).
    Returns (mean, p, ci_low, ci_high, n).
    """
    x = series.dropna().to_numpy()
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan, np.nan, len(x)
    m = x.mean()
    rng = np.random.default_rng(seed)
    null = rng.choice(x - m, size=(n_boot, len(x)), replace=True).mean(axis=1)
    p = float((np.abs(null) >= abs(m)).mean())
    boot = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(m), p, float(lo), float(hi), len(x)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def descriptive_statistics(res):
    """Table 1 (descriptives) and Tables 2-3 (CAR/BHAR t-tests, full sample)."""
    table1 = []
    for col, label, pct in [("car_0_5", "CAR [0,+5] (%)", True),
                            ("bhar_6_60", "BHAR [+6,+60] (%)", True),
                            ("volatility", "Return Volatility", False)]:
        s = res[col].dropna() * (100 if pct else 1)
        table1.append({"Variable": label, "N": len(s), "Mean": round(s.mean(), 4),
                       "Median": round(s.median(), 4), "Std Dev": round(s.std(), 4),
                       "P10": round(s.quantile(.10), 4), "P90": round(s.quantile(.90), 4)})

    print("\nTable 1 — Descriptive statistics")
    print(pd.DataFrame(table1).to_string(index=False))

    print("\nTable 2 — Mean CAR [0,+5]")
    _ttest(res["car_0_5"], "Full sample")
    for t in TICKERS:
        _ttest(res.loc[res["ticker"] == t, "car_0_5"], t)

    print("\nTable 3 — Mean BHAR [+6,+60]")
    _ttest(res["bhar_6_60"], "Full sample")
    for t in TICKERS:
        _ttest(res.loc[res["ticker"] == t, "bhar_6_60"], t)


def robustness_tests(res, crsp, label="full sample"):
    """Calendar-time portfolio (cross-sectional dependence) and bootstrap (return
    skewness) for the CAR and BHAR means of whatever event set is passed.
    """
    print(f"\nRobustness ({label}) — dependence- and skew-robust inference")

    ct_mean, ct_t, ct_p, ct_n = _calendar_time_portfolio(res, crsp)
    print(f"  Calendar-time portfolio [0,+{CAR_END}]: "
          f"daily mean AR={ct_mean*100:+.3f}%  t={ct_t:+.3f}  p={ct_p:.4f}  (obs={ct_n})")

    for col, name in [("car_0_5", "CAR [0,+5]"), ("bhar_6_60", "BHAR [+6,+60]")]:
        m, p, lo, hi, n = _bootstrap_mean(res[col])
        print(f"  Bootstrap {name}: mean={m*100:+.2f}%  p={p:.4f}  "
              f"95% CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]  (n={n}, {N_BOOTSTRAP} resamples)")


def excluding_gme(res, crsp):
    """Table 5: repeat the headline means and the robustness tests on the sample
    that excludes GameStop, to check the short-run result is not a GME artefact.
    """
    res_x = res[res["ticker"] != DROP_TICKER]

    print("\nMean abnormal returns — full sample vs excluding GME")
    _ttest(res["car_0_5"], "CAR [0,+5] full")
    _ttest(res_x["car_0_5"], "CAR [0,+5] ex-GME")
    _ttest(res["bhar_6_60"], "BHAR [+6,+60] full")
    _ttest(res_x["bhar_6_60"], "BHAR [+6,+60] ex-GME")

    robustness_tests(res_x, crsp, label="excluding GME")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_event_time(crsp, events):
    """Figure 1: cumulative average abnormal return in event time, day -5 to +60."""
    if not HAS_MATPLOTLIB:
        return

    window = range(-5, 61)
    ar_matrix = []
    for _, event in events.iterrows():
        stock = crsp[crsp["ticker"] == event["ticker"]].set_index("date").sort_index()
        dates = list(stock.index)
        future = [d for d in dates if d >= pd.to_datetime(event["date"])]
        if not future:
            continue
        pos = dates.index(future[0])

        est_end, est_start = pos - EST_GAP, pos - EST_GAP - EST_WINDOW
        if est_start < 0 or (est_end - est_start) < MIN_EST_OBS:
            continue
        est = stock.iloc[est_start:est_end]
        x = np.column_stack([np.ones(len(est)), est["mkt_ret"].values])
        coef, *_ = np.linalg.lstsq(x, est["ret"].values, rcond=None)
        alpha, beta = coef

        ar = {}
        for d in window:
            p = pos + d
            ar[d] = (stock.iloc[p]["ret"] - (alpha + beta * stock.iloc[p]["mkt_ret"])
                     if 0 <= p < len(dates) else np.nan)
        ar_matrix.append(ar)

    if not ar_matrix:
        return
    ar_df = pd.DataFrame(ar_matrix)
    cum = ar_df.mean().cumsum()
    se = (ar_df.std() / np.sqrt(ar_df.notna().sum())).cumsum()
    days = list(window)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, cum * 100, color="black", linewidth=2, label="Cumulative AAR")
    ax.fill_between(days, (cum - 1.96 * se) * 100, (cum + 1.96 * se) * 100,
                    alpha=0.15, color="black", label="95% confidence band")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Event date")
    ax.axvline(5, color="gray", linestyle=":", linewidth=1, label="End of CAR window")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Event time (trading days)")
    ax.set_ylabel("Cumulative average abnormal return (%)")
    ax.set_title("Cumulative Average Abnormal Returns Around Elevated Attention Events")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG_EVT, dpi=150, bbox_inches="tight")
    plt.close()


def plot_decomposition(res):
    """Figure 2: per-ticker mean CAR and BHAR, GME highlighted, with full-sample and
    excluding-GME mean reference lines and +/- 1 standard-error whiskers.
    """
    if not HAS_MATPLOTLIB:
        return
    order = [t for t in TICKERS if t != "PLTR" and (res["ticker"] == t).any()]
    grey, grey_e, red, red_e, ink = "#9a9a9a", "#6f6f6f", "#b23a48", "#7d2832", "#1a1a1a"

    def by(col):
        d = {}
        for tk in order:
            s = res.loc[res["ticker"] == tk, col].dropna()
            d[tk] = (len(s), s.mean()*100, (s.std()/np.sqrt(len(s))*100) if len(s) >= 2 else np.nan)
        full = res[col].dropna().mean()*100
        ex = res.loc[res["ticker"] != DROP_TICKER, col].dropna().mean()*100
        return d, full, ex

    def panel(ax, col, title, ylab):
        d, full, ex = by(col)
        xs = np.arange(len(order)); means = [d[t][1] for t in order]; ses = [d[t][2] for t in order]
        edges = [red_e if t == DROP_TICKER else grey_e for t in order]
        colors = [red if t == DROP_TICKER else grey for t in order]
        ax.bar(xs, means, color=colors, edgecolor=edges, linewidth=1.1, width=0.66, zorder=3)
        for xi, m, se, t in zip(xs, means, ses, order):
            if not np.isnan(se):
                ax.errorbar(xi, m, yerr=se, fmt="none", ecolor=edges[order.index(t)],
                            elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
        ax.axhline(0, color=ink, linewidth=0.9, zorder=2)
        ax.axhline(full, color="#555", linewidth=1.3, zorder=2, label=f"Full-sample mean ({full:+.2f}%)")
        ax.axhline(ex, color=ink, linewidth=1.4, linestyle=(0, (5, 3)), zorder=2,
                   label=f"Excl. {DROP_TICKER} mean ({ex:+.2f}%)")
        ax.set_xticks(xs); ax.set_xticklabels(order, fontsize=10.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(ylab, fontsize=10.5); ax.tick_params(axis="y", labelsize=9.5)
        ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color("#888")
        ax.grid(axis="y", color="#e3e3e3", linewidth=0.8, zorder=0)
        ax.legend(fontsize=8.6, frameon=False, loc="upper left")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    panel(a1, "car_0_5", "Mean CAR [0, +5] by ticker", "Cumulative abnormal return (%)")
    panel(a2, "bhar_6_60", "Mean BHAR [+6, +60] by ticker", "Buy-and-hold abnormal return (%)")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DECOMP, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    crsp = load_crsp()
    events = identify_events(load_or_build_mentions())
    events.to_csv(EVENTS_CSV, index=False, date_format="%Y-%m-%d")
    res = compute_abnormal_returns(events, crsp)

    descriptive_statistics(res)                  # Tables 1, 2, 3
    robustness_tests(res, crsp, "full sample")   # Section 5.3
    excluding_gme(res, crsp)                     # Section 5.4 (Table 5)

    plot_event_time(crsp, events)                # Figure 1
    plot_decomposition(res)                      # Figure 2

    res.to_csv(OUT_CSV, index=False)
    print(f"\nSample: {res['car_0_5'].notna().sum()} CAR events, "
          f"{res['bhar_6_60'].notna().sum()} BHAR events")
    print(f"Saved: {EVENTS_CSV}, {OUT_CSV}, {OUT_FIG_EVT}, {OUT_FIG_DECOMP}")


if __name__ == "__main__":
    main()

