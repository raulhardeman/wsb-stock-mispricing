"""
WSB event study — Bachelor thesis: Social Media Attention and Stock Mispricing.

Market-model event study (MacKinlay, 1997) on elevated-attention events from
r/WallStreetBets. Counts daily ticker mentions, flags days above a rolling
90th-percentile threshold (Hasso et al., 2022), estimates the market model on a
pre-event window, computes CARs over [0, +5] and BHARs over [+6, +60], and then
relates them to firm characteristics in a cross-sectional regression and checks
their robustness to cross-sectional dependence and return skewness.

Inputs:  r_wallstreetbets_posts.csv, crsp_returns.csv, crsp_market.csv
Outputs: thesis_results.csv, thesis_results.xlsx, regression_results.csv,
         robustness_results.csv, figure1_eventtime.png
Usage:   python wsb_pipeline_final.py

Dependencies: numpy, pandas, scipy (matplotlib and openpyxl are optional).
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

REDDIT_CSV     = "r_wallstreetbets_posts.csv"
CRSP_CSV       = "crsp_returns.csv"
MARKET_CSV     = "crsp_market.csv"
OUT_MENTIONS   = "wsb_mentions_daily.csv"
OUT_EVENTS     = "wsb_events.csv"
OUT_CSV        = "thesis_results.csv"
OUT_EXCEL      = "thesis_results.xlsx"
OUT_REGRESSION = "regression_results.csv"
OUT_ROBUSTNESS = "robustness_results.csv"
OUT_FIGURE     = "figure1_eventtime.png"

SAMPLE_START = "2020-01-01"
SAMPLE_END   = "2021-02-16"

# Universe from wsb_ticker_selection.py (>= 500 dollar-sign mentions, equity only).
TICKERS       = ["AMC", "BB", "GME", "NOK", "PLTR", "SPCE", "TSLA"]
TICKER_RENAME = {"IPOA": "SPCE"}   # SPCE traded as IPOA in CRSP before the Oct-2019 merger

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

# Cross-sectional regression and robustness.
REGRESSORS   = ["log_attention", "volatility", "log_mktcap"]
N_BOOTSTRAP  = 10000  # bootstrap resamples
BOOT_SEED    = 42     # bootstrap RNG seed (reproducibility)


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
    mentions = mentions.sort_values(["date", "ticker"]).reset_index(drop=True)
    mentions.to_csv(OUT_MENTIONS, index=False)
    return mentions


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
    events.to_csv(OUT_EVENTS, index=False)
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

        log_mktcap = np.nan
        if {"prc", "shrout"}.issubset(stock.columns):
            last = stock.iloc[pos]
            mc = abs(last["prc"]) * last["shrout"] * 1000
            log_mktcap = float(np.log(mc)) if mc > 0 else np.nan

        rows.append({
            "event_id": event["event_id"], "ticker": ticker,
            "event_date": future[0], "alpha": round(alpha, 6), "beta": round(beta, 4),
            "n_est_days": len(est), "car_0_5": round(car, 6),
            "bhar_6_60": round(bhar, 6) if not np.isnan(bhar) else np.nan,
            "volatility": round(volatility, 6) if not np.isnan(volatility) else np.nan,
            "log_mktcap": round(log_mktcap, 4) if not np.isnan(log_mktcap) else np.nan,
        })
    return pd.DataFrame(rows)


def _ttest(series, label):
    s = series.dropna()
    if len(s) < 2:
        print(f"  {label:<13} N={len(s):>2}  (insufficient)")
        return
    t, p = stats.ttest_1samp(s, 0)
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"  {label:<13} N={len(s):>2}  mean={s.mean()*100:+.2f}%  t={t:+.3f}{stars:<3} p={p:.4f}")


def descriptive_statistics(res):
    """Print Table 1 (descriptives) and Tables 2-3 (CAR/BHAR t-tests)."""
    table1 = []
    for col, label, pct in [("car_0_5", "CAR [0,+5] (%)", True),
                            ("bhar_6_60", "BHAR [+6,+60] (%)", True),
                            ("volatility", "Return Volatility", False),
                            ("log_mktcap", "Log Market Cap", False)]:
        s = res[col].dropna() * (100 if pct else 1)
        table1.append({"Variable": label, "N": len(s), "Mean": round(s.mean(), 4),
                       "Median": round(s.median(), 4), "Std Dev": round(s.std(), 4),
                       "P10": round(s.quantile(.10), 4), "P90": round(s.quantile(.90), 4)})
    table1 = pd.DataFrame(table1)

    print("\nTable 1 — Descriptive statistics")
    print(table1.to_string(index=False))

    print("\nTable 2 — Mean CAR [0,+5]")
    _ttest(res["car_0_5"], "Full sample")
    for t in TICKERS:
        _ttest(res.loc[res["ticker"] == t, "car_0_5"], t)

    print("\nTable 3 — Mean BHAR [+6,+60]")
    _ttest(res["bhar_6_60"], "Full sample")
    for t in TICKERS:
        _ttest(res.loc[res["ticker"] == t, "bhar_6_60"], t)
    return table1


def _ols_hc1(y, X):
    """OLS with HC1 (heteroskedasticity-robust) standard errors.

    X must already contain an intercept column. Returns
    (params, se, t_stats, p_values, r_squared, n_obs, n_params).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta

    # HC1 robust covariance: (X'X)^-1 (sum e_i^2 x_i x_i') (X'X)^-1 * n/(n-k).
    weighted = X * resid[:, None]
    cov = xtx_inv @ (weighted.T @ weighted) @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=n - k)

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return beta, se, t_stats, p_values, r2, n, k


def cross_sectional_regressions(res, events):
    """Table 4: regress CAR[0,+5] and BHAR[+6,+60] on event-day attention, pre-event
    return volatility, and firm size, with HC1 robust standard errors.

    Tests the Baker and Wurgler (2006) prediction that sentiment moves prices most
    in small, volatile, hard-to-value stocks. Returns a tidy results table.
    """
    df = res.merge(events[["event_id", "mention_count"]], on="event_id", how="left")
    df["log_attention"] = np.log(df["mention_count"].where(df["mention_count"] > 0))

    print("\nTable 4 — Cross-sectional regressions (HC1 robust SE)")
    out = []
    for dep, name in [("car_0_5", "CAR [0,+5]"), ("bhar_6_60", "BHAR [+6,+60]")]:
        d = df.dropna(subset=[dep] + REGRESSORS)
        if len(d) <= len(REGRESSORS) + 1:
            print(f"  {name}: insufficient observations (N={len(d)})")
            continue
        X = np.column_stack([np.ones(len(d)), d[REGRESSORS].values])
        beta, se, t_stats, p_values, r2, n, _ = _ols_hc1(d[dep].values, X)

        print(f"  {name} (N={n})")
        for label, b, t, p in zip(["const"] + REGRESSORS, beta, t_stats, p_values):
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            print(f"    {label:<14} coef={b:+.4f}  t={t:+.3f}{stars:<3} p={p:.4f}")
            out.append({"model": name, "variable": label, "coef": round(float(b), 6),
                        "t_stat": round(float(t), 4), "p_value": round(float(p), 4)})
        print(f"    {'R-squared':<14} {r2:.4f}")
        out.append({"model": name, "variable": "R-squared", "coef": round(float(r2), 4),
                    "t_stat": np.nan, "p_value": np.nan})
        out.append({"model": name, "variable": "N", "coef": n,
                    "t_stat": np.nan, "p_value": np.nan})
    return pd.DataFrame(out)


def _calendar_time_portfolio(res, crsp, day_start=0, day_end=CAR_END):
    """Equal-weighted calendar-time portfolio over the [day_start, day_end] window.

    Averaging same-date abnormal returns into a single portfolio observation removes
    the cross-sectional correlation that inflates the simple cross-sectional t-test.
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
    plus a percentile 95% confidence interval. Returns (mean, p, ci_low, ci_high, n).
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


def robustness_tests(res, crsp):
    """Robustness for the headline means: a calendar-time portfolio over the CAR
    window (addresses cross-sectional dependence) and a skew-robust bootstrap for
    both CAR and BHAR. Returns a summary table.
    """
    print("\nRobustness — dependence- and skew-robust inference")
    out = []

    ct_mean, ct_t, ct_p, ct_n = _calendar_time_portfolio(res, crsp)
    print(f"  Calendar-time portfolio [0,+{CAR_END}]: "
          f"daily mean AR={ct_mean*100:+.3f}%  t={ct_t:+.3f}  p={ct_p:.4f}  (obs={ct_n})")
    out.append({"test": "calendar_time_portfolio", "series": f"AR [0,+{CAR_END}]",
                "mean_pct": round(ct_mean * 100, 4), "t_stat": round(ct_t, 4),
                "p_value": round(ct_p, 4), "ci_low_pct": np.nan, "ci_high_pct": np.nan,
                "n": ct_n})

    for col, name in [("car_0_5", "CAR [0,+5]"), ("bhar_6_60", "BHAR [+6,+60]")]:
        m, p, lo, hi, n = _bootstrap_mean(res[col])
        print(f"  Bootstrap {name}: mean={m*100:+.2f}%  p={p:.4f}  "
              f"95% CI=[{lo*100:+.2f}%, {hi*100:+.2f}%]  (n={n}, {N_BOOTSTRAP} resamples)")
        out.append({"test": "bootstrap", "series": name, "mean_pct": round(m * 100, 4),
                    "t_stat": np.nan, "p_value": round(p, 4),
                    "ci_low_pct": round(lo * 100, 4), "ci_high_pct": round(hi * 100, 4),
                    "n": n})
    return pd.DataFrame(out)


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
    plt.savefig(OUT_FIGURE, dpi=150, bbox_inches="tight")
    plt.close()


def export_results(res, table1, reg_table, robust_table):
    """Write all results to CSV, and to a multi-sheet workbook when openpyxl is present."""
    res.to_csv(OUT_CSV, index=False)
    reg_table.to_csv(OUT_REGRESSION, index=False)
    robust_table.to_csv(OUT_ROBUSTNESS, index=False)
    try:
        with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
            res.to_excel(writer, sheet_name="Events", index=False)
            table1.to_excel(writer, sheet_name="Table1", index=False)
            reg_table.to_excel(writer, sheet_name="Table4_Regression", index=False)
            robust_table.to_excel(writer, sheet_name="Robustness", index=False)
    except ModuleNotFoundError:
        pass  # openpyxl not installed; CSV files are still written


def main():
    mentions = build_mentions(load_reddit())
    events = identify_events(mentions)
    crsp = load_crsp()
    res = compute_abnormal_returns(events, crsp)

    table1 = descriptive_statistics(res)
    reg_table = cross_sectional_regressions(res, events)
    robust_table = robustness_tests(res, crsp)

    plot_event_time(crsp, events)
    export_results(res, table1, reg_table, robust_table)

    print(f"\nSample: {res['car_0_5'].notna().sum()} CAR events, "
          f"{res['bhar_6_60'].notna().sum()} BHAR events")
    print(f"Saved: {OUT_CSV}, {OUT_EXCEL}, {OUT_REGRESSION}, {OUT_ROBUSTNESS}, {OUT_FIGURE}")


if __name__ == "__main__":
    main()
