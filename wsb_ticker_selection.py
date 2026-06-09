"""
WSB ticker selection — Bachelor thesis: Social Media Attention and Stock Mispricing.

Builds the ticker universe organically from the data: scans r/WallStreetBets post
titles for dollar-sign mentions (e.g. $GME), keeps tickers above a mention threshold,
and flags tokens that are not U.S. common equity in CRSP for exclusion.

Output: candidate_tickers.csv
Usage:  python wsb_ticker_selection.py
"""

import re
import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict

REDDIT_CSV          = "r_wallstreetbets_posts.csv"
CANDIDATES_CSV      = "candidate_tickers.csv"
SAMPLE_START        = "2020-01-01"
SAMPLE_END          = "2021-02-16"
MIN_DOLLAR_MENTIONS = 500

# Tokens that pass the mention threshold but are not U.S. common equity in CRSP.
NOT_EQUITY = {
    "NAKD": "Delisted from NASDAQ during sample period",
    "SPY":  "ETF (SPDR S&P 500), not common equity",
    "SLV":  "ETF (iShares Silver Trust), not common equity",
    "DOGE": "Cryptocurrency, not exchange-listed",
    "SNDL": "Canadian issuer (Sundial Growers), not in CRSP",
}

# $TICKER only: the dollar prefix signals an intentional ticker reference,
# avoiding false positives from ordinary uppercase words.
DOLLAR_RE = re.compile(r"\$([A-Z]{1,5})\b")


def load_reddit():
    if not Path(REDDIT_CSV).exists():
        sys.exit(f"Error: {REDDIT_CSV} not found.")
    df = pd.read_csv(REDDIT_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df = df[(df["date"] >= SAMPLE_START) & (df["date"] <= SAMPLE_END)]
    return df.dropna(subset=["date", "title"])


def count_dollar_mentions(df):
    """Count dollar-sign mention occurrences per ticker over the whole sample.

    Every occurrence counts: a title that contains $GME twice adds 2 to GME.
    This is a coarse popularity screen. The event study (wsb_event_study.py)
    instead counts distinct posts per ticker-day, so its totals are slightly
    lower (e.g. GME: 15,344 posts vs 15,679 occurrences). Both screens give
    the same ticker universe.
    """
    counts = defaultdict(int)
    for title in df["title"]:
        for m in DOLLAR_RE.finditer(str(title).upper()):
            if len(m.group(1)) >= 2:
                counts[m.group(1)] += 1
    return (pd.DataFrame(counts.items(), columns=["ticker", "dollar_mentions"])
              .sort_values("dollar_mentions", ascending=False)
              .reset_index(drop=True))


def build_candidates(mentions):
    cand = mentions[mentions["dollar_mentions"] >= MIN_DOLLAR_MENTIONS].copy()
    cand["include"] = (~cand["ticker"].isin(NOT_EQUITY)).astype(int)
    cand["exclusion_reason"] = cand["ticker"].map(NOT_EQUITY).fillna("")
    return cand


def main():
    df = load_reddit()
    print(f"{len(df):,} posts loaded ({SAMPLE_START} to {SAMPLE_END})")

    candidates = build_candidates(count_dollar_mentions(df))
    candidates.to_csv(CANDIDATES_CSV, index=False)

    included = candidates[candidates["include"] == 1]["ticker"].tolist()
    print(f"{len(candidates)} candidates, {len(included)} included after review")
    print(f"Final universe: {sorted(included)}")
    print(f"Saved: {CANDIDATES_CSV}")


if __name__ == "__main__":
    main()

