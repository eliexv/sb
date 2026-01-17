"""
build_dataset.py

Reads Football-Data.co.uk season CSVs from data/raw/*.csv and outputs:
data/clean/epl_matches_clean.csv

Key outputs:
- season, date, home_team, away_team, home_goals, away_goals, total_goals, over25
- odds_over25_bet / odds_under25_bet          (bettable "open" odds, pref B365)
- odds_over25_ref_close / odds_under25_ref_close  (reference market "closing", pref AvgC)
- odds_over25_bet_close / odds_under25_bet_close  (optional B365 close)

Also includes backwards-compat aliases:
- odds_over25 / odds_under25                == *_bet
- odds_over25_close / odds_under25_close    == *_ref_close
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import List, Optional

import numpy as np
import pandas as pd

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT_FILE = "data/clean/epl_matches_clean.csv"
DEFAULT_LEAGUE = "E0"  # EPL

def first_non_null_series(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Return the first candidate column found that has ANY non-null numeric values.
    If none found, return all-NaN series.
    """
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any():
                return s
    return pd.Series([np.nan] * len(df), index=df.index)

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def infer_season_from_date(d: pd.Timestamp) -> Optional[str]:
    """
    EPL seasons start in AUGUST.
    Using month>=8 fixes COVID seasons where matches were played in July.
    """
    if pd.isna(d):
        return None
    y = int(d.year)
    m = int(d.month)
    start = y if m >= 8 else (y - 1)
    return f"{start}-{start+1}"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def build_dataset(raw_dir: str, league: Optional[str] = DEFAULT_LEAGUE) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="latin-1")

        df["__source_file"] = os.path.basename(fp)

        if league is not None and "Div" in df.columns:
            df = df[df["Div"].astype(str).str.upper().eq(league.upper())].copy()

        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    required_base = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required_base if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in raw data: {missing}")

    out = pd.DataFrame()
    out["date"] = parse_date_series(raw["Date"])
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(raw["FTAG"], errors="coerce")
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["over25"] = (out["total_goals"] >= 3).astype(int)
    out["season"] = out["date"].apply(infer_season_from_date)

    # -----------------------------
    # Odds mapping
    # -----------------------------
    # Bet odds (open): prefer B365, but older seasons sometimes use BbAv/BbMx.
    BET_OVER_CANDS = [
        "B365>2.5", "P>2.5", "Max>2.5", "Avg>2.5", "BFE>2.5",
        "BbMx>2.5", "BbAv>2.5"
    ]
    BET_UNDER_CANDS = [
        "B365<2.5", "P<2.5", "Max<2.5", "Avg<2.5", "BFE<2.5",
        "BbMx<2.5", "BbAv<2.5"
    ]

    # Reference close (market): prefer AvgC, then other close columns if present.
    REF_CLOSE_OVER_CANDS = [
        "AvgC>2.5", "MaxC>2.5", "PC>2.5", "B365C>2.5", "BFEC>2.5",
        "BbMxC>2.5", "BbAvC>2.5"
    ]
    REF_CLOSE_UNDER_CANDS = [
        "AvgC<2.5", "MaxC<2.5", "PC<2.5", "B365C<2.5", "BFEC<2.5",
        "BbMxC<2.5", "BbAvC<2.5"
    ]

    # Optional: bet close (B365 close)
    BET_CLOSE_OVER = ["B365C>2.5"]
    BET_CLOSE_UNDER = ["B365C<2.5"]

    out["odds_over25_bet"] = first_non_null_series(raw, BET_OVER_CANDS)
    out["odds_under25_bet"] = first_non_null_series(raw, BET_UNDER_CANDS)

    out["odds_over25_ref_close"] = first_non_null_series(raw, REF_CLOSE_OVER_CANDS)
    out["odds_under25_ref_close"] = first_non_null_series(raw, REF_CLOSE_UNDER_CANDS)

    out["odds_over25_bet_close"] = first_non_null_series(raw, BET_CLOSE_OVER)
    out["odds_under25_bet_close"] = first_non_null_series(raw, BET_CLOSE_UNDER)

    # Backwards-compat aliases
    out["odds_over25"] = out["odds_over25_bet"]
    out["odds_under25"] = out["odds_under25_bet"]
    out["odds_over25_close"] = out["odds_over25_ref_close"]
    out["odds_under25_close"] = out["odds_under25_ref_close"]

    out["source_file"] = raw["__source_file"].astype(str)

    # Coerce numeric columns
    num_cols = [
        "home_goals","away_goals","total_goals",
        "odds_over25_bet","odds_under25_bet",
        "odds_over25_ref_close","odds_under25_ref_close",
        "odds_over25_bet_close","odds_under25_bet_close",
        "odds_over25","odds_under25","odds_over25_close","odds_under25_close"
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop essentials (do NOT require ref close odds)
    out = out.dropna(subset=[
        "date","season","home_team","away_team",
        "home_goals","away_goals",
        "odds_over25_bet","odds_under25_bet"
    ])

    # De-dup
    out = out.sort_values(["date","home_team","away_team"]).drop_duplicates(
        subset=["date","home_team","away_team"], keep="last"
    ).reset_index(drop=True)

    return out

def print_summary(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Min date: {df['date'].min()} Max date: {df['date'].max()}")
    seasons = sorted(df["season"].unique(), key=lambda s: int(s.split("-")[0]))
    print(f"Seasons: {len(seasons)} -> {seasons[:3]} ... {seasons[-3:]}")

    mps = df.groupby("season").size()
    print("\nMatches per season:")
    print(mps)

    def miss_pct(col: str) -> float:
        return float(df[col].isna().mean() * 100.0)

    print("\nMissing odds % (bet open):")
    print(f"  over25_bet:  {miss_pct('odds_over25_bet'):.2f}%")
    print(f"  under25_bet: {miss_pct('odds_under25_bet'):.2f}%")

    print("\nMissing odds % (reference close):")
    print(f"  over25_ref_close:  {miss_pct('odds_over25_ref_close'):.2f}%")
    print(f"  under25_ref_close: {miss_pct('odds_under25_ref_close'):.2f}%")

    label_err = ((df["home_goals"] + df["away_goals"] >= 3).astype(int) != df["over25"]).sum()
    print(f"\nOver25 label errors: {int(label_err)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_FILE)
    ap.add_argument("--league", default=DEFAULT_LEAGUE, help="E0 for EPL; set '' to disable filter")
    args = ap.parse_args()

    league = args.league.strip()
    if league == "":
        league = None

    ensure_dir(os.path.dirname(args.out))
    df = build_dataset(raw_dir=args.raw_dir, league=league)
    df.to_csv(args.out, index=False)

    print(f"Saved: {os.path.abspath(args.out)}")
    print_summary(df)

if __name__ == "__main__":
    main()
