"""
build_dataset.py

Goal:
- Read all Football-Data.co.uk season CSVs from data/raw/*.csv
- Build one clean EPL dataset with:
    season, date, home_team, away_team, home_goals, away_goals, over25,
    odds_over25_bet, odds_under25_bet,
    odds_over25_ref_close, odds_under25_ref_close,
    odds_over25_bet_close, odds_under25_bet_close
- Also includes backwards-compat aliases:
    odds_over25/odds_under25 (same as bet odds)
    odds_over25_close/odds_under25_close (same as ref close odds)

Recommended mapping for your header:
- Bet odds (open): B365>2.5 / B365<2.5
- Reference market (closing): AvgC>2.5 / AvgC<2.5
- Optional bet close: B365C>2.5 / B365C<2.5

Run:
    py src/build_dataset.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_RAW_DIR = "data/raw"
DEFAULT_OUT_FILE = "data/clean/epl_matches_clean.csv"
DEFAULT_LEAGUE = "E0"  # EPL in Football-Data.co.uk


# -----------------------------
# Column picking helpers
# -----------------------------
def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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
    """
    Football-Data dates vary (dd/mm/yy, dd/mm/yyyy, etc).
    dayfirst=True is correct for these files.
    """
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def infer_season_from_date(d: pd.Timestamp) -> Optional[str]:
    """
    EPL season labeling: YYYY-YYYY+1
    If month >= 7 => season starts that year (e.g., Aug 2010 => 2010-2011)
    Else => season starts previous year (e.g., May 2011 => 2010-2011)
    """
    if pd.isna(d):
        return None
    y = int(d.year)
    m = int(d.month)
    start = y if m >= 7 else (y - 1)
    return f"{start}-{start+1}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Main build
# -----------------------------
def build_dataset(
    raw_dir: str,
    out_file: str,
    league: Optional[str] = DEFAULT_LEAGUE,
) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Put your Football-Data season CSVs there."
        )

    frames = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="latin-1")

        df["__source_file"] = os.path.basename(fp)

        # Optional league filter (E0)
        if league is not None and "Div" in df.columns:
            df = df[df["Div"].astype(str).str.upper().eq(league.upper())].copy()

        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    # Required base cols
    required_base = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required_base if c not in raw.columns]
    if missing:
        raise RuntimeError(
            f"Missing required columns in raw data: {missing}. "
            f"Check you downloaded Football-Data match CSVs."
        )

    out = pd.DataFrame()

    out["date"] = parse_date_series(raw["Date"])
    out["home_team"] = raw["HomeTeam"].astype(str).str.strip()
    out["away_team"] = raw["AwayTeam"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(raw["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(raw["FTAG"], errors="coerce")

    # Label
    out["over25"] = ((out["home_goals"] + out["away_goals"]) >= 3).astype(int)

    # Season (from date)
    out["season"] = out["date"].apply(infer_season_from_date)

    # -----------------------------
    # Odds mapping (based on your header)
    # -----------------------------
    # Bet odds (open)
    BET_OVER = ["B365>2.5"]
    BET_UNDER = ["B365<2.5"]

    # Reasonable fallbacks (in case some seasons lack B365 OU columns)
    BET_OVER_FALLBACKS = ["P>2.5", "Max>2.5", "Avg>2.5", "BFE>2.5"]
    BET_UNDER_FALLBACKS = ["P<2.5", "Max<2.5", "Avg<2.5", "BFE<2.5"]

    # Reference close (market) odds: prefer average closing
    REF_CLOSE_OVER = ["AvgC>2.5", "MaxC>2.5", "PC>2.5", "B365C>2.5", "BFEC>2.5"]
    REF_CLOSE_UNDER = ["AvgC<2.5", "MaxC<2.5", "PC<2.5", "B365C<2.5", "BFEC<2.5"]

    # Optional: bet close (B365 close)
    BET_CLOSE_OVER = ["B365C>2.5"]
    BET_CLOSE_UNDER = ["B365C<2.5"]

    out["odds_over25_bet"] = first_non_null_series(raw, BET_OVER)  # primary
    out["odds_under25_bet"] = first_non_null_series(raw, BET_UNDER)

    # Fill missing bet odds with fallbacks (only where still NaN)
    over_fallback = first_non_null_series(raw, BET_OVER_FALLBACKS)
    under_fallback = first_non_null_series(raw, BET_UNDER_FALLBACKS)
    out.loc[out["odds_over25_bet"].isna(), "odds_over25_bet"] = over_fallback[out["odds_over25_bet"].isna()]
    out.loc[out["odds_under25_bet"].isna(), "odds_under25_bet"] = under_fallback[out["odds_under25_bet"].isna()]

    out["odds_over25_ref_close"] = first_non_null_series(raw, REF_CLOSE_OVER)
    out["odds_under25_ref_close"] = first_non_null_series(raw, REF_CLOSE_UNDER)

    out["odds_over25_bet_close"] = first_non_null_series(raw, BET_CLOSE_OVER)
    out["odds_under25_bet_close"] = first_non_null_series(raw, BET_CLOSE_UNDER)

    # Backwards-compat aliases (for your older scripts)
    out["odds_over25"] = out["odds_over25_bet"]
    out["odds_under25"] = out["odds_under25_bet"]
    out["odds_over25_close"] = out["odds_over25_ref_close"]
    out["odds_under25_close"] = out["odds_under25_ref_close"]

    # Keep file reference for debugging (optional)
    out["source_file"] = raw["__source_file"].astype(str)

    # Coerce odds to numeric cleanly
    num_cols = [
        "home_goals", "away_goals",
        "odds_over25_bet", "odds_under25_bet",
        "odds_over25_ref_close", "odds_under25_ref_close",
        "odds_over25_bet_close", "odds_under25_bet_close",
        "odds_over25", "odds_under25",
        "odds_over25_close", "odds_under25_close",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop rows missing essentials (DO NOT require reference close odds here)
    out = out.dropna(subset=[
        "date", "season",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "odds_over25_bet", "odds_under25_bet",
    ])

    # De-dup (some downloaded files can overlap)
    out = out.sort_values(["date", "home_team", "away_team"]).drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
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

    # Quick label sanity
    label_err = ((df["home_goals"] + df["away_goals"] >= 3).astype(int) != df["over25"]).sum()
    print(f"\nOver25 label errors: {int(label_err)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Folder containing season CSVs")
    ap.add_argument("--out", default=DEFAULT_OUT_FILE, help="Output clean CSV path")
    ap.add_argument("--league", default=DEFAULT_LEAGUE, help="League code (E0 for EPL). Use '' to disable filter.")
    args = ap.parse_args()

    league = args.league.strip()
    if league == "":
        league = None

    ensure_dir(os.path.dirname(args.out))

    df = build_dataset(raw_dir=args.raw_dir, out_file=args.out, league=league)
    df.to_csv(args.out, index=False)

    print(f"Saved: {os.path.abspath(args.out)}")
    print_summary(df)


if __name__ == "__main__":
    main()
