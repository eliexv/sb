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


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_dates_flexible(s: pd.Series) -> pd.Series:
    """
    Football-Data dates can be dd/mm/yy or dd/mm/yyyy (and sometimes inconsistent).
    This tries two explicit formats first (fast + consistent), then falls back.
    """
    s = s.astype(str).str.strip()

    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    m = d.isna()
    if m.any():
        d2 = pd.to_datetime(s[m], format="%d/%m/%y", errors="coerce")
        d.loc[m] = d2

    # final fallback (covers oddities)
    m = d.isna()
    if m.any():
        d3 = pd.to_datetime(s[m], dayfirst=True, errors="coerce")
        d.loc[m] = d3

    return d


def infer_season_from_date(d: pd.Timestamp) -> Optional[str]:
    # EPL seasons start in AUGUST. (month>=8 fixes COVID July matches)
    if pd.isna(d):
        return None
    y = int(d.year)
    m = int(d.month)
    start = y if m >= 8 else (y - 1)
    return f"{start}-{start+1}"


def coalesce_numeric(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """
    Row-wise coalesce across candidate columns (first non-null numeric per row).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.fillna(s)
    return out


def build_one_file(fp: str, league: Optional[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(fp)
    except UnicodeDecodeError:
        df = pd.read_csv(fp, encoding="latin-1")

    df["__source_file"] = os.path.basename(fp)

    if league is not None and "Div" in df.columns:
        df = df[df["Div"].astype(str).str.upper().eq(league.upper())].copy()

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Skip files that aren't match files
        return pd.DataFrame()

    out = pd.DataFrame()
    out["date"] = parse_dates_flexible(df["Date"])
    out["home_team"] = df["HomeTeam"].astype(str).str.strip()
    out["away_team"] = df["AwayTeam"].astype(str).str.strip()
    out["home_goals"] = pd.to_numeric(df["FTHG"], errors="coerce")
    out["away_goals"] = pd.to_numeric(df["FTAG"], errors="coerce")
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["over25"] = (out["total_goals"] >= 3).astype(int)
    out["season"] = out["date"].apply(infer_season_from_date)

    # Bet odds (open): prefer B365 -> Avg/Max/P/BFE -> BbAv/BbMx (older seasons)
    BET_OVER = [
        "B365>2.5", "Avg>2.5", "Max>2.5", "P>2.5", "BFE>2.5",
        "BbAv>2.5", "BbMx>2.5",
    ]
    BET_UNDER = [
        "B365<2.5", "Avg<2.5", "Max<2.5", "P<2.5", "BFE<2.5",
        "BbAv<2.5", "BbMx<2.5",
    ]

    # Reference close (only newer seasons)
    REF_CLOSE_OVER = ["AvgC>2.5", "MaxC>2.5", "PC>2.5", "B365C>2.5", "BFEC>2.5"]
    REF_CLOSE_UNDER = ["AvgC<2.5", "MaxC<2.5", "PC<2.5", "B365C<2.5", "BFEC<2.5"]

    out["odds_over25_bet"] = coalesce_numeric(df, BET_OVER)
    out["odds_under25_bet"] = coalesce_numeric(df, BET_UNDER)

    out["odds_over25_ref_close"] = coalesce_numeric(df, REF_CLOSE_OVER)
    out["odds_under25_ref_close"] = coalesce_numeric(df, REF_CLOSE_UNDER)

    out["odds_over25_bet_close"] = coalesce_numeric(df, ["B365C>2.5"])
    out["odds_under25_bet_close"] = coalesce_numeric(df, ["B365C<2.5"])

    out["has_ref_close"] = (
        out["odds_over25_ref_close"].notna() & out["odds_under25_ref_close"].notna()
    ).astype(int)

    # Backwards-compat aliases
    out["odds_over25"] = out["odds_over25_bet"]
    out["odds_under25"] = out["odds_under25_bet"]
    out["odds_over25_close"] = out["odds_over25_ref_close"]
    out["odds_under25_close"] = out["odds_under25_ref_close"]

    out["source_file"] = df["__source_file"].astype(str)

    # Drop essentials (DO NOT require ref close)
    out = out.dropna(subset=[
        "date", "season",
        "home_team", "away_team",
        "home_goals", "away_goals",
        "odds_over25_bet", "odds_under25_bet",
    ])

    return out


def build_dataset(raw_dir: str, league: Optional[str] = DEFAULT_LEAGUE) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    parts = []
    for fp in files:
        part = build_one_file(fp, league)
        if len(part) == 0:
            continue
        parts.append(part)

    if not parts:
        raise RuntimeError("No usable match rows were produced from the raw files.")

    out = pd.concat(parts, ignore_index=True)

    # De-dup across files (rare but safe)
    out = out.sort_values(["date", "home_team", "away_team"]).drop_duplicates(
        subset=["date", "home_team", "away_team"], keep="last"
    ).reset_index(drop=True)

    return out


def print_summary(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Min date: {df['date'].min()} Max date: {df['date'].max()}")

    seasons = sorted(df["season"].unique(), key=lambda s: int(s.split("-")[0]))
    print(f"Seasons: {len(seasons)} -> {seasons[:3]} ... {seasons[-3:]}")

    print("\nMatches per season:")
    print(df.groupby("season").size())

    print("\nShare with reference close available:")
    share = (df.groupby("season")["has_ref_close"].mean() * 100).round(1).astype(str) + "%"
    print(share)

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
