import re
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/clean/epl_matches_clean.csv")

OVER25_CANDIDATES = [
    "B365>2.5",
    "Avg>2.5",
    "Max>2.5",
    "P>2.5",
    "BbAv>2.5",
    "AvgO2.5",
    "B365O2.5",
]
UNDER25_CANDIDATES = [
    "B365<2.5",
    "Avg<2.5",
    "Max<2.5",
    "P<2.5",
    "BbAv<2.5",
    "AvgU2.5",
    "B365U2.5",
]

OVER25_CLOSE_CANDIDATES = [
    "B365C>2.5",
    "AvgC>2.5",
    "MaxC>2.5",
    "PC>2.5",
    "BbAvC>2.5",
    "AvgCO2.5",
    "B365CO2.5",
]
UNDER25_CLOSE_CANDIDATES = [
    "B365C<2.5",
    "AvgC<2.5",
    "MaxC<2.5",
    "PC<2.5",
    "BbAvC<2.5",
    "AvgCU2.5",
    "B365CU2.5",
]


def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def infer_season_from_filename(name: str) -> str | None:
    # tries patterns like 1011, 1920, 2122 etc
    m = re.search(r"(\d{2})(\d{2})", name)
    if not m:
        return None
    yy1, yy2 = int(m.group(1)), int(m.group(2))
    start_year = 2000 + yy1
    end_year = 2000 + yy2
    if end_year < start_year:
        end_year += 100
    return f"{start_year}-{end_year}"


def infer_season_from_file_dates(dates: pd.Series) -> str:
    mn = dates.min()
    if pd.isna(mn):
        raise ValueError("Could not infer season: all dates are NaT")
    start_year = mn.year if mn.month >= 7 else mn.year - 1
    return f"{start_year}-{start_year+1}"


def first_non_null_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    # build a series by taking the first non-null value across candidate columns
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series([pd.NA] * len(df), index=df.index)

    s = pd.to_numeric(df[existing[0]], errors="coerce")
    for c in existing[1:]:
        s2 = pd.to_numeric(df[c], errors="coerce")
        s = s.where(s.notna(), s2)
    return s


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR.resolve()}")

    all_rows = []

    for fp in files:
        df = pd.read_csv(fp)

        required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[SKIP] {fp.name}: missing required columns {missing}")
            continue

        out = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
        out.rename(
            columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
            },
            inplace=True,
        )

        out["date"] = parse_date_series(out["date"])

        # Odds with fallback chain (fix #2)
        out["odds_over25"] = first_non_null_series(df, OVER25_CANDIDATES)
        out["odds_under25"] = first_non_null_series(df, UNDER25_CANDIDATES)

        out["odds_over25_close"] = first_non_null_series(df, OVER25_CLOSE_CANDIDATES)
        out["odds_under25_close"] = first_non_null_series(df, UNDER25_CLOSE_CANDIDATES)

        # Infer season per file (fix #1)
        season = infer_season_from_filename(fp.stem)
        if season is None:
            season = infer_season_from_file_dates(out["date"])
        out["season"] = season

        # Convert to numeric
        for c in [
            "home_goals",
            "away_goals",
            "odds_over25",
            "odds_under25",
            "odds_over25_close",
            "odds_under25_close",
        ]:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        # Drop unusable rows
        out = out.dropna(
            subset=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
                "odds_over25",
                "odds_under25",
            ]
        )

        out["total_goals"] = out["home_goals"] + out["away_goals"]
        out["over25"] = (out["total_goals"] >= 3).astype(int)

        all_rows.append(out)

    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "home_team", "away_team"])
    merged = merged.sort_values(["date", "home_team", "away_team"]).reset_index(
        drop=True
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)

    print(f"Saved: {OUT_FILE.resolve()}")
    print(f"Rows: {len(merged)}")
    print("Matches per season:")
    print(merged.groupby("season").size())


if __name__ == "__main__":
    main()
