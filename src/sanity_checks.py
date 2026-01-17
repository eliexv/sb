import pandas as pd

df = pd.read_csv("data/clean/epl_matches_clean.csv", parse_dates=["date"])

print("Rows:", len(df))
print("Min date:", df["date"].min(), "Max date:", df["date"].max())
print("Matches per season (should be ~380 each):")
print(df.groupby("season").size())

# Odds sanity
print("\nOdds ranges:")
print(df[["odds_over25", "odds_under25"]].describe())

# Make sure over25 aligns with total goals
df["total_goals"] = df["home_goals"] + df["away_goals"]
bad = df[(df["over25"] == 1) & (df["total_goals"] < 3)]
bad2 = df[(df["over25"] == 0) & (df["total_goals"] >= 3)]
print("\nOver25 label errors:", len(bad) + len(bad2))
