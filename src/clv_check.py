import pandas as pd
import numpy as np

bets = pd.read_csv("data/clean/team_poisson_bets_log.csv", parse_dates=["date"])

bets = bets.dropna(subset=["odds_over25", "odds_over25_close"])

bets["clv_ratio"] = bets["odds_over25_close"] / bets["odds_over25"]
bets["log_clv"] = np.log(bets["odds_over25_close"]) - np.log(bets["odds_over25"])

print("Bets:", len(bets))
print("Mean CLV ratio:", bets["clv_ratio"].mean())
print("Mean log CLV:", bets["log_clv"].mean())
print("Pct beating close:", (bets["clv_ratio"] > 1).mean())
