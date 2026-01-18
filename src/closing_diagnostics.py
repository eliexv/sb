import numpy as np
import pandas as pd

BET_LOG = "data/clean/walkforward_bets_log.csv"

def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())

def brier(y, p):
    return float(np.mean((p-y)**2))

def devig_two_way(over_odds, under_odds):
    over = 1.0 / over_odds
    under = 1.0 / under_odds
    s = over + under
    return over / s

df = pd.read_csv(BET_LOG, parse_dates=["date"])
if "has_close" in df.columns:
    df = df[df["has_close"] == 1].copy()
df = df.dropna(subset=["odds_taken","odds_close"])

# Your model probability at time of bet (already in log as q_over25 if you kept it;
# if not, we can add it—see note below).
if "q_over25" not in df.columns:
    raise RuntimeError("walkforward_bets_log.csv needs q_over25 column. Add it to the logged columns.")

y = df["over25"].to_numpy(dtype=float)
p_model = df["q_over25"].to_numpy(dtype=float)

p_taken = devig_two_way(df["odds_taken"].to_numpy(float), df["odds_taken_under"].to_numpy(float))
p_close = devig_two_way(df["odds_close"].to_numpy(float), df["odds_close_under"].to_numpy(float))

print("Bets:", len(df))
print("\nProbability diagnostics on BETS:")
print(f"MODEL  logloss={log_loss(y,p_model):.5f}  brier={brier(y,p_model):.5f}  mean_p={p_model.mean():.5f}")
print(f"TAKEN  logloss={log_loss(y,p_taken):.5f}  brier={brier(y,p_taken):.5f}  mean_p={p_taken.mean():.5f}")
print(f"CLOSE  logloss={log_loss(y,p_close):.5f}  brier={brier(y,p_close):.5f}  mean_p={p_close.mean():.5f}")
