import numpy as np
import pandas as pd

BET_LOG = "data/clean/walkforward_bets_log.csv"

def log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())

def brier(y, p):
    return float(np.mean((p-y)**2))

df = pd.read_csv(BET_LOG, parse_dates=["date"])
df = df.dropna(subset=["odds_taken","odds_close"])

# Your model probability at time of bet (already in log as q_over25 if you kept it;
# if not, we can add it—see note below).
if "q_over25" not in df.columns:
    raise RuntimeError("walkforward_bets_log.csv needs q_over25 column. Add it to the logged columns.")

y = df["over25"].to_numpy(dtype=float)
p_model = df["q_over25"].to_numpy(dtype=float)

# Convert odds to implied probability (de-vig not possible with only over odds; still useful)
p_taken = 1.0 / df["odds_taken"].to_numpy(dtype=float)
p_close = 1.0 / df["odds_close"].to_numpy(dtype=float)

print("Bets:", len(df))
print("\nProbability diagnostics on BETS:")
print(f"MODEL  logloss={log_loss(y,p_model):.5f}  brier={brier(y,p_model):.5f}  mean_p={p_model.mean():.5f}")
print(f"TAKEN  logloss={log_loss(y,p_taken):.5f}  brier={brier(y,p_taken):.5f}  mean_p={p_taken.mean():.5f}")
print(f"CLOSE  logloss={log_loss(y,p_close):.5f}  brier={brier(y,p_close):.5f}  mean_p={p_close.mean():.5f}")
