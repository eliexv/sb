import numpy as np
import pandas as pd

BET_LOG = "data/clean/walkforward_bets_log.csv"
N_BOOT = 20000

df = pd.read_csv(BET_LOG, parse_dates=["date"])
df = df.dropna(subset=["odds_taken","odds_close"])
df = df[(df["odds_taken"] > 0) & (df["odds_close"] > 0)]

log_clv = np.log(df["odds_close"].to_numpy()) - np.log(df["odds_taken"].to_numpy())

mean_log = log_clv.mean()
mean_ratio = np.exp(mean_log)

# bootstrap CI for mean log CLV
rng = np.random.default_rng(0)
boot = []
n = len(log_clv)
for _ in range(N_BOOT):
    sample = rng.choice(log_clv, size=n, replace=True)
    boot.append(sample.mean())
boot = np.array(boot)

lo, hi = np.quantile(boot, [0.025, 0.975])

print("Bets:", n)
print("Mean log CLV:", mean_log)
print("Mean CLV ratio (exp(mean log)):", mean_ratio)
print("95% CI mean log CLV:", (lo, hi))
print("95% CI mean CLV ratio:", (np.exp(lo), np.exp(hi)))
print("Pct beating close:", (log_clv > 0).mean())
