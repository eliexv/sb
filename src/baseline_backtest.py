import math
import numpy as np
import pandas as pd
from scipy.stats import poisson

DATA_FILE = "data/clean/epl_matches_clean.csv"

# Backtest configuration
TRAIN_END_SEASON = "2022-2023"   # train on 2010-2011 .. 2022-2023
TEST_START_SEASON = "2023-2024"  # test on 2023-2024 .. 2024-2025
TEST_END_SEASON = "2024-2025"

EV_THRESHOLD = 0.02             # only bet if EV >= 2%
STAKE_FRACTION = 0.0025         # 0.25% bankroll per bet
START_BANKROLL = 1000.0         # bankroll units (can be $)

MAX_GOALS = 10                  # score grid limit for Poisson sums

def season_leq(a: str, b: str) -> bool:
    # Compare seasons like "2019-2020" by start year
    return int(a.split("-")[0]) <= int(b.split("-")[0])

def season_between(s: str, start: str, end: str) -> bool:
    y = int(s.split("-")[0])
    return int(start.split("-")[0]) <= y <= int(end.split("-")[0])

def p_over25(lam_home: float, lam_away: float, max_goals: int = 10) -> float:
    """
    Compute P(total_goals >= 3) using independent Poisson for home/away.
    Sum P(h,a) for h+a >= 3 on grid 0..max_goals.
    """
    ph = poisson.pmf(np.arange(0, max_goals + 1), lam_home)
    pa = poisson.pmf(np.arange(0, max_goals + 1), lam_away)
    # Outer product gives joint probabilities
    joint = np.outer(ph, pa)
    total = np.add.outer(np.arange(0, max_goals + 1), np.arange(0, max_goals + 1))
    return float(joint[total >= 3].sum())

def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))

def run_backtest(df: pd.DataFrame):
    # Split train/test by season labels (no leakage)
    train = df[df["season"].apply(lambda s: season_leq(s, TRAIN_END_SEASON))].copy()
    test = df[df["season"].apply(lambda s: season_between(s, TEST_START_SEASON, TEST_END_SEASON))].copy()

    # Fit league-average lambdas on training data
    lam_home = train["home_goals"].mean()
    lam_away = train["away_goals"].mean()

    print(f"Training seasons: {train['season'].min()} .. {train['season'].max()}  (rows={len(train)})")
    print(f"Test seasons:     {test['season'].min()} .. {test['season'].max()}   (rows={len(test)})")
    print(f"Fitted lambdas:   lam_home={lam_home:.4f}, lam_away={lam_away:.4f}")

    # Predict probability for each match
    test["q_over25"] = [p_over25(lam_home, lam_away, MAX_GOALS) for _ in range(len(test))]

    # Probability metrics (not profit)
    y = test["over25"].to_numpy(dtype=float)
    p = test["q_over25"].to_numpy(dtype=float)
    print("\nProbability quality:")
    print(f"  Log loss:   {log_loss(y, p):.5f}")
    print(f"  Brier:      {brier_score(y, p):.5f}")
    print(f"  Mean p:     {p.mean():.5f}")
    print(f"  Base rate:  {y.mean():.5f}  (actual Over2.5 frequency)")

    # Betting rule
    # EV = q*O - 1
    test["ev"] = test["q_over25"] * test["odds_over25"] - 1.0
    bets = test[test["ev"] >= EV_THRESHOLD].copy()

    print("\nBet selection:")
    print(f"  EV threshold: {EV_THRESHOLD:.3f}")
    print(f"  Bets placed:  {len(bets)} out of {len(test)} matches ({len(bets)/len(test)*100:.2f}%)")
    if len(bets) == 0:
        print("  No bets met threshold. Lower EV_THRESHOLD or improve model (next step).")
        return

    # Flat staking backtest
    bankroll = START_BANKROLL
    peak = bankroll
    max_dd = 0.0

    profits = []
    stakes = []
    results = []

    for _, r in bets.sort_values("date").iterrows():
        stake = bankroll * STAKE_FRACTION
        O = float(r["odds_over25"])
        win = int(r["over25"]) == 1

        if win:
            profit = stake * (O - 1.0)
            results.append(1)
        else:
            profit = -stake
            results.append(0)

        bankroll += profit
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)

        profits.append(profit)
        stakes.append(stake)

    profits = np.array(profits)
    stakes = np.array(stakes)

    total_staked = float(stakes.sum())
    total_profit = float(profits.sum())
    roi = total_profit / total_staked if total_staked > 0 else 0.0

    print("\nBetting performance (flat staking):")
    print(f"  Start bankroll: {START_BANKROLL:.2f}")
    print(f"  End bankroll:   {bankroll:.2f}")
    print(f"  Total staked:   {total_staked:.2f}")
    print(f"  Total profit:   {total_profit:.2f}")
    print(f"  ROI:            {roi*100:.2f}%")
    print(f"  Win rate:       {np.mean(results)*100:.2f}%")
    print(f"  Avg EV (bets):  {bets['ev'].mean()*100:.2f}%")
    print(f"  Max drawdown:   {max_dd*100:.2f}%")

    # Save bet log for inspection
    out_cols = ["date","season","home_team","away_team","home_goals","away_goals","total_goals",
                "q_over25","odds_over25","ev","over25"]
    bets[out_cols].to_csv("data/clean/baseline_bets_log.csv", index=False)
    print("\nSaved bet log: data/clean/baseline_bets_log.csv")

def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    # basic safety
    df = df.sort_values("date").reset_index(drop=True)
    run_backtest(df)

if __name__ == "__main__":
    main()
