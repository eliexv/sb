import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_FILE = "data/clean/epl_matches_clean.csv"

TRAIN_START_SEASON = "2018-2019"
TRAIN_END_SEASON = "2022-2023"
TEST_START_SEASON = "2023-2024"
TEST_END_SEASON = "2024-2025"

EV_THRESHOLD = 0.02
STAKE_FRACTION = 0.0025
START_BANKROLL = 1000.0
MAX_GOALS = 10

# Regularization strength (prevents absurd team params)
RIDGE_ALPHA = 0.002

# Optional time decay: set to None for no decay, or e.g. 365 for ~1-year half-life
HALF_LIFE_DAYS = 365  # you can try None, 365, 730 later


def season_leq(a: str, b: str) -> bool:
    return int(a.split("-")[0]) <= int(b.split("-")[0])


def season_between(s: str, start: str, end: str) -> bool:
    y = int(s.split("-")[0])
    return int(start.split("-")[0]) <= y <= int(end.split("-")[0])


def p_over25(lam_home: float, lam_away: float, max_goals: int = 10) -> float:
    ph = poisson.pmf(np.arange(0, max_goals + 1), lam_home)
    pa = poisson.pmf(np.arange(0, max_goals + 1), lam_away)
    joint = np.outer(ph, pa)
    total = np.add.outer(np.arange(0, max_goals + 1), np.arange(0, max_goals + 1))
    return float(joint[total >= 3].sum())


def log_loss(y, p, eps=1e-15) -> float:
    p = np.clip(p, eps, 1 - eps)
    y = y.astype(float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def brier(y, p) -> float:
    y = y.astype(float)
    return float(np.mean((p - y) ** 2))


def market_fair_prob_over(row) -> float:
    # Remove overround using over & under odds
    po = 1.0 / row["odds_over25"]
    pu = 1.0 / row["odds_under25"]
    s = po + pu
    return po / s if s > 0 else np.nan


def build_weights(dates: pd.Series) -> np.ndarray:
    if HALF_LIFE_DAYS is None:
        return np.ones(len(dates), dtype=float)
    # More recent matches get higher weight. Use last train date as reference.
    ref = dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    # weight = 0.5^(age/half_life)
    return np.power(0.5, age_days / float(HALF_LIFE_DAYS))


def fit_team_poisson(train: pd.DataFrame):
    teams = sorted(set(train["home_team"]).union(set(train["away_team"])))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_i = train["home_team"].map(idx).to_numpy()
    away_i = train["away_team"].map(idx).to_numpy()
    hg = train["home_goals"].to_numpy(dtype=float)
    ag = train["away_goals"].to_numpy(dtype=float)

    w = build_weights(train["date"]).astype(float)

    # Reparameterization for identifiability:
    # We optimize attacks[0:n-1] and defenses[0:n-1], then set last = -sum(others) so sums are 0.
    def unpack(x):
        home_adv = x[0]
        a_free = x[1 : 1 + (n - 1)]
        d_free = x[1 + (n - 1) : 1 + 2 * (n - 1)]
        attack = np.concatenate([a_free, [-a_free.sum()]])
        defense = np.concatenate([d_free, [-d_free.sum()]])
        return home_adv, attack, defense

    # Negative weighted log-likelihood with ridge penalty
    def nll(x):
        home_adv, attack, defense = unpack(x)

        lam_h = np.exp(home_adv + attack[home_i] + defense[away_i])
        lam_a = np.exp(attack[away_i] + defense[home_i])

        # Poisson log-likelihood (ignore constants log(y!) since they don't affect optimum much)
        eps = 1e-12
        ll = (hg * np.log(lam_h + eps) - lam_h) + (ag * np.log(lam_a + eps) - lam_a)
        wll = (w * ll).sum()

        # ridge penalty on team params (not on home_adv)
        pen = RIDGE_ALPHA * (np.sum(attack**2) + np.sum(defense**2))
        return -(wll - pen)

    x0 = np.zeros(1 + 2 * (n - 1), dtype=float)
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        options={
            "maxiter": 5000,
            "maxfun": 50000,
            "ftol": 1e-9,
        },
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    home_adv, attack, defense = unpack(res.x)
    return teams, home_adv, attack, defense, res.fun


def run():
    df = (
        pd.read_csv(DATA_FILE, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    train = df[
        df["season"].apply(
            lambda s: season_between(s, TRAIN_START_SEASON, TRAIN_END_SEASON)
        )
    ].copy()
    test = df[
        df["season"].apply(
            lambda s: season_between(s, TEST_START_SEASON, TEST_END_SEASON)
        )
    ].copy()

    print(
        f"Training: {train['season'].min()} .. {train['season'].max()} rows={len(train)}"
    )
    print(
        f"Testing:  {test['season'].min()} .. {test['season'].max()} rows={len(test)}"
    )
    print(f"Half-life days: {HALF_LIFE_DAYS} | Ridge alpha: {RIDGE_ALPHA}")

    # Fit model on train
    teams, home_adv, attack, defense, obj = fit_team_poisson(train)
    idx = {t: i for i, t in enumerate(teams)}

    # Map test teams -> indices (handle unseen teams)
    hi_series = test["home_team"].map(idx)
    ai_series = test["away_team"].map(idx)

    missing = sorted(
        set(test.loc[hi_series.isna(), "home_team"]).union(
            set(test.loc[ai_series.isna(), "away_team"])
        )
    )
    if missing:
        print("\nWARNING: Teams in test not seen in training (or name mismatch):")
        for t in missing:
            print("  -", t)
        print(
            "These matches will use neutral team parameters (0 attack/defense) for missing teams."
        )

    attack_ext = np.append(attack, 0.0)
    defense_ext = np.append(defense, 0.0)
    neutral_idx = len(attack_ext) - 1

    hi = hi_series.fillna(neutral_idx).astype(int).to_numpy()
    ai = ai_series.fillna(neutral_idx).astype(int).to_numpy()

    lam_h = np.exp(home_adv + attack_ext[hi] + defense_ext[ai])
    lam_a = np.exp(attack_ext[ai] + defense_ext[hi])

    print("\nGoal rate sanity:")
    print(
        f"  Pred mean home goals: {lam_h.mean():.3f} | Actual: {test['home_goals'].mean():.3f}"
    )
    print(
        f"  Pred mean away goals: {lam_a.mean():.3f} | Actual: {test['away_goals'].mean():.3f}"
    )
    print(
        f"  Pred mean total:      {(lam_h+lam_a).mean():.3f} | Actual: {(test['home_goals']+test['away_goals']).mean():.3f}"
    )

    # Predictions on FULL test
    test["q_over25"] = [p_over25(lh, la, MAX_GOALS) for lh, la in zip(lam_h, lam_a)]
    test["p_mkt_over"] = test.apply(market_fair_prob_over, axis=1)

    # Probability metrics on FULL test
    y = test["over25"].to_numpy(dtype=int)
    p_model = test["q_over25"].to_numpy(dtype=float)
    p_mkt = test["p_mkt_over"].to_numpy(dtype=float)

    print("\nProbability quality (MODEL):")
    print(f"  Log loss: {log_loss(y, p_model):.5f}")
    print(f"  Brier:    {brier(y, p_model):.5f}")
    print(f"  Mean p:   {p_model.mean():.5f} | Base rate: {y.mean():.5f}")

    print("\nProbability quality (MARKET, de-vigged using over/under):")
    print(f"  Log loss: {log_loss(y, p_mkt):.5f}")
    print(f"  Brier:    {brier(y, p_mkt):.5f}")
    print(f"  Mean p:   {np.nanmean(p_mkt):.5f} | Base rate: {y.mean():.5f}")

    # Eligibility filter (do AFTER q_over25 exists)
    N_MIN_MATCHES = 25
    home_counts = train["home_team"].value_counts()
    away_counts = train["away_team"].value_counts()
    team_counts = home_counts.add(away_counts, fill_value=0)

    test["home_hist_matches"] = test["home_team"].map(team_counts).fillna(0).astype(int)
    test["away_hist_matches"] = test["away_team"].map(team_counts).fillna(0).astype(int)

    eligible = test[
        (test["home_hist_matches"] >= N_MIN_MATCHES)
        & (test["away_hist_matches"] >= N_MIN_MATCHES)
    ].copy()

    # Probability metrics on ELIGIBLE subset (the one you actually bet from)
    y_e = eligible["over25"].to_numpy(dtype=int)
    p_model_e = eligible["q_over25"].to_numpy(dtype=float)
    p_mkt_e = eligible["p_mkt_over"].to_numpy(dtype=float)

    print("\nProbability quality on ELIGIBLE subset only:")
    print(
        f"  MODEL  Log loss: {log_loss(y_e, p_model_e):.5f} | Brier: {brier(y_e, p_model_e):.5f} | Mean p: {p_model_e.mean():.5f} | Base: {y_e.mean():.5f}"
    )
    print(
        f"  MARKET Log loss: {log_loss(y_e, p_mkt_e):.5f} | Brier: {brier(y_e, p_mkt_e):.5f} | Mean p: {np.nanmean(p_mkt_e):.5f} | Base: {y_e.mean():.5f}"
    )

    print(f"\nEligibility filter (min matches in training = {N_MIN_MATCHES}):")
    print(f"  Eligible matches: {len(eligible)} / {len(test)}")

    # Betting rule on eligible
    eligible["ev"] = eligible["q_over25"] * eligible["odds_over25"] - 1.0
    bets = eligible[eligible["ev"] >= EV_THRESHOLD].copy()

    print("\nBet selection:")
    print(f"  EV threshold: {EV_THRESHOLD:.3f}")
    print(
        f"  Bets placed:  {len(bets)} / {len(eligible)} ({(len(bets)/len(eligible)*100 if len(eligible) else 0):.2f}%)"
    )

    if len(bets) == 0:
        print(
            "  No bets met threshold. Try lowering EV_THRESHOLD (e.g. 0.01) or improving model."
        )
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

        profit = stake * (O - 1.0) if win else -stake
        bankroll += profit

        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak
        max_dd = max(max_dd, dd)

        profits.append(profit)
        stakes.append(stake)
        results.append(1 if win else 0)

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

    out_cols = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "total_goals",
        "q_over25",
        "p_mkt_over",
        "odds_over25",
        "odds_under25",
        "odds_over25_close",
        "odds_under25_close",
        "ev",
        "over25",
    ]
    bets[out_cols].to_csv("data/clean/team_poisson_bets_log.csv", index=False)
    print("\nSaved bet log: data/clean/team_poisson_bets_log.csv")


if __name__ == "__main__":
    run()
