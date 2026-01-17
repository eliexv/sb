import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

# -----------------------------
# Files
# -----------------------------
DATA_FILE = "data/clean/epl_matches_clean.csv"
SUMMARY_FILE = "data/clean/walkforward_summary.csv"
BET_LOG_FILE = "data/clean/walkforward_bets_log.csv"

# -----------------------------
# Walk-forward configuration
# -----------------------------
TRAIN_WINDOW_SEASONS = 5       # fit on last K seasons
USE_TIME_DECAY = True
HALF_LIFE_DAYS = 365           # ignored if USE_TIME_DECAY=False

RIDGE_ALPHA = 0.002
MAX_GOALS = 10

# -----------------------------
# Betting configuration (development only: flat staking)
# -----------------------------
EV_THRESHOLD = 0.02
STAKE_FRACTION = 0.0025
START_BANKROLL = 1000.0

# Eligibility filter: avoid teams with too little history in training window
N_MIN_MATCHES = 25

# -----------------------------
# Blending configuration
# -----------------------------
# Blend: p_blend = (1-w)*p_mkt_raw + w*p_model
# Regularize w toward BLEND_PRIOR_W (0 = "trust market unless model clearly helps")
BLEND_PRIOR_W = 0.0
BLEND_L2 = 0.25  # increase to shrink harder toward market; decrease to let w move more

# -----------------------------
# Helpers
# -----------------------------
def season_start_year(season: str) -> int:
    return int(season.split("-")[0])

def sorted_seasons(df: pd.DataFrame):
    return sorted(df["season"].unique(), key=season_start_year)

def p_over25(lam_home: float, lam_away: float, max_goals: int = 10) -> float:
    """P(total goals >= 3) under independent Poisson model."""
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
    """De-vig implied probability for Over2.5 using Over & Under odds (kept for optional diagnostics)."""
    po = 1.0 / row["odds_over25"]
    pu = 1.0 / row["odds_under25"]
    s = po + pu
    return po / s if s > 0 else np.nan

def market_raw_prob_over(row) -> float:
    """RAW implied probability from the price you can bet (no de-vig)."""
    o = row["odds_over25"]
    return (1.0 / o) if (o is not None and np.isfinite(o) and o > 0) else np.nan

def build_weights(dates: pd.Series) -> np.ndarray:
    if not USE_TIME_DECAY:
        return np.ones(len(dates), dtype=float)
    ref = dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / float(HALF_LIFE_DAYS))

def eligibility_filter(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Keep only matches where both teams have >= N_MIN_MATCHES appearances in training window."""
    home_counts = train["home_team"].value_counts()
    away_counts = train["away_team"].value_counts()
    team_counts = home_counts.add(away_counts, fill_value=0)

    t = test.copy()
    t["home_hist_matches"] = t["home_team"].map(team_counts).fillna(0).astype(int)
    t["away_hist_matches"] = t["away_team"].map(team_counts).fillna(0).astype(int)

    eligible = t[
        (t["home_hist_matches"] >= N_MIN_MATCHES) &
        (t["away_hist_matches"] >= N_MIN_MATCHES)
    ].copy()
    return eligible

# -----------------------------
# Model fitting
# -----------------------------
def fit_team_poisson(train: pd.DataFrame):
    """
    Fit:
      log(lambda_home) = home_adv + attack(home) + defense(away)
      log(lambda_away) = attack(away) + defense(home)
    Identifiability enforced by constraining last team's attack/defense so sums are 0.
    """
    teams = sorted(set(train["home_team"]).union(set(train["away_team"])))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_i = train["home_team"].map(idx).to_numpy()
    away_i = train["away_team"].map(idx).to_numpy()
    hg = train["home_goals"].to_numpy(dtype=float)
    ag = train["away_goals"].to_numpy(dtype=float)

    w = build_weights(train["date"]).astype(float)

    def unpack(x):
        home_adv = x[0]
        a_free = x[1:1 + (n - 1)]
        d_free = x[1 + (n - 1):1 + 2 * (n - 1)]
        attack = np.concatenate([a_free, [-a_free.sum()]])
        defense = np.concatenate([d_free, [-d_free.sum()]])
        return home_adv, attack, defense

    def nll(x):
        home_adv, attack, defense = unpack(x)
        lam_h = np.exp(home_adv + attack[home_i] + defense[away_i])
        lam_a = np.exp(attack[away_i] + defense[home_i])

        eps = 1e-12
        ll = (hg * np.log(lam_h + eps) - lam_h) + (ag * np.log(lam_a + eps) - lam_a)
        wll = (w * ll).sum()

        pen = RIDGE_ALPHA * (np.sum(attack ** 2) + np.sum(defense ** 2))
        return -(wll - pen)

    x0 = np.zeros(1 + 2 * (n - 1), dtype=float)
    res = minimize(
        nll,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 5000, "maxfun": 50000, "ftol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    home_adv, attack, defense = unpack(res.x)
    return teams, home_adv, attack, defense

def predict_lambdas(df_part: pd.DataFrame, teams, home_adv, attack, defense):
    idx = {t: i for i, t in enumerate(teams)}

    hi_series = df_part["home_team"].map(idx)
    ai_series = df_part["away_team"].map(idx)

    # Neutral slot for unseen teams
    attack_ext = np.append(attack, 0.0)
    defense_ext = np.append(defense, 0.0)
    neutral_idx = len(attack_ext) - 1

    hi = hi_series.fillna(neutral_idx).astype(int).to_numpy()
    ai = ai_series.fillna(neutral_idx).astype(int).to_numpy()

    lam_h = np.exp(home_adv + attack_ext[hi] + defense_ext[ai])
    lam_a = np.exp(attack_ext[ai] + defense_ext[hi])
    return lam_h, lam_a

def best_goal_scale(cal_lam_h, cal_lam_a, y_cal):
    """Fit a single multiplicative goal-rate scale s on the calibration season."""
    y_cal = np.asarray(y_cal, dtype=int)

    def obj(log_s):
        s = np.exp(log_s)
        p = np.array([p_over25(s * lh, s * la, MAX_GOALS) for lh, la in zip(cal_lam_h, cal_lam_a)])
        return log_loss(y_cal, p)

    # s in [0.50, 2.01]
    res = minimize_scalar(obj, bounds=(-0.7, 0.7), method="bounded")
    return float(np.exp(res.x))

def best_blend_weight(y, p_mkt_raw, p_model):
    """
    Find w in [0,1] minimizing:
      logloss(y, (1-w)*p_mkt_raw + w*p_model) + BLEND_L2*(w - BLEND_PRIOR_W)^2
    Also drops NaNs safely.
    """
    y = np.asarray(y, dtype=int)
    p_mkt_raw = np.asarray(p_mkt_raw, dtype=float)
    p_model = np.asarray(p_model, dtype=float)

    mask = np.isfinite(p_mkt_raw) & np.isfinite(p_model) & np.isfinite(y)
    y2 = y[mask]
    m2 = p_mkt_raw[mask]
    q2 = p_model[mask]

    if len(y2) < 50:
        # Too little data to estimate a stable w
        return 0.0

    def obj(w):
        p = (1.0 - w) * m2 + w * q2
        return log_loss(y2, p) + BLEND_L2 * (w - BLEND_PRIOR_W) ** 2

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    w = float(res.x)
    return max(0.0, min(1.0, w))

# -----------------------------
# Backtest staking (development)
# -----------------------------
def flat_stake_backtest(bets: pd.DataFrame, bankroll_start: float) -> dict:
    bankroll = bankroll_start
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

    total_staked = float(np.sum(stakes))
    total_profit = float(np.sum(profits))
    roi = total_profit / total_staked if total_staked > 0 else 0.0

    return {
        "end_bankroll": bankroll,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
        "win_rate": float(np.mean(results)) if results else np.nan,
        "max_dd": max_dd,
        "n_bets": len(bets),
    }

# -----------------------------
# Main
# -----------------------------
def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    required_cols = {
        "season", "date", "home_team", "away_team",
        "home_goals", "away_goals", "over25",
        "odds_over25", "odds_under25",
        "odds_over25_close", "odds_under25_close",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns in clean dataset: {sorted(missing)}")

    # Keep both versions around (raw for blending/EV comparisons; fair for optional diagnostics)
    df["p_mkt_raw"] = df.apply(market_raw_prob_over, axis=1)
    df["p_mkt_fair"] = df.apply(market_fair_prob_over, axis=1)

    seasons = sorted_seasons(df)

    min_needed = TRAIN_WINDOW_SEASONS + 2  # train window + cal + test
    if len(seasons) < min_needed:
        raise RuntimeError(
            f"Not enough seasons ({len(seasons)}) for train_window={TRAIN_WINDOW_SEASONS}. "
            f"Need at least {min_needed}."
        )

    # We start at: first season index that allows [train_window seasons] + [cal season] before it
    start_idx = TRAIN_WINDOW_SEASONS + 1
    end_idx = len(seasons) - 1

    bankroll = START_BANKROLL
    rows = []
    all_bets = []

    print(f"Seasons in data: {seasons[0]} .. {seasons[-1]}")
    print(
        f"Walk-forward: train_window={TRAIN_WINDOW_SEASONS} seasons | calibration=previous season | test=next season"
    )
    print(f"Time decay: {USE_TIME_DECAY} (half-life={HALF_LIFE_DAYS}) | ridge={RIDGE_ALPHA}")
    print(f"Eligibility: min matches={N_MIN_MATCHES} | EV threshold={EV_THRESHOLD}\n")

    for t in range(start_idx, end_idx + 1):
        test_season = seasons[t]
        cal_season = seasons[t - 1]
        train_seasons = seasons[t - 1 - TRAIN_WINDOW_SEASONS : t - 1]

        train = df[df["season"].isin(train_seasons)].copy()
        cal = df[df["season"] == cal_season].copy()
        test = df[df["season"] == test_season].copy()

        # Fit model on train
        teams, home_adv, attack, defense = fit_team_poisson(train)

        # Calibrate goal-scale on cal season
        cal_lh, cal_la = predict_lambdas(cal, teams, home_adv, attack, defense)
        y_cal = cal["over25"].to_numpy(dtype=int)
        s = best_goal_scale(cal_lh, cal_la, y_cal)

        # Predict on test
        lh, la = predict_lambdas(test, teams, home_adv, attack, defense)
        lh *= s
        la *= s
        test["p_model"] = [p_over25(a, b, MAX_GOALS) for a, b in zip(lh, la)]

        # Market probs for the test season
        test["p_mkt_raw"] = test.apply(market_raw_prob_over, axis=1)
        test["p_mkt_fair"] = test.apply(market_fair_prob_over, axis=1)

        # Eligibility subset (this defines the universe you actually bet in)
        eligible = eligibility_filter(train, test)

        # Blend weight fit on CAL season (use RAW market, because that's the tradable reference)
        # We fit w on the calibration season, then apply to the test season.
        cal["p_model"] = [p_over25(a * s, b * s, MAX_GOALS) for a, b in zip(cal_lh, cal_la)]
        cal["p_mkt_raw"] = cal.apply(market_raw_prob_over, axis=1)

        w = best_blend_weight(
            y=cal["over25"].to_numpy(dtype=int),
            p_mkt_raw=cal["p_mkt_raw"].to_numpy(dtype=float),
            p_model=cal["p_model"].to_numpy(dtype=float),
        )

        # Compute blended probs on eligible subset
        eligible = eligible.copy()
        eligible["p_model"] = eligible["p_model"].astype(float)
        eligible["p_mkt_raw"] = eligible["p_mkt_raw"].astype(float)
        eligible["p_blend"] = (1.0 - w) * eligible["p_mkt_raw"] + w * eligible["p_model"]

        # Metrics (on eligible subset, and NaN-safe)
        m = np.isfinite(eligible["p_mkt_raw"].to_numpy(float)) & np.isfinite(eligible["p_model"].to_numpy(float))
        y = eligible.loc[m, "over25"].to_numpy(dtype=int)
        p_blend = eligible.loc[m, "p_blend"].to_numpy(dtype=float)
        p_mkt = eligible.loc[m, "p_mkt_raw"].to_numpy(dtype=float)

        blend_ll = log_loss(y, p_blend) if len(y) else np.nan
        mkt_ll = log_loss(y, p_mkt) if len(y) else np.nan
        blend_bs = brier(y, p_blend) if len(y) else np.nan
        mkt_bs = brier(y, p_mkt) if len(y) else np.nan

        # Betting on eligible subset using MODEL probability (not blend) for EV
        # (If you want to switch EV to p_blend later, that’s one line.)
        eligible["ev"] = eligible["p_blend"] * eligible["odds_over25"] - 1.0
        bets = eligible[eligible["ev"] >= EV_THRESHOLD].copy()

        perf = flat_stake_backtest(bets, bankroll)
        bankroll = perf["end_bankroll"]

        print(
            f"{test_season} | train={train_seasons[0]}..{train_seasons[-1]} cal={cal_season} | "
            f"s={s:.3f} w={w:.3f} | LL(blend)={blend_ll:.4f} vs LL(mkt)={mkt_ll:.4f} | "
            f"bets={perf['n_bets']} | ROI={perf['roi']*100:.2f}%"
        )

        rows.append({
            "test_season": test_season,
            "train_start": train_seasons[0],
            "train_end": train_seasons[-1],
            "cal_season": cal_season,
            "scale_s": s,
            "blend_w": w,
            "eligible_matches": len(eligible),
            "logloss_blend": blend_ll,
            "logloss_market_raw": mkt_ll,
            "brier_blend": blend_bs,
            "brier_market_raw": mkt_bs,
            "bets": perf["n_bets"],
            "roi": perf["roi"],
            "bankroll_end": bankroll,
        })

        # Bet log for CLV + diagnostics
        if len(bets) > 0:
            bets = bets.copy()
            bets["odds_taken"] = bets["odds_over25"]
            bets["odds_close"] = bets["odds_over25_close"]  # CLV scripts drop NaNs
            bets["test_season"] = test_season

            bet_log_cols = [
                "date", "test_season", "home_team", "away_team",
                "p_model", "p_mkt_raw", "p_blend",
                "odds_taken", "odds_close",
                "ev", "over25"
            ]
            all_bets.append(bets[bet_log_cols])

    # Save summary
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSaved: {SUMMARY_FILE}")

    # Save bets log (always create the file, even if empty)
    if all_bets:
        bet_df = pd.concat(all_bets, ignore_index=True)
    else:
        bet_df = pd.DataFrame(columns=[
            "date", "test_season", "home_team", "away_team",
            "p_model", "p_mkt_raw", "p_blend",
            "odds_taken", "odds_close", "ev", "over25"
        ])
    bet_df.to_csv(BET_LOG_FILE, index=False)
    print(f"Saved: {BET_LOG_FILE}")

    print(f"Final bankroll: {bankroll:.2f}")

if __name__ == "__main__":
    main()
