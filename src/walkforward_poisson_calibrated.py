import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

DATA_FILE = "data/clean/epl_matches_clean.csv"
SUMMARY_FILE = "data/clean/walkforward_summary.csv"
BET_LOG_FILE = "data/clean/walkforward_bets_log.csv"

TRAIN_WINDOW_SEASONS = 3
USE_TIME_DECAY = True
HALF_LIFE_DAYS = 365
RIDGE_ALPHA = 0.002
MAX_GOALS = 10

EV_THRESHOLD = 0.02
STAKE_FRACTION = 0.0025
START_BANKROLL = 1000.0

N_MIN_MATCHES = 25

# Regularize w towards market unless model clearly helps
BLEND_PRIOR_W = 0.0
BLEND_L2 = 0.25  # increase => w shrinks more toward 0


def season_start_year(season: str) -> int:
    return int(season.split("-")[0])


def sorted_seasons(df: pd.DataFrame):
    return sorted(df["season"].unique(), key=season_start_year)


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


def devig_over_prob(odds_over: np.ndarray, odds_under: np.ndarray) -> np.ndarray:
    """De-vig implied P(over) from paired OU odds."""
    po = 1.0 / odds_over
    pu = 1.0 / odds_under
    s = po + pu
    out = np.full_like(po, np.nan, dtype=float)
    m = np.isfinite(s) & (s > 0)
    out[m] = po[m] / s[m]
    return out


def build_weights(dates: pd.Series) -> np.ndarray:
    if not USE_TIME_DECAY:
        return np.ones(len(dates), dtype=float)
    ref = dates.max()
    age_days = (ref - dates).dt.days.to_numpy(dtype=float)
    return np.power(0.5, age_days / float(HALF_LIFE_DAYS))


def eligibility_filter(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    home_counts = train["home_team"].value_counts()
    away_counts = train["away_team"].value_counts()
    team_counts = home_counts.add(away_counts, fill_value=0)

    t = test.copy()
    t["home_hist_matches"] = t["home_team"].map(team_counts).fillna(0).astype(int)
    t["away_hist_matches"] = t["away_team"].map(team_counts).fillna(0).astype(int)

    return t[
        (t["home_hist_matches"] >= N_MIN_MATCHES)
        & (t["away_hist_matches"] >= N_MIN_MATCHES)
    ].copy()


def fit_team_poisson(train: pd.DataFrame):
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
        a_free = x[1 : 1 + (n - 1)]
        d_free = x[1 + (n - 1) : 1 + 2 * (n - 1)]
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

        pen = RIDGE_ALPHA * (np.sum(attack**2) + np.sum(defense**2))
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

    attack_ext = np.append(attack, 0.0)
    defense_ext = np.append(defense, 0.0)
    neutral_idx = len(attack_ext) - 1

    hi = hi_series.fillna(neutral_idx).astype(int).to_numpy()
    ai = ai_series.fillna(neutral_idx).astype(int).to_numpy()

    lam_h = np.exp(home_adv + attack_ext[hi] + defense_ext[ai])
    lam_a = np.exp(attack_ext[ai] + defense_ext[hi])
    return lam_h, lam_a


def best_goal_scale(cal_lam_h, cal_lam_a, y_cal):
    def obj(log_s):
        s = np.exp(log_s)
        p = np.array(
            [
                p_over25(s * lh, s * la, MAX_GOALS)
                for lh, la in zip(cal_lam_h, cal_lam_a)
            ]
        )
        return log_loss(y_cal, p)

    res = minimize_scalar(obj, bounds=(-0.7, 0.7), method="bounded")
    return float(np.exp(res.x))


def best_blend_weight(y, p_ref, p_model):
    y = np.asarray(y, dtype=int)
    p_ref = np.asarray(p_ref, dtype=float)
    p_model = np.asarray(p_model, dtype=float)

    mask = np.isfinite(y) & np.isfinite(p_ref) & np.isfinite(p_model)
    y2 = y[mask]
    r2 = p_ref[mask]
    q2 = p_model[mask]

    if len(y2) < 50:
        return 0.0

    def obj(w):
        p = (1.0 - w) * r2 + w * q2
        return log_loss(y2, p) + BLEND_L2 * (w - BLEND_PRIOR_W) ** 2

    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)


def flat_stake_backtest(bets: pd.DataFrame, bankroll_start: float) -> dict:
    bankroll = bankroll_start
    peak = bankroll
    max_dd = 0.0

    profits = []
    stakes = []
    results = []

    for _, r in bets.sort_values("date").iterrows():
        stake = bankroll * STAKE_FRACTION
        O = float(r["odds_over25_bet"])
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


def main():
    df = (
        pd.read_csv(DATA_FILE, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    required = {
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "over25",
        "odds_over25_bet",
        "odds_under25_bet",
        "odds_over25_ref_close",
        "odds_under25_ref_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns in clean dataset: {sorted(missing)}"
        )

    seasons = sorted_seasons(df)
    min_needed = TRAIN_WINDOW_SEASONS + 2
    if len(seasons) < min_needed:
        raise RuntimeError(
            f"Not enough seasons ({len(seasons)}) for train_window={TRAIN_WINDOW_SEASONS}. Need {min_needed}."
        )

    start_idx = TRAIN_WINDOW_SEASONS + 1
    end_idx = len(seasons) - 1

    bankroll = START_BANKROLL
    rows = []
    all_bets = []

    print(f"Seasons in data: {seasons[0]} .. {seasons[-1]}")
    print(
        f"Walk-forward: train_window={TRAIN_WINDOW_SEASONS} seasons | calibration=previous season | test=next season"
    )
    print(
        f"Time decay: {USE_TIME_DECAY} (half-life={HALF_LIFE_DAYS}) | ridge={RIDGE_ALPHA}"
    )
    print(f"Eligibility: min matches={N_MIN_MATCHES} | EV threshold={EV_THRESHOLD}\n")

    for t in range(start_idx, end_idx + 1):
        test_season = seasons[t]
        cal_season = seasons[t - 1]
        train_seasons = seasons[t - 1 - TRAIN_WINDOW_SEASONS : t - 1]

        train = df[df["season"].isin(train_seasons)].copy()
        cal_raw = df[df["season"] == cal_season].copy()
        test_raw = df[df["season"] == test_season].copy()

        teams, home_adv, attack, defense = fit_team_poisson(train)

        # Eligibility defines what you can bet on
        cal = eligibility_filter(train, cal_raw)
        test = eligibility_filter(train, test_raw)

        # -----------------------------
        # Market probability available at BET TIME (OPEN odds)
        # IMPORTANT: do NOT use closing odds here (that would be lookahead leakage).
        # -----------------------------
        cal_ref = devig_over_prob(
            cal["odds_over25_bet"].to_numpy(float),
            cal["odds_under25_bet"].to_numpy(float),
        )
        test_ref = devig_over_prob(
            test["odds_over25_bet"].to_numpy(float),
            test["odds_under25_bet"].to_numpy(float),
        )

        # -----------------------------
        # Closing-market probability (CLOSE odds) - EVALUATION ONLY
        # Used only for logging/diagnostics/CLV, never for EV selection.
        # -----------------------------
        cal_close = devig_over_prob(
            cal["odds_over25_ref_close"].to_numpy(float),
            cal["odds_under25_ref_close"].to_numpy(float),
        )
        test_close = devig_over_prob(
            test["odds_over25_ref_close"].to_numpy(float),
            test["odds_under25_ref_close"].to_numpy(float),
        )

        # Model calibration: scale s on calibration season
        cal_lh, cal_la = predict_lambdas(cal, teams, home_adv, attack, defense)
        y_cal = cal["over25"].to_numpy(int)
        s = best_goal_scale(cal_lh, cal_la, y_cal)

        # Model probs on cal/test with scale s
        cal_lh *= s
        cal_la *= s
        cal_model = np.array(
            [p_over25(a, b, MAX_GOALS) for a, b in zip(cal_lh, cal_la)]
        )

        lh, la = predict_lambdas(test, teams, home_adv, attack, defense)
        lh *= s
        la *= s
        test_model = np.array([p_over25(a, b, MAX_GOALS) for a, b in zip(lh, la)])

        # Blend weight w learned on calibration season vs reference close market
        w = best_blend_weight(y_cal, cal_ref, cal_model)

        # Blended probs for test
        test = test.copy()
        test["p_ref"] = test_ref
        test["p_model"] = test_model
        test["p_blend"] = (1.0 - w) * test["p_ref"] + w * test["p_model"]

        # Metrics vs reference close (NaN-safe)
        m = np.isfinite(test["p_ref"].to_numpy(float)) & np.isfinite(
            test["p_blend"].to_numpy(float)
        )
        y = test.loc[m, "over25"].to_numpy(int)
        ll_blend = (
            log_loss(y, test.loc[m, "p_blend"].to_numpy(float)) if len(y) else np.nan
        )
        ll_ref = log_loss(y, test.loc[m, "p_ref"].to_numpy(float)) if len(y) else np.nan
        ll_open = (
            log_loss(y, test.loc[m, "p_ref"].to_numpy(float)) if len(y) else np.nan
        )

        # also compute vs close market on the same subset (if finite)
        mc = np.isfinite(test_close) & m
        y_c = test.loc[mc, "over25"].to_numpy(int)
        ll_close = log_loss(y_c, test_close[mc]) if len(y_c) else np.nan

        # EV uses bettable odds (open) against blended "true prob"
        test["ev"] = test["p_blend"] * test["odds_over25_bet"] - 1.0
        bets = test[(test["ev"] >= EV_THRESHOLD) & np.isfinite(test["p_ref"])].copy()

        perf = flat_stake_backtest(bets, bankroll)
        bankroll = perf["end_bankroll"]

        print(
            f"{test_season} | train={train_seasons[0]}..{train_seasons[-1]} cal={cal_season} | "
            f"s={s:.3f} w={w:.3f} | LL(blend)={ll_blend:.4f} vs LL(open)={ll_open:.4f} vs LL(close)={ll_close:.4f} | "
            f"bets={perf['n_bets']} | ROI={perf['roi']*100:.2f}%"
        )

        rows.append(
            {
                "test_season": test_season,
                "train_start": train_seasons[0],
                "train_end": train_seasons[-1],
                "cal_season": cal_season,
                "scale_s": s,
                "blend_w": w,
                "eligible_matches": len(test),
                "logloss_blend": ll_blend,
                "logloss_ref": ll_ref,
                "bets": perf["n_bets"],
                "roi": perf["roi"],
                "bankroll_end": bankroll,
            }
        )

        # Bet log for CLV vs reference close
        if len(bets) > 0:
            bets = bets.copy()
            bets["odds_taken"] = bets["odds_over25_bet"]
            bets["odds_close"] = bets["odds_over25_ref_close"]
            bets["has_close"] = np.isfinite(bets["odds_close"]).astype(int)
            # Also log the UNDER odds so we can de-vig taken/close probabilities properly
            bets["odds_taken_under"] = bets["odds_under25_bet"]
            bets["odds_close_under"] = bets["odds_under25_ref_close"]
            bets["test_season"] = test_season
            bets["q_over25"] = bets["p_model"]

            bet_log_cols = [
                "date",
                "test_season",
                "home_team",
                "away_team",
                "p_ref",
                "p_model",
                "p_blend",
                "q_over25",
                "odds_taken",
                "odds_close",
                "odds_taken_under",
                "odds_close_under",
                "has_close",
                "ev",
                "over25",
            ]

            all_bets.append(bets[bet_log_cols])

    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    print(f"\nSaved: {SUMMARY_FILE}")

    if all_bets:
        pd.concat(all_bets, ignore_index=True).to_csv(BET_LOG_FILE, index=False)
    else:
        pd.DataFrame(
            columns=[
                "date",
                "test_season",
                "home_team",
                "away_team",
                "p_ref",
                "p_model",
                "p_blend",
                "q_over25",
                "odds_taken",
                "odds_close",
                "has_close",
                "ev",
                "over25",
            ]
        ).to_csv(BET_LOG_FILE, index=False)

    print(f"Saved: {BET_LOG_FILE}")
    print(f"Final bankroll: {bankroll:.2f}")


if __name__ == "__main__":
    main()
