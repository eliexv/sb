import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

DATA_FILE = "data/clean/epl_matches_clean.csv"
BET_LOG_FILE = "data/clean/walkforward_bets_log.csv"

# Walk-forward config
TRAIN_WINDOW_SEASONS = 5      # fit on last K seasons
USE_TIME_DECAY = True
HALF_LIFE_DAYS = 365          # ignored if USE_TIME_DECAY=False

RIDGE_ALPHA = 0.002
MAX_GOALS = 10

# Betting config (keep flat staking while developing)
EV_THRESHOLD = 0.02
STAKE_FRACTION = 0.0025
START_BANKROLL = 1000.0

# Eligibility filter (avoid teams with little history in the training window)
N_MIN_MATCHES = 25

def season_start_year(season: str) -> int:
	return int(season.split("-")[0])

def sorted_seasons(df: pd.DataFrame):
	seasons = sorted(df["season"].unique(), key=season_start_year)
	return seasons

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
	po = 1.0 / row["odds_over25"]
	pu = 1.0 / row["odds_under25"]
	s = po + pu
	return po / s if s > 0 else np.nan

def build_weights(dates: pd.Series) -> np.ndarray:
	if not USE_TIME_DECAY:
		return np.ones(len(dates), dtype=float)
	ref = dates.max()
	age_days = (ref - dates).dt.days.to_numpy(dtype=float)
	return np.power(0.5, age_days / float(HALF_LIFE_DAYS))

def best_blend_weight(y, p_mkt, p_model):
    # minimize log loss of blended probs on calibration set
    def obj(w):
        p = (1 - w) * p_mkt + w * p_model
        return log_loss(y, p)
    res = minimize_scalar(obj, bounds=(0.0, 1.0), method="bounded")
    return float(res.x)

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
	# Optimize log scale to keep s positive and search stable
	def obj(log_s):
		s = np.exp(log_s)
		p = np.array([p_over25(s*lh, s*la, MAX_GOALS) for lh, la in zip(cal_lam_h, cal_lam_a)])
		return log_loss(y_cal, p)

	# s in [exp(-0.7), exp(0.7)] ~ [0.50, 2.01]
	res = minimize_scalar(obj, bounds=(-0.7, 0.7), method="bounded")
	return float(np.exp(res.x))

def eligibility_filter(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
	home_counts = train["home_team"].value_counts()
	away_counts = train["away_team"].value_counts()
	team_counts = home_counts.add(away_counts, fill_value=0)

	test = test.copy()
	test["home_hist_matches"] = test["home_team"].map(team_counts).fillna(0).astype(int)
	test["away_hist_matches"] = test["away_team"].map(team_counts).fillna(0).astype(int)

	eligible = test[(test["home_hist_matches"] >= N_MIN_MATCHES) &
					(test["away_hist_matches"] >= N_MIN_MATCHES)].copy()
	return eligible

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

def main():
	df = pd.read_csv(DATA_FILE, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
	df["p_mkt_over"] = df.apply(market_fair_prob_over, axis=1)

	seasons = sorted_seasons(df)

	# We need at least TRAIN_WINDOW + 1 season (for calibration) + 1 season (for test)
	start_idx = TRAIN_WINDOW_SEASONS + 1
	end_idx = len(seasons) - 1  # last season used as test at most

	bankroll = START_BANKROLL
	rows = []

	print(f"Seasons in data: {seasons[0]} .. {seasons[-1]}")
	print(f"Walk-forward: train_window={TRAIN_WINDOW_SEASONS} seasons | calibration=previous season | test=next season")
	print(f"Time decay: {USE_TIME_DECAY} (half-life={HALF_LIFE_DAYS}) | ridge={RIDGE_ALPHA}")
	print(f"Eligibility: min matches={N_MIN_MATCHES} | EV threshold={EV_THRESHOLD}\n")
 
	all_bets = []

	for t in range(start_idx, end_idx + 1):
		test_season = seasons[t]
		cal_season = seasons[t - 1]
		train_seasons = seasons[t - 1 - TRAIN_WINDOW_SEASONS : t - 1]  # window before calibration season

		train = df[df["season"].isin(train_seasons)].copy()
		cal = df[df["season"] == cal_season].copy()
		test = df[df["season"] == test_season].copy()

		# Fit team model on train
		teams, home_adv, attack, defense = fit_team_poisson(train)
  
  		# --- Build calibration probabilities to learn blend weight w ---
		cal = eligibility_filter(train, cal)  # use same eligibility logic

		cal_lh, cal_la = predict_lambdas(cal, teams, home_adv, attack, defense)
		cal_lh *= s
		cal_la *= s
		cal["p_model"] = [p_over25(a, b, MAX_GOALS) for a, b in zip(cal_lh, cal_la)]
		cal["p_mkt"] = cal["p_mkt_over"]

		y_cal = cal["over25"].to_numpy(dtype=int)
		w = best_blend_weight(y_cal, cal["p_mkt"].to_numpy(), cal["p_model"].to_numpy())

		# Calibrate goal scale on cal season
		cal_lh, cal_la = predict_lambdas(cal, teams, home_adv, attack, defense)
		y_cal = cal["over25"].to_numpy(dtype=int)
		s = best_goal_scale(cal_lh, cal_la, y_cal)

		# Predict on test
		lh, la = predict_lambdas(test, teams, home_adv, attack, defense)
		lh *= s
		la *= s

		test["q_over25"] = [p_over25(a, b, MAX_GOALS) for a, b in zip(lh, la)]

		test["p_model"] = test["q_over25"]
		test["p_mkt"] = test["p_mkt_over"]
		test["p_blend"] = (1 - w) * test["p_mkt"] + w * test["p_model"]

		# Eligibility subset
		eligible = eligibility_filter(train, test)

		# recompute these on eligible (because eligibility_filter returns a fresh copy)
		lh_e, la_e = predict_lambdas(eligible, teams, home_adv, attack, defense)
		lh_e *= s
		la_e *= s
		eligible["q_over25"] = [p_over25(a, b, MAX_GOALS) for a, b in zip(lh_e, la_e)]
		eligible["p_model"] = eligible["q_over25"]
		eligible["p_mkt"] = eligible["p_mkt_over"]
		eligible["p_blend"] = (1 - w) * eligible["p_mkt"] + w * eligible["p_model"]

		# Metrics on eligible (fair comparison to your betting universe)
		y = eligible["over25"].to_numpy(dtype=int)
		p_model = eligible["q_over25"].to_numpy(dtype=float)
		p_mkt = eligible["p_mkt_over"].to_numpy(dtype=float)

		model_ll = log_loss(y, p_model)
		mkt_ll = log_loss(y, p_mkt)
		model_bs = brier(y, p_model)
		mkt_bs = brier(y, p_mkt)

		# Betting
		eligible["ev"] = eligible["p_blend"] * eligible["odds_over25"] - 1.0
		bets = eligible[eligible["ev"] >= EV_THRESHOLD].copy()
		
		# Add closing odds to the bet log
		bets["odds_taken"] = bets["odds_over25"]
		bets["odds_close"] = bets["odds_over25_close"]  # make sure your clean file has this column
  
		bets = bets.dropna(subset=["odds_taken", "odds_close"])

		# Keep only the columns we need
		bet_log_cols = ["date","test_season","home_team","away_team",
                "p_model","p_mkt","p_blend","odds_taken","odds_close","ev","over25"]

		tmp = bets.copy()
		tmp["test_season"] = test_season
		tmp = tmp[bet_log_cols]

		# Append to a list; we will write once after the loop
		all_bets.append(tmp)

		perf = flat_stake_backtest(bets, bankroll)
		bankroll = perf["end_bankroll"]

		print(f"{test_season} | train={train_seasons[0]}..{train_seasons[-1]} cal={cal_season} | s={s:.3f} | "
			  f"LL(model)={model_ll:.4f} vs LL(mkt)={mkt_ll:.4f} | bets={perf['n_bets']} | ROI={perf['roi']*100:.2f}%")

		rows.append({
			"test_season": test_season,
			"train_start": train_seasons[0],
			"train_end": train_seasons[-1],
			"cal_season": cal_season,
			"scale_s": s,
			"eligible_matches": len(eligible),
			"logloss_model": model_ll,
			"logloss_market": mkt_ll,
			"brier_model": model_bs,
			"brier_market": mkt_bs,
			"bets": perf["n_bets"],
			"roi": perf["roi"],
			"bankroll_end": bankroll,
		})
	
	if all_bets:
		pd.concat(all_bets, ignore_index=True).to_csv(BET_LOG_FILE, index=False)
		print(f"\nSaved: {BET_LOG_FILE}")
	else:
		print("\nNo bets were placed; nothing to log.")


	out = pd.DataFrame(rows)
	out.to_csv("data/clean/walkforward_summary.csv", index=False)
	print("\nSaved: data/clean/walkforward_summary.csv")
	print(f"Final bankroll: {bankroll:.2f}")

if __name__ == "__main__":
	main()
