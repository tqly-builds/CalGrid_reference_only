"""STEP 1: simulate the purchased day-ahead forecast (the "vendor"), then check it is realistic.

Run:   python sim/step1_vendor.py            (full run, about 5-15 min the first time: downloads)
       python sim/step1_vendor.py --quick    (tests every 3rd month only, to check it works)

Vendor models (both standard, both published):
  mlr   Tao Hong's regression with the recency effect and relative humidity:
        Trend + Weekday x Hour + Holiday x Hour + Month
        + Hour x (T, T^2, T^3) for T now, T 1-3 h earlier, and the average of the previous 24 h and 24-48 h
        + Month x (T, T^2, T^3) + Hour x (RH, RH^2, T x RH)
        + Hour x (solar radiation, its square, trend)   [rooftop solar lowers metered demand at midday]
        Hong, Pinson & Fan 2014 (vanilla benchmark); Wang, Liu & Hong 2016 (recency effect);
        Xie, Chen, Hong & Laing 2016 (relative humidity).
  gbm   Gradient boosting (machine learning, like Myst AI for EBCE) on the same weather and calendar
        inputs plus demand 7 and 14 days earlier.
  combo The average of mlr and gbm (forecast combination).
Training uses ACTUAL weather. Prediction uses the weather FORECAST that existed before 10 a.m. on D-1,
after removing its recent average error (bias correction, as weather vendors do).
Demand data: EIA-930 hours more than 40% away from the same hour's 2-week median are removed.
Walk-forward: each month from Jan 2024 is predicted by a model trained only on data before 10 a.m.
on the day before that month (expanding window from Jan 2021).

Realism check: compare each vendor's error with the published 3-BA forecast's error (same hours).
Also a "perfect weather" back-cast (mlr fed actual weather): Ava/Myst did the same test in 2021.
Results go to sim/results/ (step1_*.csv, step1_log.txt). Push that folder.
"""
import sys, time, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, __import__("os").path.dirname(__file__))
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="test every 3rd month only")
args = ap.parse_args()
sys.stdout = C.Tee(f"{C.RESULTS_DIR}/step1_log.txt")
t0 = time.time()
print(f"STEP 1 vendor simulation  (quick={args.quick})")

# ----------------------------- data -----------------------------
print("\n1) Demand + published forecast (EIA)")
key = C.load_key()
dem = C.load_demand(key)
print(f"   hours: {len(dem):,}  ({dem['period'].min()} -> {dem['period'].max()})")

print("\n2) Actual weather (Open-Meteo archive), 22 counties")
wx_act = C.load_actual_weather()
print("\n3) Forecast weather (Open-Meteo Previous Runs), 22 counties")
wx_fc_raw = C.load_forecast_weather()
print(f"   forecast hours using the 24-h-old run: {wx_fc_raw['use_d1'].mean() * 100:.0f}% (rest use the 48-h-old run)")
wx_fc = C.forecast_for_bid(wx_fc_raw, wx_act, mos=True)
wx_nomos = C.forecast_for_bid(wx_fc_raw, wx_act, mos=False)
_chk = wx_fc.merge(wx_act, on="period", suffixes=("", "_act")).merge(wx_nomos, on="period", suffixes=("", "_raw"))
print(f"   temperature forecast bias (forecast - actual): before correction {(_chk.temp_c_raw - _chk.temp_c_act).mean():+.2f} C, "
      f"after {(_chk.temp_c - _chk.temp_c_act).mean():+.2f} C")


# ----------------------------- features -----------------------------
def weather_feats(w):
    """Recency features on a complete hourly grid. T in deg F, scaled around 65 F."""
    w = w.set_index("period").sort_index()
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="h", tz="UTC"))
    w["temp_c"] = w["temp_c"].interpolate(limit=3)
    w["rh"] = w["rh"].interpolate(limit=3)
    w["ghi"] = w["ghi"].interpolate(limit=3)
    T = (w["temp_c"] * 9 / 5 + 32 - 65) / 15
    f = pd.DataFrame(index=w.index)
    f["T"] = T
    for k in (1, 2, 3):
        f[f"T_l{k}"] = T.shift(k)
    f["T_ma1"] = T.shift(1).rolling(24).mean()          # previous 24 h
    f["T_ma2"] = T.shift(25).rolling(24).mean()         # 24-48 h before
    f["RH"] = (w["rh"] - 50) / 25
    f["G"] = w["ghi"] / 1000                            # solar radiation, kW/m2 (rooftop solar)
    f["temp_c"], f["rh"], f["ghi"] = w["temp_c"], w["rh"], w["ghi"]
    return f.rename_axis("period").reset_index()


TEMP_VARS = ["T", "T_l1", "T_l2", "T_l3", "T_ma1", "T_ma2"]


def mlr_matrix(t):
    """Design matrix for Hong's recency + RH regression."""
    n = len(t)
    H = np.eye(24)[t["hour"].to_numpy()]
    M = np.eye(12)[t["month"].to_numpy() - 1]
    WH = np.eye(168)[(t["dayofweek"] * 24 + t["hour"]).to_numpy()]
    hol = t["is_holiday"].to_numpy(float)[:, None]
    trend = ((t["period"] - pd.Timestamp("2021-01-01", tz="UTC")).dt.days / 365.25).to_numpy()[:, None]
    cols = [np.ones((n, 1)), trend, WH, H * hol, M]
    for v in TEMP_VARS:
        x = t[v].to_numpy()[:, None]
        cols += [H * x, H * x ** 2, H * x ** 3]
    T = t["T"].to_numpy()[:, None]
    RH = t["RH"].to_numpy()[:, None]
    G = t["G"].to_numpy()[:, None]
    cols += [M * T, M * T ** 2, M * T ** 3, H * RH, H * RH ** 2, H * T * RH,
             H * G, H * G ** 2, H * trend]          # solar and its growth by hour (rooftop solar)
    return np.hstack(cols)


GBM_FEATS = ["temp_c", "rh", "ghi", "trend", "T_l1", "T_l2", "T_l3", "T_ma1", "T_ma2",
             "hour", "dayofweek", "month", "is_holiday", "lag168", "lag336"]


def frame(wx):
    f = dem.merge(weather_feats(wx), on="period", how="inner")
    lag_src = dem.set_index("period")["demand_for_lags"]
    f["lag168"] = f["period"].map(lag_src.shift(168))
    f["lag336"] = f["period"].map(lag_src.shift(336))
    f["is_holiday"] = f["is_holiday"].astype(int)
    f["trend"] = (f["period"] - pd.Timestamp("2021-01-01", tz="UTC")).dt.days
    return f.dropna(subset=TEMP_VARS + ["RH", "G", "lag168", "lag336"]).reset_index(drop=True)


print("\n4) Building model tables")
train_tab = frame(wx_act)                  # actual weather (training + perfect-weather back-cast)
test_tab = frame(wx_fc)                    # forecast weather (what the vendor really had)
train_tab = train_tab[train_tab["local_date"] >= pd.Timestamp(C.TRAIN_START)]
print(f"   actual-weather rows {len(train_tab):,} | forecast-weather rows {len(test_tab):,}")

# ----------------------------- walk-forward -----------------------------
months = pd.date_range(C.SIM_START, pd.Timestamp(C.END_DATE).replace(day=1), freq="MS")
if args.quick:
    months = months[::3]
print(f"\n5) Walk-forward over {len(months)} months")
preds = []
for m in months:
    ts = m.tz_localize(C.PT).tz_convert("UTC")
    te_end = (m + pd.offsets.MonthBegin(1)).tz_localize(C.PT).tz_convert("UTC")
    cutoff = ts - pd.Timedelta(hours=14)                 # 10 a.m. Pacific the day before the month
    tr = train_tab[(train_tab["period"] < cutoff) & train_tab["demand_mwh"].notna()]
    te = test_tab[(test_tab["period"] >= ts) & (test_tab["period"] < te_end)]
    te_perfect = train_tab.set_index("period").reindex(te["period"])
    if len(te) == 0:
        continue
    y = tr["demand_mwh"].to_numpy()
    mlr = Ridge(alpha=1.0, fit_intercept=False).fit(mlr_matrix(tr), y)
    gbm = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                        random_state=42).fit(tr[GBM_FEATS], y)
    out = te[["period", "local_date", "hour", "dayofweek", "month", "is_holiday",
              "demand_mwh", "published_mwh", "ciso_mwh", "ldwp_mwh", "banc_mwh",
              "ciso_df", "ldwp_df", "banc_df", "temp_c", "rh", "ghi"]].copy()
    out["vendor_mlr"] = mlr.predict(mlr_matrix(te))
    out["vendor_gbm"] = gbm.predict(te[GBM_FEATS])
    out["vendor_combo"] = (out["vendor_mlr"] + out["vendor_gbm"]) / 2
    ok = te_perfect[TEMP_VARS + ["RH", "G"]].notna().all(axis=1).to_numpy()
    pf = np.full(len(te), np.nan)
    if ok.any():
        pf[ok] = mlr.predict(mlr_matrix(te_perfect[ok].reset_index()))
    out["vendor_mlr_perfect_weather"] = pf
    out["temp_c_actual"] = te_perfect["temp_c"].to_numpy()
    out["fold"] = m.strftime("%Y-%m")
    preds.append(out)
    v = out.dropna(subset=["demand_mwh"])
    print(f"   {m:%Y-%m}: train {len(tr):6,d} h | MAPE mlr {C.metrics(v.demand_mwh, v.vendor_mlr)['MAPE_%']:5.2f}%"
          f"  gbm {C.metrics(v.demand_mwh, v.vendor_gbm)['MAPE_%']:5.2f}%"
          f"  published {C.metrics(v.demand_mwh, v.published_mwh)['MAPE_%']:5.2f}%")

P = pd.concat(preds, ignore_index=True)
P = P[P["local_date"] <= pd.Timestamp(C.END_DATE)]

# ----------------------------- realism check -----------------------------
METHODS = {"Published 3-BA forecast (yardstick)": "published_mwh",
           "Vendor mlr (Hong recency + RH)": "vendor_mlr",
           "Vendor gbm": "vendor_gbm",
           "Vendor combo (mlr+gbm average)": "vendor_combo",
           "Vendor mlr with PERFECT weather (diagnostic)": "vendor_mlr_perfect_weather"}
E = P.dropna(subset=["demand_mwh", *METHODS.values()]).copy()
E["period_group"] = np.where(E["local_date"] < pd.Timestamp("2025-01-01"),
                             "2024 (selection year)", "2025-01 to 2026-10-15 (test)")
rows = []
for grp, g in [("All", E), *E.groupby("period_group")]:
    for name, col in METHODS.items():
        rows.append({"period": grp, "method": name, **C.metrics(g["demand_mwh"], g[col])})
summary = pd.DataFrame(rows)

monthly = pd.DataFrame([{"month": f, "method": n, **C.metrics(g["demand_mwh"], g[c])}
                        for f, g in E.groupby("fold") for n, c in METHODS.items()])
monthly_wide = monthly.pivot(index="month", columns="method", values="MAPE_%")
by_hour = pd.DataFrame([{"hour": h, "method": n, **C.metrics(g["demand_mwh"], g[c])}
                        for h, g in E.groupby("hour") for n, c in METHODS.items()]) \
    .pivot(index="hour", columns="method", values="MAPE_%")
wx_err = (E["temp_c"] - E["temp_c_actual"]).abs()

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print("\n6) REALISM CHECK (same hours for every method)")
print(summary.to_string(index=False))
print("\nMonthly MAPE %:")
print(monthly_wide.round(2).to_string())
print(f"\nWeather forecast error used by the vendor: mean |forecast - actual| temperature = "
      f"{wx_err.mean():.2f} C (county-weighted)")

summary.to_csv(f"{C.RESULTS_DIR}/step1_summary.csv", index=False)
monthly.to_csv(f"{C.RESULTS_DIR}/step1_monthly.csv", index=False)
by_hour.round(3).to_csv(f"{C.RESULTS_DIR}/step1_by_hour.csv")
P.to_csv(f"{C.RESULTS_DIR}/step1_hourly.csv.gz", index=False, compression="gzip")
print(f"\nSaved sim/results/step1_*.csv  ({time.time() - t0:.0f}s). Now push sim/results/.")
