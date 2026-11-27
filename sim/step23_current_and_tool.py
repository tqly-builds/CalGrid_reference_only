"""STEPS 2 + 3: the current process on top of the purchased (vendor) forecast, then the tool.

Run (after step1_vendor.py):   python sim/step23_current_and_tool.py

STEP 2  Current process. The user corrects the purchased forecast with judgement and experience;
        the exact steps are not known. So we test the standard recent-history corrections
        (ratio or MWh gap; 3-91 days; mean, median, weighted; same weekday; weekday/weekend)
        and use the strongest on 2024 only (before any test month). Each correction uses only
        days up to D-2 (complete before the 10 a.m. bid on D-1). Schedule = correction x 1.05.
STEP 3  Tool. A correction model on top of the current process:
        target = actual / current-process forecast; inputs = bid-time weather forecast
        (temperature, humidity, solar), hour and weekday, holiday, demand 7 and 14 days earlier,
        the purchased forecast, and the current-process forecast. Walk-forward: each month of
        Jan 2025 - Oct 2026 is predicted by models trained only on earlier data; model settings
        are picked inside each fold (last 28 days of its training data). Schedule = tool x 1.05.
Outputs (sim/results/):
  step23_log.txt, step23_asis_selection_2024.csv, step23_asis_test_report.csv,
  step23_summary.csv, step23_monthly.csv, step23_model_choices.csv, step23_hourly.csv.gz
  tableau/calgrid_plan_tableau.csv and tableau/calgrid_compare_hourly.csv: same file names,
  columns and types as the current Tableau files (checked against 09_data/ before saving).
"""
import os, sys, time
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# ============================ SETTINGS ============================
BUFFER = 1.05
ASIS_SELECT_END = "2024-12-31"
FIRST_TEST_MONTH = "2025-01"
INNER_VAL_DAYS = 28
TRACK_DAYS, TEMP_DAYS, MIN_DAYS = 30, 28, 14
TABLEAU_REF = os.path.join(C.HERE, "..", "09_data")          # current Tableau CSVs (schema reference)
# ==================================================================
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--test-missing", metavar="YYYY-MM-DD:H1-H2",
                help="TEST ONLY: blank the purchased forecast for hours H1..H2 of one day, to see the "
                     "status warning. Writes everything to sim/results/test_missing/ (normal files untouched).")
args = ap.parse_args()
OUT = os.path.join(C.RESULTS_DIR, "test_missing") if args.test_missing else C.RESULTS_DIR
TABLEAU_OUT = os.path.join(OUT, "tableau")
os.makedirs(TABLEAU_OUT, exist_ok=True)
sys.stdout = C.Tee(f"{OUT}/step23_log.txt")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
t0 = time.time()
print("STEPS 2 + 3: current process and tool on the simulated purchased forecast")

# ----------------------------- data -----------------------------
s1 = pd.read_csv(f"{C.RESULTS_DIR}/step1_hourly.csv.gz", parse_dates=["period"])
s1sum = pd.read_csv(f"{C.RESULTS_DIR}/step1_summary.csv")
pick = s1sum[(s1sum["period"] == "2024 (selection year)") & s1sum["method"].str.startswith("Vendor")
             & ~s1sum["method"].str.contains("PERFECT")].sort_values("MAPE_%").iloc[0]
VENDOR_COL = {"Vendor mlr (Hong recency + RH)": "vendor_mlr", "Vendor gbm": "vendor_gbm",
              "Vendor combo (mlr+gbm average)": "vendor_combo"}[pick["method"]]
print(f"Purchased forecast = {pick['method']} (lowest 2024 MAPE in step 1: {pick['MAPE_%']:.2f}%)")

dem = C.load_demand(C.load_key())                              # cached; gives lags before 2024
lag_src = dem.set_index("period")["demand_for_lags"]
D = s1[["period", "temp_c", "rh", "ghi", VENDOR_COL]].rename(columns={VENDOR_COL: "vendor_mwh"})
D = D.merge(dem.drop(columns=["published_mwh"]), on="period", how="left")
D["lag168"] = D["period"].map(lag_src.shift(168))
D["lag336"] = D["period"].map(lag_src.shift(336))
D["is_holiday"] = D["is_holiday"].astype(int)
D = D.sort_values("period").reset_index(drop=True)
if args.test_missing:
    day, hrs = args.test_missing.split(":")
    h1, h2 = (int(x) for x in hrs.split("-"))
    cut = (D["local_date"] == pd.Timestamp(day)) & D["hour"].between(h1, h2)
    D.loc[cut, "vendor_mwh"] = np.nan
    print(f"TEST: purchased forecast blanked for {int(cut.sum())} hours on {day} ({h1}:00-{h2}:00)")
# Missing purchased forecast (vendor file incomplete): use that hour's demand one week earlier,
# the same rule v10 used for a missing published forecast, and flag it for the status tile.
D["vendor_missing"] = D["vendor_mwh"].isna()
D["vendor_mwh"] = D["vendor_mwh"].fillna(D["period"].map(lag_src.shift(168)))
print(f"Purchased forecast missing (filled with last week's demand): {int(D['vendor_missing'].sum())} hours")
print(f"Hours: {len(D):,} ({D['local_date'].min().date()} -> {D['local_date'].max().date()}); "
      f"actual demand missing (EIA errors removed): {int(D['demand_mwh'].isna().sum())}")

# ----------------------------- STEP 2: current process -----------------------------
print("\nSTEP 2: current-process corrections of the purchased forecast")


def tab(col):
    return D.pivot_table(index="local_date", columns="hour", values=col, aggfunc="mean").asfreq("D")


base_t, act_t = tab("vendor_mwh"), tab("demand_mwh")
ratio_t, gap_t = act_t / base_t, act_t - base_t
we = pd.DataFrame(np.repeat((ratio_t.index.dayofweek >= 5)[:, None], ratio_t.shape[1], axis=1),
                  index=ratio_t.index, columns=ratio_t.columns)


def roll(t, n, how="mean"):            # last n days ending D-2; needs 70% of the days present
    r = t.rolling(n, min_periods=max(2, int(np.ceil(0.7 * n))))
    return (r.mean() if how == "mean" else r.median()).shift(2)


def ewm(t, hl):
    return t.ewm(halflife=hl, min_periods=28, ignore_na=True).mean().shift(2)


def weekday(t, k):                     # same hour, same weekday, last k weeks (D-7, D-14, ...)
    return pd.concat([t.shift(7 * j) for j in range(1, k + 1)]).groupby(level=0).mean() \
        .where(sum(t.shift(7 * j).notna() for j in range(1, k + 1)) >= int(np.ceil(k / 2)))


def daytype(t, n=28):                  # same hour, weekday days or weekend days only
    wd = t.where(~we).rolling(n, min_periods=4).mean().shift(2)
    wk = t.where(we).rolling(n, min_periods=4).mean().shift(2)
    return wd.where(~we, wk)


ones = pd.DataFrame(1.0, index=ratio_t.index, columns=ratio_t.columns)
VARIANTS = {
    "none (purchased forecast only)": ("ratio", ones),
    **{f"ratio_{n}": ("ratio", roll(ratio_t, n)) for n in [3, 7, 14, 28, 56, 91]},
    "ratio_median_28": ("ratio", roll(ratio_t, 28, "median")),
    "ratio_weighted_hl7": ("ratio", ewm(ratio_t, 7)),
    "ratio_weighted_hl14": ("ratio", ewm(ratio_t, 14)),
    "ratio_same_weekday_4wk": ("ratio", weekday(ratio_t, 4)),
    "ratio_same_weekday_8wk": ("ratio", weekday(ratio_t, 8)),
    "ratio_daytype_28": ("ratio", daytype(ratio_t)),
    **{f"add_{n}": ("add", roll(gap_t, n)) for n in [3, 7, 14, 28, 56]},
    "add_median_28": ("add", roll(gap_t, 28, "median")),
    "add_weighted_hl7": ("add", ewm(gap_t, 7)),
    "add_same_weekday_4wk": ("add", weekday(gap_t, 4)),
    "add_daytype_28": ("add", daytype(gap_t)),
}
key = D[["local_date", "hour"]]
A = D[["period", "local_date", "hour", "vendor_mwh", "demand_mwh"]].copy()
for name, (kind, t) in VARIANTS.items():
    f = key.merge(t.stack().rename("f").reset_index(), on=["local_date", "hour"], how="left")["f"].to_numpy()
    A[name] = D["vendor_mwh"].to_numpy() * f if kind == "ratio" else D["vendor_mwh"].to_numpy() + f
NAMES = list(VARIANTS)


def score(rows):
    out, a = [], rows["demand_mwh"]
    for n in NAMES:
        b = rows[n]; s = BUFFER * b
        out.append({"current-process variant": n, "MAPE_%": (np.abs(b - a) / a).mean() * 100,
                    "over_MWh_x1.05": (s - a).clip(lower=0).sum(), "short_MWh_x1.05": (a - s).clip(lower=0).sum(),
                    "short_hours_x1.05": int((s < a).sum())})
    return pd.DataFrame(out).sort_values("MAPE_%").reset_index(drop=True)


ok = A[NAMES].notna().all(axis=1) & A["demand_mwh"].notna()
sel = A[ok & (A["local_date"] <= pd.Timestamp(ASIS_SELECT_END))]
tst = A[ok & (A["local_date"] >= pd.Timestamp(FIRST_TEST_MONTH + "-01"))]
print(f"Selection: {sel['local_date'].min().date()} -> {sel['local_date'].max().date()} "
      f"({sel['local_date'].nunique()} days with every variant available)")
asis_sel = score(sel)
print(asis_sel.round(3).to_string(index=False))
asis_test = score(tst)
print("\nSame variants on the test period (REPORT ONLY, not used for the choice):")
print(asis_test.round(3).to_string(index=False))
asis_sel.to_csv(f"{OUT}/step23_asis_selection_2024.csv", index=False)
asis_test.to_csv(f"{OUT}/step23_asis_test_report.csv", index=False)
CHOSEN = asis_sel.iloc[0]["current-process variant"]
D["asis_mwh"] = A[CHOSEN].fillna(D["vendor_mwh"])
D["asis_factor"] = D["asis_mwh"] / D["vendor_mwh"]
print(f"\nCurrent process used from here on: {CHOSEN}")

# ----------------------------- STEP 3: tool -----------------------------
print("\nSTEP 3: tool (correction model), walk-forward")
HOUR_COLS = [f"hour_{h}" for h in range(24)]
DOW_COLS = [f"dow_{d}" for d in range(7)]
for h in range(24):
    D[f"hour_{h}"] = (D["hour"] == h).astype(int)
for d in range(7):
    D[f"dow_{d}"] = (D["dayofweek"] == d).astype(int)
FEATS = ["temp_c", "rh", "ghi", *HOUR_COLS, *DOW_COLS, "is_holiday", "lag168", "lag336",
         "vendor_mwh", "asis_mwh", "asis_factor"]
D["target"] = D["demand_mwh"] / D["asis_mwh"]


def candidates():
    c = [("Linear Regression", "default", LinearRegression())]
    for a in [1, 10, 100, 1000]:
        c.append(("Ridge Regression", f"alpha={a}", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=a))])))
    for dep in [6, 10, 14]:
        for leaf in [5, 20]:
            c.append(("Decision Tree", f"depth={dep},leaf={leaf}",
                      DecisionTreeRegressor(max_depth=dep, min_samples_leaf=leaf, random_state=42)))
    for lr in [0.05, 0.1]:
        for it in [300, 600]:
            for lv in [15, 31]:
                c.append(("Gradient Boosting", f"lr={lr},iter={it},leaves={lv}",
                          HistGradientBoostingRegressor(learning_rate=lr, max_iter=it, max_leaf_nodes=lv,
                                                        random_state=42)))
    return c


def mape(y, p):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y)) / np.asarray(y)) * 100)


usable = D.dropna(subset=FEATS).copy()
train_ok = usable["target"].notna()
months = pd.date_range(FIRST_TEST_MONTH + "-01", pd.Timestamp(C.END_DATE).replace(day=1), freq="MS")
preds, choices = [], []
for m in months:
    ts = m.tz_localize(C.PT).tz_convert("UTC")
    te_end = (m + pd.offsets.MonthBegin(1)).tz_localize(C.PT).tz_convert("UTC")
    tr = usable[train_ok & (usable["period"] < ts - pd.Timedelta(hours=14))]
    te = usable[(usable["period"] >= ts) & (usable["period"] < te_end)]
    cut = tr["period"].max() - pd.Timedelta(days=INNER_VAL_DAYS)
    itr, iva = tr[tr["period"] <= cut], tr[tr["period"] > cut]
    sc = []
    for fam, cfg, est in candidates():
        mdl = clone(est).fit(itr[FEATS], itr["target"])
        sc.append((fam, cfg, est, mape(iva["target"], mdl.predict(iva[FEATS]))))
    sc = pd.DataFrame(sc, columns=["family", "cfg", "est", "inner"])
    best = sc.loc[sc.groupby("family")["inner"].idxmin()].sort_values("inner")
    fam, cfg, est = best.iloc[0][["family", "cfg", "est"]]
    mdl = clone(est).fit(tr[FEATS], tr["target"])
    out = te[["period"]].copy()
    out["tool_mwh"] = mdl.predict(te[FEATS]) * te["asis_mwh"].to_numpy()
    out["model_family"] = fam
    out["fold"] = m.strftime("%Y-%m")
    preds.append(out)
    for _, r in best.iterrows():
        choices.append({"fold": out["fold"].iloc[0], "family": r["family"], "settings": r["cfg"],
                        "inner_val_MAPE_%": round(r["inner"], 3), "selected": r["family"] == fam,
                        "train_hours": len(tr), "test_hours": len(te)})
    v = te.assign(tool=out["tool_mwh"].to_numpy()).dropna(subset=["demand_mwh"])
    print(f"  {m:%Y-%m}: train {len(tr):6,d} h | {fam:17s} | MAPE current {mape(v.demand_mwh, v.asis_mwh):5.2f}%"
          f"  tool {mape(v.demand_mwh, v.tool):5.2f}%")

P = pd.concat(preds, ignore_index=True)
H = D[D["local_date"].between(pd.Timestamp(FIRST_TEST_MONTH + "-01"), pd.Timestamp(C.END_DATE))].merge(
    P, on="period", how="left")
H["tool_mwh"] = H["tool_mwh"].fillna(H["asis_mwh"])            # no prediction -> tool = current process
H["model_family"] = H["model_family"].fillna("Current process (no tool input)")
pd.DataFrame(choices).to_csv(f"{OUT}/step23_model_choices.csv", index=False)


# ----------------------------- results -----------------------------
def sched(actual, base, mult=BUFFER):
    a, s = np.asarray(actual, float), np.asarray(base, float) * mult
    ok_ = ~np.isnan(a)
    a, s, b = a[ok_], s[ok_], np.asarray(base, float)[ok_]
    e = s - a
    return {"MAPE_%": round(np.mean(np.abs(b - a) / a) * 100, 3),
            "R2": round(1 - ((b - a) ** 2).sum() / ((a - a.mean()) ** 2).sum(), 4),
            "over_MWh": round(e.clip(min=0).sum()), "short_hours": int((e < 0).sum()),
            "short_MWh": round((-e).clip(min=0).sum()), "hours": int(len(a))}


cur = sched(H["demand_mwh"], H["asis_mwh"])
ups = np.arange(0.98, 1.10, 0.0005)
u_eq = min(ups, key=lambda x: abs(sched(H["demand_mwh"], H["tool_mwh"], x)["short_hours"] - cur["short_hours"]))
rows = [{"method": "Purchased forecast only x1.05", **sched(H["demand_mwh"], H["vendor_mwh"])},
        {"method": f"Current process ({CHOSEN}) x1.05", **cur},
        {"method": "Tool x1.05", **sched(H["demand_mwh"], H["tool_mwh"])},
        {"method": f"Tool at equal safety x{u_eq:.4f} (same short hours)", **sched(H["demand_mwh"], H["tool_mwh"], u_eq)}]
summary = pd.DataFrame(rows)
t_, c_ = summary.iloc[2], summary.iloc[1]
print("\nRESULTS, Jan 2025 - Oct 15 2026 (MAPE and R2 before the buffer; MWh and hours after it)")
print(summary.to_string(index=False))
print(f"\nTool vs current process: over-scheduled MWh {(t_.over_MWh / c_.over_MWh - 1) * 100:+.1f}%, "
      f"short hours {(t_.short_hours / c_.short_hours - 1) * 100:+.1f}%, "
      f"short MWh {(t_.short_MWh / c_.short_MWh - 1) * 100:+.1f}%, MAPE {c_['MAPE_%']:.2f}% -> {t_['MAPE_%']:.2f}%")
monthly = pd.DataFrame([{"month": mo, **{f"current_{k}": v for k, v in sched(g["demand_mwh"], g["asis_mwh"]).items()},
                         **{f"tool_{k}": v for k, v in sched(g["demand_mwh"], g["tool_mwh"]).items()}}
                        for mo, g in H.assign(mo=H["local_date"].dt.strftime("%Y-%m")).groupby("mo")])
monthly["tool_more_short_hours"] = monthly["tool_short_hours"] > monthly["current_short_hours"]
print(f"Months where the tool had more short hours: {int(monthly['tool_more_short_hours'].sum())} of {len(monthly)}")
print(monthly[["month", "current_MAPE_%", "tool_MAPE_%", "current_over_MWh", "tool_over_MWh",
               "current_short_hours", "tool_short_hours"]].to_string(index=False))
summary.to_csv(f"{OUT}/step23_summary.csv", index=False)
monthly.to_csv(f"{OUT}/step23_monthly.csv", index=False)
H[["period", "local_date", "hour", "demand_mwh", "vendor_mwh", "asis_mwh", "tool_mwh", "model_family",
   "temp_c", "rh", "ghi"]].to_csv(f"{OUT}/step23_hourly.csv.gz", index=False, compression="gzip")

# ----------------------------- Tableau files (same schema) -----------------------------
print("\nTableau files")
H = H.reset_index(drop=True)
local = H["period"].dt.tz_convert(C.PT)
a = H["demand_mwh"]
U = pd.DataFrame({"base": H["asis_mwh"]}); T = pd.DataFrame({"base": H["tool_mwh"]})
for X in (U, T):
    X["schedule"] = X["base"] * BUFFER
    X["over"] = (X["schedule"] - a).clip(lower=0)
    X["short"] = (a - X["schedule"]).clip(lower=0)
    X["short_flag"] = (X["schedule"] < a).astype(float).where(a.notna())
    X["abs_pct"] = (X["base"] - a).abs() / a


def past_window(values, days, how="mean"):
    t = pd.DataFrame({"d": H["local_date"], "hr": H["hour"], "v": values}) \
        .pivot_table(index="d", columns="hr", values="v", aggfunc="mean").asfreq("D")
    r = t.rolling(days, min_periods=MIN_DAYS).mean().shift(2)
    r = r.stack().rename("v").reset_index()
    return pd.DataFrame({"d": H["local_date"], "hr": H["hour"]}).merge(r, on=["d", "hr"], how="left")["v"].to_numpy()


hol = USFederalHolidayCalendar().holidays(start="2024-01-01", end="2027-12-31", return_name=True)
hol.index = pd.DatetimeIndex(hol.index).normalize()
p = pd.DataFrame()
p["target_date"] = H["local_date"].dt.strftime("%Y-%m-%d")
p["decision_date"] = (H["local_date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
p["datetime_pt"] = local.dt.strftime("%Y-%m-%d %H:%M")
p["hour"] = H["hour"]
p["hour_label"] = H["hour"].map(lambda x: f"{(x % 12) or 12} {'a.m.' if x < 12 else 'p.m.'}")
p["day_of_week"] = H["local_date"].dt.day_name()
p["is_holiday"] = H["is_holiday"].astype(int)
p["holiday_name"] = H["local_date"].map(hol).fillna("")
p["published_mwh"] = H["vendor_mwh"]             # the purchased forecast (simulated vendor)
p["ciso_df"], p["ldwp_df"], p["banc_df"] = H["ciso_df"], H["ldwp_df"], H["banc_df"]
p["asis_factor"] = H["asis_factor"]
p["usual_corr_mwh"] = H["asis_mwh"] - H["vendor_mwh"]
p["usual_mwh"] = H["asis_mwh"]
p["tool_adj_mwh"] = H["tool_mwh"] - H["asis_mwh"]
p["tool_mwh"] = H["tool_mwh"]
p["buffer_mwh"] = (BUFFER - 1) * H["tool_mwh"]
p["usual_schedule_mwh"] = U["schedule"]
p["tool_schedule_mwh"] = T["schedule"]
p["change_mwh"] = T["schedule"] - U["schedule"]
p["change_pct"] = (H["tool_mwh"] / H["asis_mwh"] - 1) * 100
p["direction"] = np.where(p["change_mwh"] >= 0, "Buy more", "Buy less")
p["temp_c"] = H["temp_c"]
p["temp_4wk_avg_c"] = past_window(H["temp_c"], TEMP_DAYS)
p["temp_vs_4wk_c"] = p["temp_c"] - p["temp_4wk_avg_c"]
p["track_mape_tool_pct"] = past_window(T["abs_pct"] * 100, TRACK_DAYS)
p["track_mape_usual_pct"] = past_window(U["abs_pct"] * 100, TRACK_DAYS)
p["track_short_tool_pct"] = past_window(T["short_flag"] * 100, TRACK_DAYS)
p["track_short_usual_pct"] = past_window(U["short_flag"] * 100, TRACK_DAYS)
p["hours_in_day"] = p["target_date"].map(p["target_date"].value_counts()).astype(int)
vend_missing = H["vendor_missing"]
for b in ["ciso", "ldwp", "banc"]:              # purchased forecast is one total; count missing hours once
    p[f"sub_hours_{b}"] = vend_missing.groupby(p["target_date"]).transform("sum").astype(int) if b == "ciso" else 0
p["status_text"] = np.where(p["sub_hours_ciso"] > 0, "Purchased forecast missing for " + p["sub_hours_ciso"].astype(str) + " h (last week's demand used)",
                            np.where(p["hours_in_day"].isin([23, 24, 25]), "All sources loaded",
                                     p["hours_in_day"].astype(str) + " hourly rows (expected 23-25)"))
p["status_level"] = np.where(p["status_text"] == "All sources loaded", "OK", "Check")
p["model_family"] = H["model_family"]
p["forecast_made_pt"] = p["decision_date"] + " 10:00"

# track record: last 30 complete days at this hour (D-31 .. D-2), cost = over + short MWh
cost = pd.DataFrame({"d": H["local_date"], "hr": H["hour"], "ct": T["over"] + T["short"], "cu": U["over"] + U["short"]})
cost = cost[a.notna().to_numpy()]
cost["less"] = (cost["ct"] < cost["cu"]).astype(float)
days = pd.date_range(H["local_date"].min(), H["local_date"].max(), freq="D")


def roll30(col, how):
    t = cost.pivot_table(index="d", columns="hr", values=col, aggfunc="sum").reindex(days)
    r = t.rolling(TRACK_DAYS, min_periods=1)
    o = (r.sum() if how == "sum" else r.count()).shift(2)
    o.index.name = "d"
    return o.reset_index().melt(id_vars="d", var_name="hr", value_name=col).set_index(["d", "hr"])[col]


tr_ = pd.concat([roll30("less", "sum"), roll30("less", "count").rename("n"),
                 (roll30("ct", "sum") - roll30("cu", "sum")).rename("diff")], axis=1).reset_index()
tr_ = pd.DataFrame({"d": H["local_date"], "hr": H["hour"]}).merge(tr_, on=["d", "hr"], how="left")
enough = tr_["n"] >= 20
p["track_days_tool_less"] = tr_["less"].where(enough).to_numpy()
p["track_days_with_data"] = tr_["n"].to_numpy()
p["track_cost_diff_mwh"] = tr_["diff"].where(enough).round(0).to_numpy()
p["track_line"] = [("Not enough history yet" if pd.isna(l) else
                    f"Last 30 days at this hour: tool cost less on {int(l)} of {int(n)} days, and "
                    f"{abs(int(dd)):,} MWh {'less' if dd <= 0 else 'more'} in total")
                   for l, n, dd in zip(p["track_days_tool_less"], p["track_days_with_data"], p["track_cost_diff_mwh"])]

c = pd.DataFrame({"datetime_pt": p["datetime_pt"], "local_date": p["target_date"],
                  "month": H["local_date"].dt.strftime("%Y-%m"), "hour": H["hour"], "actual_mwh": a,
                  "usual_base_mwh": U["base"], "tool_base_mwh": T["base"],
                  "usual_schedule_mwh": U["schedule"], "tool_schedule_mwh": T["schedule"],
                  "usual_over_mwh": U["over"], "tool_over_mwh": T["over"],
                  "usual_short_mwh": U["short"], "tool_short_mwh": T["short"],
                  "usual_short_flag": U["short_flag"], "tool_short_flag": T["short_flag"],
                  "usual_abs_pct": U["abs_pct"], "tool_abs_pct": T["abs_pct"]})
c = c[c["actual_mwh"].notna()].reset_index(drop=True)          # hours with an EIA error are left out
c["usual_short_flag"] = c["usual_short_flag"].astype(int)
c["tool_short_flag"] = c["tool_short_flag"].astype(int)


def worse(g):
    items = []
    if g.tool_over_mwh.sum() > g.usual_over_mwh.sum(): items.append("over-scheduled MWh")
    if g.tool_short_flag.sum() > g.usual_short_flag.sum(): items.append("short hours")
    if g.tool_short_mwh.sum() > g.usual_short_mwh.sum(): items.append("short MWh")
    if g.tool_abs_pct.mean() > g.usual_abs_pct.mean(): items.append("MAPE")
    return ", ".join(items) if items else "none"


parts = [np.where(c.tool_over_mwh > c.usual_over_mwh, "over-scheduled MWh", ""),
         np.where(c.tool_short_flag > c.usual_short_flag, "short hours", ""),
         np.where(c.tool_short_mwh > c.usual_short_mwh, "short MWh", ""),
         np.where(c.tool_abs_pct > c.usual_abs_pct, "MAPE", "")]
c["worse_hour"] = [", ".join(x for x in row if x) or "none" for row in zip(*parts)]
c["worse_day"] = c["local_date"].map({k: worse(g) for k, g in c.groupby("local_date")})
c["worse_month"] = c["month"].map({k: worse(g) for k, g in c.groupby("month")})

# ---- schema check against the current Tableau files, then save ----
def check(new, ref_path):
    ref = pd.read_csv(ref_path, nrows=200)
    probs = []
    if list(new.columns) != list(ref.columns):
        probs.append(f"columns differ: missing {sorted(set(ref.columns) - set(new.columns))}, "
                     f"extra {sorted(set(new.columns) - set(ref.columns))}")
    for col in ref.columns.intersection(new.columns):
        rn, nn = pd.api.types.is_numeric_dtype(ref[col]), pd.api.types.is_numeric_dtype(new[col])
        ri, ni = pd.api.types.is_integer_dtype(ref[col]), pd.api.types.is_integer_dtype(new[col])
        if rn != nn or (ri and not ni):
            probs.append(f"type of {col}: was {ref[col].dtype}, now {new[col].dtype}")
    return probs


ok_all = True
for name, df_ in [("calgrid_plan_tableau.csv", p), ("calgrid_compare_hourly.csv", c)]:
    ref = os.path.join(TABLEAU_REF, name)
    probs = check(df_, ref) if os.path.exists(ref) else ["reference file not found in 09_data/"]
    if probs:
        ok_all = False
        print(f"  {name}: NOT SAVED, schema problems: " + "; ".join(probs))
    else:
        df_.round(4).to_csv(os.path.join(TABLEAU_OUT, name), index=False)
        print(f"  {name}: saved {len(df_):,} rows, same columns and types as 09_data/{name}")
print(f"  dates {p['target_date'].min()} -> {p['target_date'].max()}; "
      f"days without 23-25 rows: {int((~p.groupby('target_date')['hours_in_day'].first().isin([23, 24, 25])).sum())}")
print(f"\nDone in {time.time() - t0:.0f}s. Push sim/results/ (Tableau files are in {os.path.relpath(TABLEAU_OUT, os.path.join(C.HERE, '..'))}/).")
