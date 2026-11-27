"""Shared settings, data downloads and helpers for the CalGrid simulation.

Data sources (all free):
  - EIA-930 hourly demand ("D") and published day-ahead forecasts ("DF") for CISO, LDWP, BANC.
    Needs an EIA API key in the environment variable EIA_API_KEY (or a line EIA_API_KEY=... in .env).
  - Open-Meteo Historical Weather API (ERA5 reanalysis = actual weather), used to TRAIN the vendor model.
  - Open-Meteo Previous Runs API (forecasts made 24 h / 48 h earlier), used to PREDICT, so the
    vendor only sees weather forecasts that existed before the 10 a.m. bid.
Every download is cached in sim/data_cache/ (kept out of the backup copy), so a second run is fast.
"""
import os, json, time
import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "data_cache")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================ SETTINGS ============================
TRAIN_START = "2021-01-01"   # first day of vendor training data (actual weather + demand)
SIM_START   = "2024-01-01"   # first simulated (tested) month; Previous Runs forecasts start Jan 2024
END_DATE    = "2026-10-15"   # same end date as the Tableau files
DECISION_HOUR_LOCAL = 10     # bid closes 10 a.m. Pacific on D-1
PUBLISH_DELAY_H     = 6      # hours from weather model run to public availability
PT = "America/Los_Angeles"
BAS = ["CISO", "LDWP", "BANC"]

EIA_URL     = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREV_URL    = "https://previous-runs-api.open-meteo.com/v1/forecast"

COUNTIES = [   # (name, lat, lon, GWh 2024)
    ("Los Angeles",    34.0522, -118.2437, 64896.32975),
    ("Orange",         33.7175, -117.8311, 19225.14514),
    ("San Diego",      32.7157, -117.1611, 19110.64376),
    ("Santa Clara",    37.3541, -121.9552, 17793.58273),
    ("Riverside",      33.9533, -117.3962, 17418.39044),
    ("San Bernardino", 34.1083, -117.2898, 16206.40128),
    ("Kern",           35.3733, -119.0187, 15272.88211),
    ("Alameda",        37.6017, -121.7195, 11699.69572),
    ("Sacramento",     38.5816, -121.4944, 11380.74488),
    ("Contra Costa",   37.8534, -121.9018,  8468.064955),
    ("Fresno",         36.7378, -119.7871,  7968.273118),
    ("San Joaquin",    37.9577, -121.2908,  6029.122984),
    ("Ventura",        34.2746, -119.2290,  5211.853053),
    ("Stanislaus",     37.5091, -120.9876,  5200.26382),
    ("San Francisco",  37.7749, -122.4194,  5126.011382),
    ("Tulare",         36.2077, -119.3473,  4615.260742),
    ("San Mateo",      37.5630, -122.3255,  4261.306056),
    ("Merced",         37.3022, -120.4830,  3698.524101),
    ("Solano",         38.2494, -122.0400,  3215.62864),
    ("Placer",         38.9451, -121.0747,  3170.687817),
    ("Sonoma",         38.2919, -122.4580,  2901.702823),
    ("Santa Barbara",  34.4208, -119.6982,  2613.012066),
]
# ==================================================================


# ---------------------------- helpers ----------------------------
def load_key():
    key = os.environ.get("EIA_API_KEY")
    if not key:
        for p in [os.path.join(HERE, ".env"), os.path.join(HERE, "..", ".env")]:
            if os.path.exists(p):
                for line in open(p):
                    if line.strip().startswith("EIA_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("No EIA key. Run:  export EIA_API_KEY=your_key   (see sim/README.md)")
    return key


def cached_json(name, producer):
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"  [download] {name}")
    data = producer()
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def get_with_retry(url, params, tries=8, timeout=150):
    for attempt in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as ex:
            wait = 5 * (attempt + 1)
            print(f"    network error ({ex.__class__.__name__}), retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            wait = 60 * (attempt + 1)      # Open-Meteo free tier: per-minute / per-hour limits
            print(f"    HTTP {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:300]}")
        return r.json()
    raise RuntimeError(f"Gave up on {url} after {tries} tries")


def year_chunks(start, end):
    """Split [start, end] into calendar-year pieces (keeps each download small)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    while s <= e:
        ce = min(pd.Timestamp(year=s.year, month=12, day=31), e)
        out.append((s.date(), ce.date()))
        s = ce + pd.Timedelta(days=1)
    return out


# ---------------------------- EIA ----------------------------
def fetch_eia(dtype, start, end, key):
    """Hourly EIA-930 values (dtype 'D' = demand, 'DF' = day-ahead forecast), wide by BA."""
    frames = []
    for s, e in year_chunks(start, end):
        base = {"api_key": key, "frequency": "hourly", "data[0]": "value",
                "facets[respondent][]": BAS, "facets[type][]": dtype,
                "start": f"{s}T00", "end": f"{e}T23",
                "sort[0][column]": "period", "sort[0][direction]": "asc"}

        def producer():
            rows, page = [], 0
            while True:
                chunk = get_with_retry(EIA_URL, {**base, "offset": page * 5000, "length": 5000})["response"]["data"]
                rows.extend(chunk)
                if len(chunk) < 5000:
                    return rows
                page += 1
        frames.append(pd.DataFrame(cached_json(f"eia_{dtype}_{s}_{e}.json", producer)))
    raw = pd.concat(frames, ignore_index=True)
    if raw.empty:
        raise RuntimeError(f"EIA returned no {dtype} rows: check the key")
    raw["period"] = pd.to_datetime(raw["period"], utc=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw = raw.dropna(subset=["value"]).drop_duplicates(["period", "respondent"])
    return raw.pivot(index="period", columns="respondent", values="value")


def clean_series(x, forecast=None, tol=0.4):
    """Flag EIA data errors in one BA's hourly demand.

    An hour is an error only if it is more than tol (40%) away from BOTH the median of the same hour
    over the 7 days before and after AND that BA's own published forecast for the hour. Real heat
    waves stay in (the BA forecast them too); glitches such as LDWP = 24 MWh are removed.
    Where no published forecast exists, the median test alone is used with a wider limit (60%).
    """
    shifts = [x.shift(24 * k) for k in range(-7, 8) if k != 0]
    med = pd.concat(shifts, axis=1).median(axis=1)
    off_med = (x / med - 1).abs()
    if forecast is None:
        forecast = pd.Series(np.nan, index=x.index)
    off_fc = (x / forecast.where(forecast > 0) - 1).abs()
    bad = np.where(off_fc.notna(), (off_med > tol) & (off_fc > tol), off_med > 0.6)
    bad = pd.Series(bad, index=x.index) & x.notna()
    return x.where(~bad), bad


def load_demand(key):
    """Hourly table: actual demand (3-BA sum) and the published forecast (3-BA DF sum), UTC grid.
    EIA-930 has data errors (spikes, drops); hours flagged by clean_series() are set to missing.
    demand_for_lags fills those hours with the published forecast, so lag inputs stay complete."""
    d = fetch_eia("D", pd.Timestamp(TRAIN_START) - pd.Timedelta(days=20), END_DATE, key)
    grid = pd.date_range(d.index.min(), d.index.max(), freq="h", tz="UTC")
    d = d[BAS].where(d[BAS] > 0).reindex(grid)
    df = fetch_eia("DF", pd.Timestamp(TRAIN_START) - pd.Timedelta(days=20), END_DATE, key)
    df = df[BAS].where(df[BAS] > 0).reindex(grid)
    print("   data cleaning (>40% away from both the 2-week median and the BA's own forecast):")
    out = pd.DataFrame(index=grid)
    for b in BAS:
        clean, bad = clean_series(d[b], df[b])
        print(f"     {b}: {int(bad.sum())} hours removed of {int(d[b].notna().sum())}")
        out[f"{b.lower()}_mwh"] = clean
    out["demand_mwh"] = out[[f"{b.lower()}_mwh" for b in BAS]].sum(axis=1, min_count=3)
    # fill rule: a missing BA forecast = that BA's demand one week earlier
    for b in BAS:
        out[f"{b.lower()}_df"] = df[b].where(df[b] > 0, out[f"{b.lower()}_mwh"].shift(168))
    out["published_mwh"] = out[[f"{b.lower()}_df" for b in BAS]].sum(axis=1, min_count=3)
    out["demand_for_lags"] = out["demand_mwh"].fillna(out["published_mwh"]).interpolate(limit=6)
    out = out.rename_axis("period").reset_index()
    return add_calendar(out)


def add_calendar(t):
    local = t["period"].dt.tz_convert(PT)
    t["local_date"] = local.dt.tz_localize(None).dt.normalize()
    t["hour"] = local.dt.hour
    t["dayofweek"] = local.dt.dayofweek
    t["month"] = local.dt.month
    hol = USFederalHolidayCalendar().holidays(start="2018-01-01", end="2027-12-31")
    t["is_holiday"] = t["local_date"].isin(set(pd.DatetimeIndex(hol).normalize()))
    return t


# ---------------------------- weather ----------------------------
def _county_series(url, hourly_vars, name, lat, lon, s, e, tag):
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars),
              "start_date": str(s), "end_date": str(e), "timezone": "UTC"}
    key = name.lower().replace(" ", "_")
    path = os.path.join(CACHE_DIR, f"{tag}_{key}_{s}_{e}.json")
    fresh = not os.path.exists(path)
    js = cached_json(os.path.basename(path), lambda: get_with_retry(url, params))
    if fresh:
        time.sleep(8)                     # stay under Open-Meteo's free per-minute limit
    h = js["hourly"]
    df = pd.DataFrame({"period": pd.to_datetime(h["time"], utc=True)})
    for v in hourly_vars:
        df[v] = pd.to_numeric(pd.Series(h.get(v, [np.nan] * len(df))), errors="coerce").to_numpy()
    return df


def _weighted(frames, cols):
    long = pd.concat(frames, ignore_index=True)
    out = {}
    for c in cols:
        ok = long[c].notna()
        num = (long[c] * long["w"]).where(ok, 0).groupby(long["period"]).sum()
        den = long["w"].where(ok, 0).groupby(long["period"]).sum()
        out[c] = num / den.replace(0, np.nan)
    return pd.DataFrame(out).rename_axis("period").reset_index()


WX = {"temp_c": "temperature_2m", "rh": "relative_humidity_2m", "ghi": "shortwave_radiation"}
# temperature + humidity were downloaded first; solar radiation (ghi) is a second, separate download,
# so the first cache stays valid.
_GROUPS = [("", ["temp_c", "rh"]), ("sol", ["ghi"])]


def _load(url, tag, suffixes, start):
    frames = []
    for name, lat, lon, w in COUNTIES:
        parts = []
        for gtag, shorts in _GROUPS:
            vars_ = [WX[c] + sfx for sfx in suffixes for c in shorts]
            pieces = [_county_series(url, vars_, name, lat, lon, s, e, tag + gtag)
                      for s, e in year_chunks(start, END_DATE)]
            parts.append(pd.concat(pieces, ignore_index=True).drop_duplicates("period").set_index("period"))
        df = pd.concat(parts, axis=1).reset_index()
        df["w"] = w
        frames.append(df)
    cols = [WX[c] + sfx for sfx in suffixes for c in WX]
    return _weighted(frames, cols)


def load_actual_weather():
    """County-weighted actual weather (ERA5): temp_c (deg C), rh (%), ghi (solar radiation, W/m2)."""
    w = _load(ARCHIVE_URL, "era5", [""], pd.Timestamp(TRAIN_START) - pd.Timedelta(days=3))
    return w.rename(columns={v: k for k, v in WX.items()})


def load_forecast_weather():
    """County-weighted weather FORECASTS, 24 h old (_d1) and 48 h old (_d2), and use_d1:
    True where the 24-h-old forecast was public by 10 a.m. on D-1."""
    sfx = ["_previous_day1", "_previous_day2"]
    w = _load(PREV_URL, "prevruns", sfx, pd.Timestamp(SIM_START) - pd.Timedelta(days=3))
    out = pd.DataFrame({"period": w["period"]})
    for k, v in WX.items():
        out[f"{k}_d1"], out[f"{k}_d2"] = w[v + sfx[0]], w[v + sfx[1]]
    valid = out["period"]
    local = valid.dt.tz_convert(PT).dt.tz_localize(None)
    decision = (local.dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=DECISION_HOUR_LOCAL)) \
        .dt.tz_localize(PT, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
    out["use_d1"] = (valid - pd.Timedelta(hours=24) + pd.Timedelta(hours=PUBLISH_DELAY_H)) <= decision
    return out


def forecast_for_bid(fc, act, mos=True, window_days=30, lag_days=7):
    """The weather forecast the vendor uses for each hour (24-h-old where allowed, else 48-h-old).

    mos=True removes the forecast's recent average error (bias) for the same hour of day and the
    same forecast age, measured over the 30 days ending 7 days before the target day (actual weather
    arrives with a delay). This is standard weather-forecast post-processing (MOS / bias correction).
    """
    t = fc.merge(act, on="period", how="left")
    t["local_date"] = t["period"].dt.tz_convert(PT).dt.tz_localize(None).dt.normalize()
    t["hour"] = t["period"].dt.tz_convert(PT).dt.hour
    out = pd.DataFrame({"period": t["period"], "use_d1": t["use_d1"]})
    for k in WX:
        for d in ("d1", "d2"):
            col = t[f"{k}_{d}"]
            if mos:
                err = (t[f"{k}_{d}"] - t[k]).groupby([t["local_date"], t["hour"]]).mean().unstack()
                err = err.asfreq("D")
                bias = err.rolling(window_days, min_periods=10).mean().shift(lag_days)
                b = pd.Series(bias.stack().values,
                              index=pd.MultiIndex.from_tuples(bias.stack().index)).reindex(
                    list(zip(t["local_date"], t["hour"]))).to_numpy()
                col = col - np.nan_to_num(b)
            t[f"_{k}_{d}"] = col
        out[k] = np.where(t["use_d1"], t[f"_{k}_d1"], t[f"_{k}_d2"])
    if "ghi" in out:
        out["ghi"] = out["ghi"].clip(lower=0)
    return out


# ---------------------------- metrics ----------------------------
def metrics(actual, pred):
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = ~(np.isnan(a) | np.isnan(p))
    a, p = a[ok], p[ok]
    err = p - a
    ss = ((a - a.mean()) ** 2).sum()
    return {"MAPE_%": round(float(np.mean(np.abs(err) / a) * 100), 3),
            "bias_%": round(float(err.sum() / a.sum() * 100), 3),
            "R2": round(float(1 - (err ** 2).sum() / ss), 4) if ss > 0 else np.nan,
            "hours": int(len(a))}


class Tee:
    """Print to the screen and to a log file in results/ at the same time."""
    def __init__(self, path):
        self.f = open(path, "w")
        import sys
        self.out = sys.stdout
    def write(self, s):
        self.out.write(s); self.f.write(s); self.f.flush()
    def flush(self):
        self.out.flush(); self.f.flush()
