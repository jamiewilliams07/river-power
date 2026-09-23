"""
End-to-end test of the SWOT pipeline on synthetic data with known answers.

    python3 tests/test_swot_pipeline.py

The test generates a synthetic gauge record and synthetic SWOT reach
observations from a prescribed rating curve, runs scripts 10-15 and
report/make_summary.py on them, and compares each output with the value
implied by the inputs. The Hydrocron and ECCC APIs are replaced by mocked
responses, so no network access is required, and all files are written to a
temporary directory. A failure here identifies a regression before a full
run on real data. Run this test after modifying any pipeline script.
"""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest import mock

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))


# Script file names begin with digits, so they are loaded by path rather than imported.
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------- synthetic river, known answers
rng = np.random.default_rng(0)
A, B, H0 = 60.0, 1.6, 236.0          # rating curve: Q = A (h - H0)^B
REACHES = ["74400000011", "74400000021", "74400000031", "74400000043"]  # final digit 3: lake reach
GAUGE_REACH = REACHES[1]
UP = [REACHES[0]]
SLOPE = {REACHES[0]: 6.0e-4, REACHES[1]: 1.0e-4, REACHES[2]: 1.5e-4, REACHES[3]: 2e-5}
LENGTH = 10000.0

# Synthetic daily discharge peaking at day of year 170 (about 19 June); the
# tributary contributes 25% of the main-stem discharge.
days = pd.date_range("1990-01-01", "2026-06-30", freq="D")
doy = days.dayofyear.values
q_main = 500 + 1800 * np.exp(-((doy - 170) / 35.0) ** 2) + rng.normal(0, 40, len(days))
q_main = np.clip(q_main, 150, None)
q_trib = 0.25 * q_main
h_true = H0 + (q_main / A) ** (1 / B)
gauge = pd.DataFrame({"date": days.date, "discharge_m3s": q_main,
                      "level_m": h_true - 230.0 + rng.normal(0, 0.03, len(days)),
                      "discharge_flag": None, "level_flag": None, "source": "synthetic"})
trib = gauge.assign(discharge_m3s=q_trib, level_m=np.nan)


# One synthetic overpass every 7 days at 18:00 UTC.
PASS_GRID = pd.date_range("2023-07-01", "2026-09-30", freq="7D") + pd.Timedelta(hours=18)
RENUMBERED = {REACHES[1]: "74400000051"}      # gauge reach is renumbered in Version D, as in the real data
NEW_TO_OLD = {v: k for k, v in RENUMBERED.items()}


def fake_swot_rows(reach, version, start, end):
    t = PASS_GRID[(PASS_GRID >= pd.Timestamp(start)) &
                  (PASS_GRID <= pd.Timestamp(end) + pd.Timedelta(days=1))]
    lookup = pd.Series(h_true, index=pd.to_datetime(days))
    rows = []
    for ts in t:
        # convert UTC to Mountain Standard Time (UTC-7) to match the gauge day
        local = (ts - pd.Timedelta(hours=7)).normalize()
        if local not in lookup.index:
            continue
        # water-surface elevation offset of each reach relative to the gauge reach
        off = {REACHES[0]: 3.0, REACHES[1]: 0.0, REACHES[2]: -1.5, REACHES[3]: -2.5}[reach]
        rows.append({
            "reach_id": reach, "time_str": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle_id": 1, "pass_id": 1, "river_name": "Athabasca River",
            "wse": lookup[local] + off + rng.normal(0, 0.08),
            "wse_u": 0.1, "width": 300 * (np.interp(lookup[local], [H0, H0 + 10], [0.8, 1.2])) + rng.normal(0, 12),
            "width_u": 10, "slope": SLOPE[reach] * (1 + rng.normal(0, 0.05)), "slope_u": 1e-6,
            "reach_q": 0, "dark_frac": 0.1, "ice_clim_f": 0, "ice_dyn_f": 0,
            "partial_f": 0, "n_good_nod": 40, "obs_frac_n": 1.0, "xovr_cal_q": 0,
            "p_width": 300, "p_length": LENGTH,
            "p_dist_out": {REACHES[0]: 1.2e6, REACHES[1]: 1.19e6,
                           REACHES[2]: 1.18e6, REACHES[3]: 1.17e6}[reach],
            "p_lat": {REACHES[0]: 56.95, REACHES[1]: 56.85,
                      REACHES[2]: 56.75, REACHES[3]: 56.65}[reach], "p_lon": -111.4,
        })
    df = pd.DataFrame(rows)
    # Inject one invalid observation of each type; 12_clean_swot.py must remove each one.
    if len(df) > 10:
        df.loc[0, "wse"] = -999999999999.0          # fill value
        df.loc[1, "reach_q"] = 3                     # reach_q = 3 (bad)
        df.loc[2, "ice_dyn_f"] = 2                   # dynamic ice flag set (ice observed)
        df.loc[3, "partial_f"] = 1                   # reach partially observed
        df.loc[4, "wse"] = df["wse"].median() + 25   # +25 m offset, must be flagged as an outlier
    return df


# Minimal stand-in for a requests.Response object.
class FakeResp:
    def __init__(self, code, payload=None, text=""):
        self.status_code, self._p, self.text = code, payload, text

    def json(self):
        return self._p


# Mock Hydrocron API: returns synthetic reach time series without network access.
def fake_hydrocron(url, params, timeout):
    reach, coll = params["feature_id"], params["collection_name"]
    start, end = params["start_time"][:10], params["end_time"][:10]
    if coll.endswith("2.0"):
        s, e = max(start, "2023-07-01"), min(end, "2025-05-03")
        if reach == REACHES[2]:
            return FakeResp(400, text='{"error": "Results with the specified Feature ID were not found"}')
    else:
        s, e = max(start, "2023-07-01"), end          # Version D also contains reprocessed earlier passes
        if reach in RENUMBERED:                        # pre-renumbering ID is not valid in Version D
            return FakeResp(400, text='{"error": "not found"}')
        if reach not in REACHES and reach not in NEW_TO_OLD:
            return FakeResp(400, text='{"error": "not found"}')
    if s > e:
        return FakeResp(400, text='{"error": "not found"}')
    if reach == REACHES[0] and coll.endswith("_D") and start[:4] != end[:4]:
        return FakeResp(413, text="Payload Too Large")   # forces the script to split the request by year
    old = NEW_TO_OLD.get(reach, reach)
    df = fake_swot_rows(old, coll, s, e)
    if reach in NEW_TO_OLD:
        df["reach_id"] = reach
        df["p_dist_out"] = df["p_dist_out"] + 500.0     # renumbered reach: distance from outlet shifted 500 m
    if coll.endswith("_D") and old == REACHES[0] and len(df):
        df["wse"] = np.where(df["wse"] > -1e10, df["wse"] + 0.35, df["wse"])  # 0.35 m C/D offset, above the 0.25 m version-check limit
    df["wse_units"] = "m"
    return FakeResp(200, {"status": "200 OK", "results": {"csv": df.to_csv(index=False), "geojson": {}}})


# Mock ECCC (Water Survey of Canada) API: station metadata and daily values.
def fake_eccc(url, params, timeout):
    stn = params["STATION_NUMBER"]
    if url.endswith("hydrometric-stations/items"):
        return FakeResp(200, {"features": [{"properties": {
            "STATION_NAME": "ATHABASCA RIVER BELOW FORT MCMURRAY" if stn == "07DA001"
            else "CLEARWATER RIVER AT DRAPER", "DRAINAGE_AREA_GROSS": 133000,
            "STATUS_EN": "Active"}, "geometry": {"coordinates": [-111.4, 56.78]}}]})
    src = gauge if stn == "07DA001" else trib
    a, b = params["datetime"].split("/")
    sub = src[(pd.to_datetime(src["date"]) >= a) & (pd.to_datetime(src["date"]) <= b)]
    sub = sub[pd.to_datetime(sub["date"]) <= "2025-12-31"]   # approved record ends 2025-12-31; later data are provisional
    feats = [{"properties": {"DATE": str(r.date), "DISCHARGE": r.discharge_m3s,
                             "LEVEL": None if pd.isna(r.level_m) else r.level_m,
                             "DISCHARGE_SYMBOL_EN": None, "LEVEL_SYMBOL_EN": None}}
             for r in sub.itertuples()]
    return FakeResp(200, {"features": feats})


# ------------------------------------------------------- run the pipeline
# All scripts run in a temporary directory so the project data/ and outputs/ are not modified.
work = tempfile.mkdtemp()
os.chdir(work)
os.makedirs("data")
# Start from the project config and replace the site-specific settings with the synthetic reaches.
import yaml
test_cfg = yaml.safe_load(open(os.path.join(ROOT, "swot_config.yaml")))
test_cfg.update({"reaches": REACHES, "gauge_reach": GAUGE_REACH,
                 "upstream_of_tributary": UP, "end": "2026-06-30",
                 "gauges": {"main": "07DA001", "tributary": "07CD001"}})
test_cfg.pop("version_d_ids", None)
cfg_text = yaml.safe_dump(test_cfg, sort_keys=False)
open("swot_config.yaml", "w").write(cfg_text)

print("=" * 66)
print("SWOT pipeline test (synthetic data, known answers)")
print("=" * 66)

# Post-2025 gauge data in the Water Office real-time CSV format, passed to 10_fetch_gauge.py with --extra.
rt = gauge[pd.to_datetime(gauge["date"]) > "2025-12-31"]
lines = ["Station, 07DA001\n", "Some header line\n",
         "ID,Date,Parameter/Paramètre,Value/Valeur,Qualifier/Qualificatif\n"]
for r in rt.itertuples():
    for hr in (0, 12):
        ts = pd.Timestamp(r.date) + pd.Timedelta(hours=hr + 7)
        lines.append(f"07DA001,{ts.strftime('%Y-%m-%dT%H:%M:%S')}-07:00,47,{r.discharge_m3s:.2f},\n")
        lines.append(f"07DA001,{ts.strftime('%Y-%m-%dT%H:%M:%S')}-07:00,46,{r.level_m:.3f},\n")
open("data/07DA001_realtime.csv", "w", encoding="utf-8").writelines(lines)

print("\n-- 10 fetch gauge data --")
g10 = load("10_fetch_gauge")
with mock.patch.object(g10.requests, "get", side_effect=fake_eccc), \
     mock.patch.object(sys, "argv", ["x", "--extra", "07DA001=data/07DA001_realtime.csv"]), \
     mock.patch("sys.stdout", new=io.StringIO()):
    g10.main()
gd = pd.read_csv("data/gauge_07DA001.csv")
check("gauge: approved and provisional records merged",
      gd["date"].max() == "2026-06-30" and (gd["source"].str.contains("provisional")).any(),
      f"{len(gd)} days, last date {gd['date'].max()}")
check("gauge: no duplicate dates", gd["date"].is_unique)

print("\n-- 11 fetch SWOT --")
s11 = load("11_fetch_swot")


def fetch_swot():
    with mock.patch.object(s11.requests, "get", side_effect=fake_hydrocron), \
         mock.patch.object(s11.time, "sleep"), \
         mock.patch.object(sys, "argv", ["x"]), \
         mock.patch("sys.stdout", new=io.StringIO()):
        s11.main()


fetch_swot()
log = pd.read_csv("data/swot_fetch_log.csv", dtype={"reach_id": str})
check("swot: renumbered reach has no Version D data before ID mapping",
      log[(log.reach_id == GAUGE_REACH) & (log.version == "D")]["rows"].iloc[0] == 0)

print("\n-- 11b find Version D IDs --")
buf = io.StringIO()
p11b = load("11b_find_version_d_ids")
with mock.patch("requests.get", side_effect=fake_hydrocron), \
     mock.patch("time.sleep"), \
     mock.patch.object(sys, "argv", ["x", "--max-num", "10", "--basins-either-side", "1"]), \
     mock.patch("sys.stdout", new=buf):
    p11b.main()
out = buf.getvalue()
# Extract the version_d_ids block printed by 11b and append it to the config, as in the manual workflow.
block = out[out.index("version_d_ids:"):out.index("Then rerun")].strip() if "version_d_ids:" in out else ""
check("11b: renumbered ID found for gauge reach",
      f'"{GAUGE_REACH}": "{RENUMBERED[GAUGE_REACH]}"' in block, block.replace("\n", " | "))
open("swot_config.yaml", "a").write("\n" + block + "\n")

# Fetch again with the renumbered ID mapping now in the config.
fetch_swot()
raw = pd.read_csv("data/swot_raw.csv", dtype={"reach_id": str, "queried_id": str})
log = pd.read_csv("data/swot_fetch_log.csv", dtype={"reach_id": str})
check("swot: Version D data stored under original reach ID after mapping",
      ((raw.reach_id == GAUGE_REACH) & (raw.version == "D")).sum() > 0 and
      (raw.loc[(raw.reach_id == GAUGE_REACH) & (raw.version == "D"), "queried_id"]
       == RENUMBERED[GAUGE_REACH]).all())
check("swot: Versions C and D both retrieved", set(raw["version"]) == {"C", "D"})
check("swot: reach absent from Version C logged, fetch continues",
      log[(log.reach_id == REACHES[2]) & (log.version == "C")]["rows"].iloc[0] == 0
      and (raw.reach_id == REACHES[2]).any())
check("swot: HTTP 413 handled by splitting request by year",
      (raw[(raw.reach_id == REACHES[0]) & (raw.version == "D")].shape[0] > 0))
check("swot: reach IDs kept as 11-character strings", raw["reach_id"].str.len().eq(11).all())
check("swot: _units columns removed", not any(c.endswith("_units") for c in raw.columns))


# Run a script as a subprocess; print the tail of stdout and stderr if it exits with an error.
def run(script):
    p = subprocess.run([sys.executable, os.path.join(SCRIPTS, script)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:])
    return p


print("\n-- 12 clean SWOT --")
p = run("12_clean_swot.py")
check("clean: completes without error", p.returncode == 0)
rep = pd.read_csv("outputs/swot_cleaning_report.csv", dtype={"reach_id": str}).set_index("reach_id")
clean = pd.read_csv("data/swot_clean.csv", dtype={"reach_id": str})
check("clean: lake reach removed", REACHES[3] not in set(clean.reach_id))
check("clean: only open-water months (May-Oct) retained", pd.to_datetime(clean["time"]).dt.month.isin([5, 6, 7, 8, 9, 10]).all())
check("clean: no fill values remain", (clean["wse"] > -1e10).all())
# One invalid row of each type was injected above; each filter must remove at least one row.
for col in ("reach_q (NASA quality)", "ice (observed)", "partly observed", "height outlier"):
    check(f"clean: '{col}' filter removed injected row",
          col in rep.columns and rep.loc[GAUGE_REACH, col] >= 1,
          f"{int(rep.loc[GAUGE_REACH, col]) if col in rep.columns else 0} removed")
check("clean: figure written", os.path.exists("figures/swot_cleaning.png"))
check("clean: Version D removed where reach ID refers to a different location",
      "v.D ID is a different place" in rep.columns and rep.loc[REACHES[0], "v.D ID is a different place"] > 0
      and set(clean.loc[clean.reach_id == REACHES[0], "version"]) == {"C"},
      f"{int(rep.loc[REACHES[0], 'v.D ID is a different place']) if 'v.D ID is a different place' in rep.columns else 0} removed")
check("clean: Version C duplicate removed when Version D has the same pass",
      "same pass in C and D" in rep.columns and rep.loc[GAUGE_REACH, "same pass in C and D"] > 5,
      f"{int(rep.loc[GAUGE_REACH, 'same pass in C and D']) if 'same pass in C and D' in rep.columns else 0} removed at gauge reach")
tt = pd.to_datetime(clean["time"]).dt.round("15min")
check("clean: no duplicate passes per reach", not clean.assign(t=tt).duplicated(["reach_id", "t"]).any())

print("\n-- 13 validate --")
p = run("13_validate.py")
check("validate: completes without error", p.returncode == 0)
summ = pd.read_csv(f"outputs/validation_summary_{GAUGE_REACH}.csv")
r1 = summ[summ.result.str.startswith("1")].iloc[0]
check("result 1: SWOT water-surface elevation tracks gauge level",
      r1.correlation > 0.95 and abs(r1.slope - 1) < 0.1,
      f"r={r1.correlation:.3f}, slope={r1.slope:.2f}, rmse={100*r1.rmse_m:.0f} cm")
r2 = summ[summ.result.str.contains("height, leave")].iloc[0]
check("result 2: rating-curve discharge error < 15%, NSE > 0.8",
      r2.median_abs_pct_err < 15 and r2.nse > 0.8,
      f"median absolute error {r2.median_abs_pct_err:.1f}%, NSE {r2.nse:.2f}")
check("rating curve: exponent within 0.5 of true value", abs(r2.b - B) < 0.5, f"b={r2.b:.2f} (true {B})")
rw = summ[summ.result.str.contains("width")]
# The width-based result row is optional, so it is checked only when present.
if len(rw):
    w_err = rw.iloc[0].get("median_abs_pct_err", np.nan)
    check("result 2b: width-based discharge error exceeds height-based error",
          (not np.isfinite(w_err)) or w_err > r2.median_abs_pct_err,
          f"width error {w_err:.1f}% vs height {r2.median_abs_pct_err:.1f}%")
ts = summ[summ.result.str.contains("time split")].iloc[0]
check("result 2: time-split validation executed", ts.n > 0, f"n={int(ts.n)}, median error {ts.median_abs_pct_err:.1f}%")

p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "13_validate.py"), "--reach", REACHES[2]],
                   capture_output=True, text=True)
check("validate: --reach on another reach keeps gauge-reach output",
      p.returncode == 0 and os.path.exists(f"outputs/validation_summary_{REACHES[2]}.csv")
      and os.path.exists(f"outputs/validation_summary_{GAUGE_REACH}.csv"))

p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "13_validate.py"), "--version", "D"],
                   capture_output=True, text=True)
check("validate: --version D uses Version D only, separate outputs",
      p.returncode == 0 and "Version D readings only" in p.stdout
      and os.path.exists(f"outputs/validation_summary_{GAUGE_REACH}_vD.csv"))

print("\n-- 13b version check --")
p = run("13b_version_check.py")
vc = pd.read_csv("outputs/version_check.csv", dtype={"reach_id": str}).set_index("reach_id")
check("13b: reach with C/D elevation offset flagged DISAGREE",
      p.returncode == 0 and vc.loc[REACHES[0], "verdict"].startswith("DISAGREE"),
      f"{vc.loc[REACHES[0], 'height_diff_D_minus_C_cm']} cm")
check("13b: gauge reach (consistent versions) classified as agree",
      vc.loc[GAUGE_REACH, "verdict"] == "agree",
      f"{vc.loc[GAUGE_REACH, 'height_diff_D_minus_C_cm']} cm")

print("\n-- 14 power --")
p = run("14_power.py")
check("power: completes without error", p.returncode == 0)
pw = pd.read_csv("outputs/reach_power.csv", dtype={"reach_id": str}).set_index("reach_id")
# Independent calculation: Q30 from the flow duration curve (Weibull plotting
# position), head H = S L, then P = rho g Q H eta in MW.
q30 = float(np.interp(30, 100 * np.arange(1, len(q_main) + 1) / (len(q_main) + 1),
                      np.sort(q_main)[::-1]))
exp = 1000 * 9.81 * q30 * SLOPE[GAUGE_REACH] * LENGTH * 0.8 / 1e6
check("power: gauge-reach power within 10% of independent calculation",
      abs(pw.loc[GAUGE_REACH, "power_MW"] - exp) / exp < 0.1,
      f"{pw.loc[GAUGE_REACH, 'power_MW']:.2f} MW vs {exp:.2f} MW expected")
check("power: upstream design flow excludes Clearwater contribution",
      pw.loc[UP[0], "design_flow_m3s"] < pw.loc[GAUGE_REACH, "design_flow_m3s"] * 0.8)
check("power: figures written",
      os.path.exists("figures/result3_profile.png") and os.path.exists("figures/result4_power.png"))

print("\n-- 15 screening --")
p = run("15_screening.py")
check("screening: completes without error", p.returncode == 0)
if p.returncode == 0:
    sc = pd.read_csv("outputs/screening.csv", dtype={"reach_id": str}).set_index("reach_id")
    summ15 = json.load(open("outputs/screening_summary.json"))
    cal = [c for c in summ15["calibrations"] if c["role"] == "gauge section"][0]
    check("screening A: SWOT-derived Q30 within 10% of gauge on matching dates",
          abs(cal["q30_diff_pct"]) < 10,
          f"gauge {cal['q30_gauge_same_dates']:.0f} vs SWOT {cal['q30_swot_same_dates']:.0f} m3/s "
          f"({cal['q30_diff_pct']:+.1f}%)")
    check("screening A: upstream reaches calibrated against main gauge minus Clearwater",
          any("Clearwater" in c["role"] for c in summ15["calibrations"])
          and os.path.exists("outputs/upstream_calibration.csv"))
    v = sc["velocity_ms"].dropna()
    check("screening B: velocities positive, higher for lower Manning n",
          (v > 0).all() and (sc["velocity_high_ms"] >= sc["velocity_ms"]).all()
          and (sc["velocity_ms"] >= sc["velocity_low_ms"]).all(),
          f"{v.min():.2f}-{v.max():.2f} m/s")
    r = sc.loc[GAUGE_REACH]
    # Manning depth for a wide rectangular channel (R ~ D): D = (Q n / (W sqrt(S)))^0.6, n = 0.035
    D = (r.design_flow_m3s * 0.035 / (r.width_m * np.sqrt(r.slope_cm_per_km / 1e5))) ** 0.6
    check("screening B: Manning depth matches independent calculation",
          abs(D - r.depth_m) < 1e-6, f"{D:.2f} m")
    check("screening C: capacity factor in [0, 1], energy not above rated output",
          sc["capacity_factor"].between(0, 1).all()
          and (sc["energy_GWh_per_yr"] <= sc["power_MW"] * 8.76 + 1e-9).all(),
          f"CF {sc['capacity_factor'].min():.2f}-{sc['capacity_factor'].max():.2f}")
    check("screening D: shortlist non-empty, all shortlisted reaches pass",
          sc["shortlisted"].any() and sc.loc[sc["shortlisted"], "passes"].all(),
          ", ".join(sc.index[sc["shortlisted"]]))
    check("screening: all three figures written",
          all(os.path.exists(f"figures/{f}") for f in
              ("result5_flow_duration.png", "result6_velocity.png", "result7_screening.png")))

# Constant 100 m3/s for 2 years: the 10% environmental flow leaves 90 m3/s,
# which exceeds the 80 m3/s design flow, so the plant runs at rated output
# throughout (CF = 1).
m15 = load("15_screening")
days2 = pd.date_range("2001-01-01", "2002-12-31", freq="D")
e, cf, share, ny = m15.annual_energy(pd.Series(100.0, index=days2.date), 10.0, 80.0, 0.8, 0.10, 0.15)
expected = 1000 * 9.81 * 80 * 10 * 0.8 * 24 * 365 / 1e9
check("screening C: constant-flow energy and CF match analytical values",
      abs(e - expected) / expected < 0.01 and abs(cf - 1.0) < 0.01,
      f"{e:.3f} vs {expected:.3f} GWh/yr, CF {cf:.3f}")

print("\n-- tidy config, summary PDF, README --")
template = open(os.path.join(ROOT, "swot_config.yaml")).read()
if "reaches: []" in template:       # applies only while the config is the unfilled template
    # Append reaches and gauge_reach (duplicate keys), then run tidy_config.py.
    dup = template + '\nreaches:\n  - "11111111111"\n  - "22222222221"\ngauge_reach: "11111111111"\n'
    open("dup_config.yaml", "w").write(dup)
    t = subprocess.run([sys.executable, os.path.join(SCRIPTS, "tidy_config.py"),
                        "--config", "dup_config.yaml", "--write"], capture_output=True, text=True)
    tidied = open("dup_config.yaml").read()
    check("tidy config: duplicate keys merged, loaded settings unchanged",
          t.returncode == 0 and yaml.safe_load(tidied) == yaml.safe_load(dup)
          and tidied.count("\nreaches:") == 1 and tidied.count("\ngauge_reach:") == 1
          and tidied.index("reaches:") < tidied.index("open_water_months"),
          t.stdout.splitlines()[0] if t.stdout else t.stderr[-300:])

# The PDF summary requires reportlab; these checks are skipped if it is not installed.
try:
    import reportlab  # noqa: F401
    have_reportlab = True
except ImportError:
    have_reportlab = False
if have_reportlab:
    ms_path = os.path.join(ROOT, "report", "make_summary.py")
    spec = importlib.util.spec_from_file_location("make_summary", ms_path)
    ms = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ms)
    if shutil.which("git"):
        # Test each GitHub remote URL format (HTTPS with and without .git, and SSH).
        subprocess.run(["git", "init", "-q"], check=True)
        found = []
        for url in ("https://github.com/someone/river-power.git",
                    "git@github.com:someone/river-power.git",
                    "https://github.com/someone/river-power"):
            subprocess.run(["git", "remote", "remove", "origin"], capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", url], check=True)
            found.append(ms.repo_url())
        check("summary: GitHub URL parsed from git remote (HTTPS and SSH)",
              set(found) == {"github.com/someone/river-power"}, ", ".join(sorted(set(found))))
    # Minimal README containing the results markers that make_summary.py writes between.
    open("README.md", "w").write("# x\n\n## Results\n\n<!-- results:start (auto) -->\nPLACEHOLDER-TEXT\n"
                                 "<!-- results:end -->\n\n## After\n")
    p = subprocess.run([sys.executable, ms_path], capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:])
    check("summary: PDF generated", p.returncode == 0 and os.path.getsize("report/summary.pdf") > 20000)
    readme = open("README.md").read()
    check("summary: README results section updated, other content unchanged",
          "PLACEHOLDER-TEXT" not in readme and "Does SWOT measure" in readme and "| Rank |" in readme
          and readme.startswith("# x") and readme.rstrip().endswith("## After"))

# Remove the temporary directory.
os.chdir(ROOT)
shutil.rmtree(work)
print("\n" + "=" * 66)
print(f"{sum(results)}/{len(results)} checks passed")
print("=" * 66)
sys.exit(0 if all(results) else 1)
