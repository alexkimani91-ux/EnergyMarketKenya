"""Hourly time-series: zonal demand, wind/solar capacity factors, run-of-river and reservoir inflows.

Synthetic profiles are calibrated to annual targets (energy, peak, capacity factor).
Any profile can be replaced by measured data: drop an 8760/8784-row CSV with a
column ``cf`` (VRE / RoR) into ``inputs/profiles/<profile>.csv``, or a file
``inputs/profiles/demand.csv`` with one MW column per zone for the base year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Inputs

RAINY_MONTHS = {3, 4, 5, 10, 11, 12}  # long rains (MAM) and short rains (OND)


def hour_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")


def _rng(inp: Inputs, salt: str) -> np.random.Generator:
    return np.random.default_rng((int(inp.cfg.get("random_seed", 0)) * 1_000_003 + sum(map(ord, salt))) % 2**32)


def _override(inp: Inputs, name: str, n: int):
    f = inp.input_dir / "profiles" / f"{name}.csv"
    if f.exists():
        v = pd.read_csv(f)["cf"].to_numpy(float)
        return np.resize(v, n)
    return None


def _fit_cf(raw: np.ndarray, target: float) -> np.ndarray:
    """Scale a non-negative series so that mean(clip(s*raw,0,1)) == target."""
    lo, hi = 0.0, 50.0 / max(raw.mean(), 1e-9)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.clip(raw * mid, 0, 1).mean() < target:
            lo = mid
        else:
            hi = mid
    return np.clip(raw * 0.5 * (lo + hi), 0, 1)


# --------------------------------------------------------------------- demand
def system_targets(inp: Inputs, year: int):
    d = inp.cfg["demand"]
    n = year - int(inp.cfg["base_year"])
    return d["base_energy_gwh"] * (1 + d["energy_growth"]) ** n, d["base_peak_mw"] * (1 + d["peak_growth"]) ** n


def _extra_shape(kind: str, idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour.to_numpy()
    if kind == "evening":     # EV home/depot charging
        w = np.select([(h >= 18) & (h <= 22), (h >= 23) | (h <= 5)], [1.6, 1.2], 0.5)
    elif kind == "cooking":   # e-cooking: breakfast, lunch, supper
        w = np.select([(h >= 6) & (h <= 7), (h >= 12) & (h <= 13), (h >= 18) & (h <= 20)], [2.0, 1.5, 3.0], 0.2)
    elif kind == "solar":     # daytime-only loads (e.g. electrolysis following PV)
        w = np.clip(np.sin(np.pi * (h + 0.5 - 6.4) / 12.3), 0, None)
    else:
        w = np.ones(len(h))
    return w / w.mean()


def demand_profile(inp: Inputs, year: int) -> pd.DataFrame:
    """Hourly sent-out demand (MW) per zone, including policy-driven extra loads."""
    idx = hour_index(year)
    d = inp.cfg["demand"]
    energy, peak = system_targets(inp, year)
    zones = inp.zones.zone.tolist()
    shares = inp.zones.set_index("zone").demand_share
    f = inp.input_dir / "profiles" / "demand.csv"
    if f.exists():  # measured base-year zonal demand, rescaled
        base = pd.read_csv(f)[zones].to_numpy(float)
        base = np.resize(base, (len(idx), len(zones)))
        tot = base.sum(1)
    else:
        shape = np.asarray(d["hourly_shape"], float)[idx.hour]
        wd = idx.dayofweek.to_numpy()
        shape = shape * np.where(wd == 5, d["weekend_factor"]["sat"], np.where(wd == 6, d["weekend_factor"]["sun"], 1.0))
        shape = shape * np.asarray(d["monthly_factor"], float)[idx.month - 1]
        rng = _rng(inp, f"demand{year}")
        e = np.zeros(len(idx))
        eps = rng.normal(0, d.get("noise_sd", 0.0), len(idx))
        for t in range(1, len(idx)):
            e[t] = 0.8 * e[t - 1] + eps[t]
        tot = shape * (1 + e)
        base = None
    # power transform so that both annual energy and peak are matched
    mean_mw = energy * 1000 / len(idx)
    lf = min(mean_mw / peak, 0.99)
    x = tot / tot.max()
    lo, hi = 0.05, 8.0
    for _ in range(60):
        k = 0.5 * (lo + hi)
        if (x ** k).mean() > lf:
            lo = k
        else:
            hi = k
    sysd = x ** k * peak
    if base is None:
        rng = _rng(inp, f"zones{year}")
        zt = np.outer(sysd, shares[zones].to_numpy())
        zt *= 1 + rng.normal(0, 0.01, zt.shape)
    else:
        zt = base / base.sum(1, keepdims=True) * sysd[:, None]
    out = pd.DataFrame(zt, index=idx, columns=zones)
    for name, x_ in (d.get("extra_loads") or {}).items():
        if x_ and x_.get("annual_gwh", 0) > 0:
            out[x_["zone"]] += _extra_shape(x_.get("profile", "flat"), idx) * x_["annual_gwh"] * 1000 / len(idx)
    return out


# ------------------------------------------------------------------ renewables
def solar_cf(inp: Inputs, name: str, idx: pd.DatetimeIndex) -> np.ndarray:
    p = inp.cfg["vre_profiles"][name]
    rng = _rng(inp, name + str(idx[0].year))
    h = idx.hour.to_numpy() + 0.5
    clear = np.clip(np.sin(np.pi * (h - 6.4) / 12.3), 0, None) ** 1.15
    days = len(idx) // 24
    months = idx.month.to_numpy()[::24][:days]
    mean_k = np.where(np.isin(months, list(RAINY_MONTHS)), 0.88 * (1 - p.get("cloud_rainy", 0.25)), 0.88)
    k = rng.beta(mean_k * 8, (1 - mean_k) * 8)
    kd = np.repeat(k, 24)[: len(idx)]
    kd = np.pad(kd, (0, len(idx) - len(kd)), mode="edge")
    raw = clear * kd * (1 + rng.normal(0, 0.05, len(idx))).clip(0.5, 1.3)
    return _fit_cf(raw, p["cf"])


def wind_cf(inp: Inputs, name: str, idx: pd.DatetimeIndex) -> np.ndarray:
    p = inp.cfg["vre_profiles"][name]
    rng = _rng(inp, name + str(idx[0].year))
    h, m = idx.hour.to_numpy(), idx.month.to_numpy()
    diurnal = 1 + p.get("night_bias", 0) * np.cos(2 * np.pi * (h - 3) / 24)       # Turkana low-level jet peaks at night
    season = 1 + p.get("season_amp", 0) * np.cos(2 * np.pi * (m - p.get("peak_month", 7)) / 12)
    phi, sig = 0.97, 0.45
    e = np.zeros(len(idx))
    eps = rng.normal(0, sig * np.sqrt(1 - phi ** 2), len(idx))
    for t in range(1, len(idx)):
        e[t] = phi * e[t - 1] + eps[t]
    speed = diurnal * season * np.exp(e - sig ** 2 / 2)
    raw = np.clip(speed - 0.25, 0, None) ** 2.2           # crude cubic-ish power curve with cut-in
    return _fit_cf(raw, p["cf"])


def ror_series(inp: Inputs, name: str, idx: pd.DatetimeIndex, scenario: str) -> np.ndarray:
    ov = _override(inp, name, len(idx))
    if ov is not None:
        return ov
    t = inp.ror_cf[inp.ror_cf.profile == name].set_index("month")[scenario]
    mult = inp.cfg["hydrology"].get("inflow_multiplier", 1.0)
    return np.clip(t.reindex(idx.month).to_numpy() * mult, 0, 1)


def profile_cf(inp: Inputs, name: str, idx: pd.DatetimeIndex, scenario: str = "normal") -> np.ndarray:
    ov = _override(inp, name, len(idx))
    if ov is not None:
        return ov
    if name in set(inp.ror_cf.profile):
        return ror_series(inp, name, idx, scenario)
    p = inp.cfg["vre_profiles"][name]
    out = solar_cf(inp, name, idx) if p["type"] == "solar" else wind_cf(inp, name, idx)
    return np.clip(out * p.get("scale", 1.0), 0, 1)


def inflow_mw(inp: Inputs, idx: pd.DatetimeIndex, scenario: str) -> pd.DataFrame:
    """Reservoir inflow as average MW-equivalent per hour (monthly GWh spread evenly)."""
    mult = inp.cfg["hydrology"].get("inflow_multiplier", 1.0)
    hrs = pd.Series(1, index=idx).groupby(idx.month).sum()
    out = {}
    for r in inp.reservoirs.reservoir:
        g = inp.inflows[inp.inflows.reservoir == r].set_index("month")[scenario] * mult
        out[r] = (g * 1000 / hrs).reindex(idx.month).to_numpy()
    return pd.DataFrame(out, index=idx)


# ------------------------------------------------------------- bundle builder
def build_timeseries(inp: Inputs, year: int | None = None) -> dict:
    year = inp.year if year is None else year
    idx = hour_index(year)
    scen = inp.cfg["hydrology"]["scenario"]
    gens = inp.active_gens(year)
    demand = demand_profile(inp, year)
    cache, avail = {}, {}
    for _, g in gens.iterrows():
        cap = g.capacity_mw * g.availability
        if g.profile:
            if g.profile not in cache:
                cache[g.profile] = profile_cf(inp, g.profile, idx, scen)
            avail[g.id] = cap * cache[g.profile]
        else:
            avail[g.id] = np.full(len(idx), cap)
    return {"index": idx, "demand": demand, "avail": pd.DataFrame(avail, index=idx),
            "inflow": inflow_mw(inp, idx, scen), "cf": pd.DataFrame(cache, index=idx), "gens": gens}
