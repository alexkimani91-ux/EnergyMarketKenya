"""Stochastic risk analysis: Monte Carlo over hydrology, demand, fuel price, wind and FX.

Each draw perturbs the inputs and re-runs the full two-stage dispatch and PPA
settlement (at a coarser time resolution for speed), giving distributions of
cost of supply, tariff, fuel cost charge, unserved energy, curtailment and CO2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Inputs
from .dispatch import run_dispatch
from .finance import settle

DEFAULT_UNCERTAINTY = {
    "hydro_probs": {"dry": 0.25, "normal": 0.50, "wet": 0.25},
    "inflow_sd": 0.10,          # lognormal sd on inflows within the chosen hydrology class
    "demand_growth_sd": 0.015,  # absolute sd on annual energy growth rate
    "fuel_price_sd": 0.20,      # lognormal sd on HFO / gasoil prices
    "wind_cf_sd": 0.06,         # relative sd on wind capacity factors
    "fx_sd": 0.05,              # relative sd on KES/USD
    "geo_avail_sd": 0.04,       # absolute sd on geothermal availability (steam decline / outages)
}


def run_monte_carlo(inp: Inputs, n: int = 20, resolution: int = 3, unc: dict | None = None,
                    seed: int = 7, progress=None) -> pd.DataFrame:
    u = {**DEFAULT_UNCERTAINTY, **(unc or {})}
    rng = np.random.default_rng(seed)
    rows = []
    classes = list(u["hydro_probs"])
    probs = np.array([u["hydro_probs"][c] for c in classes], float)
    probs /= probs.sum()
    for i in range(n):
        x = inp.copy()
        c = x.cfg
        c["random_seed"] = int(rng.integers(1, 10**6))
        hyd = str(rng.choice(classes, p=probs))
        c["hydrology"]["scenario"] = hyd
        c["hydrology"]["inflow_multiplier"] = float(c["hydrology"].get("inflow_multiplier", 1) * rng.lognormal(0, u["inflow_sd"]))
        c["demand"]["energy_growth"] = float(c["demand"]["energy_growth"] + rng.normal(0, u["demand_growth_sd"]))
        c["demand"]["peak_growth"] = float(c["demand"]["peak_growth"] + rng.normal(0, u["demand_growth_sd"]))
        fm = float(rng.lognormal(0, u["fuel_price_sd"]))
        for f in ("HFO", "GASOIL", "LNG"):
            if f in c["fuels"]:
                c["fuels"][f]["price"] *= fm
        wm = float(max(0.5, rng.normal(1, u["wind_cf_sd"])))
        for p in c["vre_profiles"].values():
            if p["type"] == "wind":
                p["cf"] = min(0.9, p["cf"] * wm)
        ga = float(rng.normal(0, u["geo_avail_sd"]))
        geo = x.gens.tech == "geothermal"
        x.gens.loc[geo, "availability"] = (x.gens.loc[geo, "availability"] + ga).clip(0.3, 0.98)
        fx = float(c["finance"]["fx_kes_per_usd"] * rng.normal(1, u["fx_sd"]))
        c["finance"]["fx_kes_per_usd"] = fx
        c["operation"]["resolution_hours"] = resolution
        c["operation"]["unit_commitment"] = "lp"
        r = run_dispatch(x)
        k = settle(x, r)["kpis"]
        rows.append({"draw": i + 1, "hydrology": hyd, "inflow_mult": c["hydrology"]["inflow_multiplier"],
                     "energy_growth": c["demand"]["energy_growth"], "fuel_mult": fm, "wind_mult": wm,
                     "geo_avail_shift": ga, "fx": fx, **k.to_dict()})
        if progress:
            progress((i + 1) / n, f"Monte Carlo draw {i + 1}/{n} ({hyd})")
    return pd.DataFrame(rows)


def summarise(mc: pd.DataFrame) -> pd.DataFrame:
    cols = ["Avg power purchase cost (US c/kWh sent-out)", "Avg retail tariff incl. taxes (KES/kWh)",
            "Fuel cost charge (KES/kWh)", "Power purchase cost (MUSD)", "Thermal generation share (%)",
            "Unserved energy (MWh)", "Curtailed take-or-pay energy (GWh)", "Deemed energy payments (MUSD)",
            "CO2 emissions (kt)", "Load-weighted LMP (USD/MWh)"]
    q = mc[cols].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    q.columns = ["P5", "P25", "P50", "P75", "P95"]
    q["mean"] = mc[cols].mean()
    return q
