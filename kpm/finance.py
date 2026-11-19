"""PPA settlement, single-buyer (KPLC) cost of supply and retail tariff build-up.

Kenya operates a single-buyer model: KPLC buys all energy under long-term PPAs
(capacity charge + energy charge + fuel pass-through, deemed energy for
take-or-pay plants) and recovers costs through EPRA-approved tariffs, the Fuel
Energy Cost Charge (FCC) and levies. This module settles each plant's PPA,
values every plant against the modelled locational marginal price (a
hypothetical wholesale market), and builds the average retail tariff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Inputs
from .dispatch import DispatchResult

VRE_TOP_TECHS = {"wind", "solar", "geothermal", "hydro_ror", "import", "biomass"}


def settle(inp: Inputs, r: DispatchResult) -> dict:
    fin = inp.cfg["finance"]
    fx = fin["fx_kes_per_usd"]
    dt = r.dt
    hours = len(r.index) * dt
    yr_frac = hours / 8760
    g = r.gens.set_index("id")
    gen = r.data["gen"] * dt                     # MWh per period
    avail = r.data["avail"] * dt
    price = r.data["price"]
    zi = {z: i for i, z in enumerate(r.zones)}
    lmp = price[:, g.zone.map(zi).to_numpy()]   # each plant's nodal price

    gen_mwh = gen.sum(0)
    avail_mwh = avail.sum(0)
    top = g.take_or_pay.astype(bool).to_numpy() & g.tech.isin(VRE_TOP_TECHS).to_numpy()
    curtailed = np.where(top, np.maximum(avail_mwh - gen_mwh, 0), 0.0)

    startup = np.zeros(len(g))
    if len(r.data["th"]):
        su_cost = (g.startup_usd_mw.to_numpy() * g.capacity_mw.to_numpy())[r.data["th"]]
        startup[r.data["th"]] = (r.data["su"] * su_cost).sum(0)

    cp = inp.cfg.get("carbon_price_usd_t", 0.0)
    tbl = pd.DataFrame({
        "name": g.name, "owner": g.owner, "zone": g.zone, "tech": g.tech, "capacity_mw": g.capacity_mw,
        "generation_gwh": gen_mwh / 1000,
        "capacity_factor": gen_mwh / (g.capacity_mw.to_numpy() * hours),
        "curtailed_gwh": curtailed / 1000,
        "capacity_payment_musd": g.capacity_charge_usd_kw_month * g.capacity_mw * 12 * yr_frac / 1000,
        "energy_payment_musd": g.energy_charge_usd_mwh * gen_mwh / 1e6,
        "deemed_energy_musd": g.energy_charge_usd_mwh * curtailed / 1e6,
        "fuel_cost_musd": g.fuel_cost_usd_mwh * gen_mwh / 1e6,
        "startup_musd": startup / 1e6,
        "var_om_musd": g.var_om_usd_mwh * gen_mwh / 1e6,
        "carbon_musd": g.co2_rate * cp * gen_mwh / 1e6,
        "co2_kt": g.co2_rate * gen_mwh / 1000,
        "market_revenue_musd": (gen * lmp).sum(0) / 1e6,
    }, index=g.index)
    pay_cols = ["capacity_payment_musd", "energy_payment_musd", "deemed_energy_musd", "fuel_cost_musd",
                "startup_musd", "var_om_musd", "carbon_musd"]
    tbl["total_cost_musd"] = tbl[pay_cols].sum(axis=1)
    tbl["unit_cost_usd_mwh"] = tbl.total_cost_musd * 1e6 / np.maximum(gen_mwh, 1)
    tbl["capture_price_usd_mwh"] = tbl.market_revenue_musd * 1e6 / np.maximum(gen_mwh, 1)
    tbl["market_margin_musd"] = tbl.market_revenue_musd - tbl.total_cost_musd

    # storage
    if len(r.storage):
        s = r.storage.set_index("id")
        dis, ch = r.data["dis"] * dt, r.data["ch"] * dt
        sp = price[:, r.storage.zone.map(zi).to_numpy()]
        stbl = pd.DataFrame({"zone": s.zone, "power_mw": s.power_mw, "energy_mwh": s.power_mw * s.duration_h,
                             "discharge_gwh": dis.sum(0) / 1000, "charge_gwh": ch.sum(0) / 1000,
                             "cycles": dis.sum(0) / (s.power_mw * s.duration_h).to_numpy(),
                             "capacity_payment_musd": s.capacity_charge_usd_kw_month * s.power_mw * 12 * yr_frac / 1000,
                             "arbitrage_revenue_musd": ((dis - ch) * sp).sum(0) / 1e6}, index=s.index)
    else:
        stbl = pd.DataFrame()

    # exports
    exp_rev = 0.0
    if len(r.exports):
        exp_rev = float((r.data["export"] * dt * r.exports.price_usd_mwh.to_numpy()).sum() / 1e6)

    demand = r.data["demand"] * dt
    ens = r.data["ens"] * dt
    served = demand.sum() - ens.sum()
    sales_mwh = served * (1 - fin["td_loss_frac"])
    sales_kwh = sales_mwh * 1000
    st_cap = float(stbl.capacity_payment_musd.sum()) if len(stbl) else 0.0
    ppc = tbl.total_cost_musd.sum() + st_cap          # power purchase cost
    fuel = tbl.fuel_cost_musd.sum() + tbl.startup_musd.sum()
    nonfuel = ppc - fuel - exp_rev
    trans = fin["transmission_cost_usd_mwh"] * served / 1e6
    lv = fin["levies"]

    k = lambda musd: musd * 1e6 * fx / sales_kwh     # MUSD -> KES/kWh sold
    gen_nf, fcc, tr = k(nonfuel), k(fuel), k(trans)
    dist = fin["distribution_cost_kes_kwh"]
    base = gen_nf + tr + dist
    rep = lv["rep_levy_frac"] * base
    pre_vat = base + fcc + rep + lv["epra_levy_kes_kwh"] + lv["wra_levy_kes_kwh"]
    vat = lv["vat_frac"] * pre_vat
    tariff = pd.Series({
        "Generation (non-fuel: capacity, energy, deemed)": gen_nf,
        "Fuel Energy Cost charge (FCC)": fcc,
        "Transmission": tr,
        "Distribution & retail": dist,
        "REP levy": rep,
        "EPRA + WRA levies": lv["epra_levy_kes_kwh"] + lv["wra_levy_kes_kwh"],
        "VAT": vat,
    }, name="KES/kWh")

    co2 = tbl.co2_kt.sum()
    lmp_avg = float((price * demand).sum() / demand.sum())
    tot_gen = max(tbl.generation_gwh.sum(), 1e-9)
    kpis = {
        "Demand sent-out (GWh)": demand.sum() / 1000,
        "Peak demand (MW)": float(r.data["demand"].sum(1).max()),
        "Unserved energy (MWh)": ens.sum(),
        "Hours with unserved energy": int((r.data["ens"].sum(1) > 1e-3).sum() * dt),
        "Reserve shortfall (MWh)": float(r.data["reserve_short"].sum() * dt),
        "Curtailed take-or-pay energy (GWh)": tbl.curtailed_gwh.sum(),
        "Deemed energy payments (MUSD)": tbl.deemed_energy_musd.sum(),
        "Thermal generation share (%)": 100 * tbl.loc[tbl.tech == "thermal", "generation_gwh"].sum() / tot_gen,
        "Renewable share (%)": 100 * tbl.loc[tbl.tech.isin(["geothermal", "hydro_res", "hydro_ror", "wind", "solar", "biomass"]), "generation_gwh"].sum() / tot_gen,
        "Import share (%)": 100 * tbl.loc[tbl.tech == "import", "generation_gwh"].sum() / tot_gen,
        "Power purchase cost (MUSD)": ppc,
        "  of which fuel & start-up (MUSD)": fuel,
        "Export revenue (MUSD)": exp_rev,
        "Avg power purchase cost (US c/kWh sent-out)": 100 * (ppc - exp_rev) * 1e6 / (served * 1000),
        "Load-weighted LMP (USD/MWh)": lmp_avg,
        "Avg retail tariff incl. taxes (KES/kWh)": float(pre_vat + vat),
        "Fuel cost charge (KES/kWh)": fcc,
        "CO2 emissions (kt)": co2,
        "Grid emission factor (tCO2/MWh)": co2 * 1000 / max(gen_mwh.sum(), 1),
        "Solve time (s)": r.runtime_s,
    }
    return {"plants": tbl, "storage": stbl, "tariff": tariff, "kpis": pd.Series(kpis)}
