"""Least-cost capacity expansion (generation planning), LCPDP-style.

Multi-year LP over the planning horizon with 12 representative days (one per
month, 24 h each, weighted by days in month). Co-optimises new-build decisions
for candidate technologies and dispatch of existing + new plant, subject to:
  - energy balance in every representative hour (dual -> long-run marginal price)
  - monthly reservoir energy balance for existing hydro (seasonal storage)
  - daily cycling of batteries, monthly energy limits for new hydro
  - planning reserve margin on firm capacity (dual -> capacity price)
  - build-rate, resource-potential and earliest-commissioning limits
  - optional minimum clean-energy share and annual CO2 cap
Investment costs are annualised with a capital-recovery factor and discounted.
Network is treated as a single node (copper plate) at this planning level.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data import FIRM_CREDIT, FOSSIL_FUELS, Inputs, plant_economics
from .lp import LPBuilder
from .profiles import demand_profile, hour_index, profile_cf


def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def run_expansion(inp: Inputs, progress=None) -> dict:
    t0 = time.time()
    ex = inp.cfg["expansion"]
    years = list(range(int(ex["start_year"]), int(ex["end_year"]) + 1, int(ex["year_step"])))
    step, dr, Y = int(ex["year_step"]), float(ex["discount_rate"]), len(years)
    hyd = ex.get("hydrology", "normal")
    idx = hour_index(years[0])
    mon, hr = idx.month.to_numpy(), idx.hour.to_numpy()
    key = (mon - 1) * 24 + hr
    S = 288
    dim = pd.Series(1, index=idx).groupby(mon).sum().to_numpy() / 24       # days in month
    w = np.repeat(dim, 24)                                                 # hours represented by a slice
    m_of_s = np.repeat(np.arange(12), 24)
    rep = lambda x: np.bincount(key, weights=x, minlength=S) / np.bincount(key, minlength=S)
    cfc = {}

    def cfp(name):
        if name not in cfc:
            cfc[name] = rep(profile_cf(inp, name, idx, hyd)) if name else np.ones(S)
        return cfc[name]

    if progress:
        progress(0.05, "Building demand projections")
    D, peak = np.zeros((Y, S)), np.zeros(Y)
    for j, y in enumerate(years):
        d = demand_profile(inp, y).sum(axis=1).to_numpy()
        D[j] = rep(d[: len(key)]) if len(d) >= len(key) else rep(np.resize(d, len(key)))
        peak[j] = d.max()
    DF = np.array([(1 + dr) ** -(y - years[0]) for y in years])
    wy = (DF * step)[:, None] * w[None, :]                                 # objective weight per (y,s)
    cp = inp.cfg.get("carbon_price_usd_t", 0.0)
    fuels = inp.cfg["fuels"]

    lp = LPBuilder()
    # ---------------- existing fleet
    gens = plant_economics(inp, inp.gens[inp.gens.capacity_mw > 0].reset_index(drop=True))
    E = len(gens)
    act = np.array([[(g.commission_year <= y) & (g.retire_year > y) for y in years] for _, g in gens.iterrows()], float)
    capav = (gens.capacity_mw * gens.availability).to_numpy()
    cf_e = np.array([cfp(p) if p else np.ones(S) for p in gens.profile])
    ub_e = capav[:, None, None] * act[:, :, None] * cf_e[:, None, :]
    cost_e = gens.dispatch_cost.to_numpy()[:, None, None] * wy[None]
    ge = lp.var((E, Y, S), 0, ub_e, cost_e)

    # ---------------- reservoirs (monthly energy balance)
    res = inp.reservoirs
    R = len(res)
    infl = np.array([[inp.inflows[(inp.inflows.reservoir == r)].set_index("month")[hyd].loc[m + 1] * 1000
                      for m in range(12)] for r in res.reservoir]) * inp.cfg["hydrology"].get("inflow_multiplier", 1.0)
    smax, smin = res.max_storage_gwh.to_numpy() * 1000, res.min_storage_gwh.to_numpy() * 1000
    s0 = res.initial_frac.to_numpy() * smax
    sres = lp.var((R, Y, 12), smin[:, None, None], smax[:, None, None])
    spill = lp.var((R, Y, 12), 0, np.inf)
    rhs = np.broadcast_to(infl[:, None, :], (R, Y, 12)).copy()
    rhs[:, :, 0] += s0[:, None]
    rb = lp.rows((R, Y, 12), "=", rhs)
    lp.terms(rb, sres, 1)
    lp.terms(rb, spill, 1)
    lp.terms(rb[:, :, 1:], sres[:, :, :-1], -1)
    rmap = {r: i for i, r in enumerate(res.reservoir)}
    for gi, rname in enumerate(gens.reservoir):
        if rname:
            ri = rmap[rname]
            for mth in range(12):
                sl = np.where(m_of_s == mth)[0]
                lp.terms(rb[ri, :, mth][:, None], ge[gi][:, sl], dim[mth])
    r_end = lp.rows((R, Y), ">", np.broadcast_to(s0[:, None], (R, Y)))
    lp.terms(r_end, sres[:, :, -1], 1)

    # ---------------- existing storage
    st = inp.storage[inp.storage.power_mw > 0].reset_index(drop=True)
    Ss = len(st)
    if Ss:
        sact = np.array([[(s.commission_year <= y) & (s.retire_year > y) for y in years] for _, s in st.iterrows()], float)
        P = st.power_mw.to_numpy()[:, None, None] * sact[:, :, None]
        Emax = (st.power_mw * st.duration_h).to_numpy()[:, None, None] * sact[:, :, None]
        eta = np.sqrt(st.efficiency_rt.to_numpy())[:, None, None]
        sch = lp.var((Ss, Y, S), 0, np.broadcast_to(P, (Ss, Y, S)))
        sdis = lp.var((Ss, Y, S), 0, np.broadcast_to(P, (Ss, Y, S)), cost=0.1 * wy[None])
        ssoc = lp.var((Ss, Y, S), 0, np.broadcast_to(Emax, (Ss, Y, S)))
        _storage_dynamics(lp, sch, sdis, ssoc, eta)

    # ---------------- candidates
    cand = inp.candidates.copy()
    C = len(cand)
    ef = cand.fuel.map(lambda f: fuels[f]["ef"]).to_numpy()
    fp = cand.fuel.map(lambda f: fuels[f]["price"]).to_numpy()
    co2c = cand.heat_rate_gj_mwh.to_numpy() * ef + cand.co2_t_mwh.to_numpy()
    vcost = cand.heat_rate_gj_mwh.to_numpy() * fp + cand.vom_usd_mwh.to_numpy() + co2c * cp
    ann = (cand.capex_usd_kw * cand.apply(lambda c: crf(dr, c.life_yr), axis=1) + cand.fom_usd_kw_yr).to_numpy() * 1000
    # cost of 1 MW built in period j = annuity paid in every modelled year >= j
    cumDF = np.array([(DF[j:] * step).sum() for j in range(Y)])
    bub = np.array([[c.max_build_mw_yr * step if y >= c.earliest_year else 0.0 for y in years] for _, c in cand.iterrows()])
    b = lp.var((C, Y), 0, bub, cost=ann[:, None] * cumDF[None, :])
    r = lp.rows(C, "<", cand.max_total_mw.to_numpy())
    lp.terms(r[:, None], b, 1)
    is_bess = (cand.tech == "bess").to_numpy()
    is_hyd = (cand.tech == "hydro_new").to_numpy()
    gen_c = np.where(~is_bess)[0]
    cf_c = np.array([cfp(p) if (p and t != "hydro_new") else np.ones(S) for p, t in zip(cand.profile, cand.tech)])
    gc = lp.var((len(gen_c), Y, S), 0, np.inf, cost=vcost[gen_c, None, None] * wy[None])
    capr = lp.rows((len(gen_c), Y, S), "<", 0)
    lp.terms(capr, gc, 1)
    av = cand.availability.to_numpy()
    for j in range(Y):
        for jj in range(j + 1):
            lp.terms(capr[:, j, :], b[gen_c, jj][:, None], -(av[gen_c, None] * cf_c[gen_c]))
    for k, ci in enumerate(gen_c):
        if is_hyd[ci]:
            prof = cand.profile[ci] or "hydro_new"
            cfm = inp.ror_cf[inp.ror_cf.profile == prof].set_index("month")[hyd].to_numpy()
            mr = lp.rows((Y, 12), "<", 0)
            for mth in range(12):
                sl = np.where(m_of_s == mth)[0]
                lp.terms(mr[:, mth][:, None], gc[k][:, sl], 1)
                for j in range(Y):
                    for jj in range(j + 1):
                        lp.terms(mr[j, mth], b[ci, jj], -24 * cfm[mth] * av[ci])
    bess_c = np.where(is_bess)[0]
    if len(bess_c):
        nb = len(bess_c)
        bch = lp.var((nb, Y, S), 0, np.inf)
        bdis = lp.var((nb, Y, S), 0, np.inf, cost=(cand.vom_usd_mwh.to_numpy()[bess_c, None, None]) * wy[None])
        bsoc = lp.var((nb, Y, S), 0, np.inf)
        _storage_dynamics(lp, bch, bdis, bsoc, np.sqrt(cand.efficiency_rt.to_numpy()[bess_c])[:, None, None])
        dur = cand.duration_h.to_numpy()[bess_c]
        for var, mult in ((bch, np.ones(nb)), (bdis, np.ones(nb)), (bsoc, dur)):
            rr = lp.rows((nb, Y, S), "<", 0)
            lp.terms(rr, var, 1)
            for j in range(Y):
                for jj in range(j + 1):
                    lp.terms(rr[:, j, :], b[bess_c, jj][:, None], -mult[:, None] * np.ones(S))

    # ---------------- balance
    voll = inp.cfg["operation"]["voll_usd_mwh"]
    ens = lp.var((Y, S), 0, D, cost=voll * wy)
    bal = lp.rows((Y, S), "=", D)
    lp.terms(bal[None], ge, 1)
    lp.terms(bal[None], gc, 1)
    lp.terms(bal, ens, 1)
    if Ss:
        lp.terms(bal[None], sdis, 1)
        lp.terms(bal[None], sch, -1)
    if len(bess_c):
        lp.terms(bal[None], bdis, 1)
        lp.terms(bal[None], bch, -1)

    # ---------------- planning reserve margin
    credit_e = gens.tech.map(FIRM_CREDIT).fillna(0.5).to_numpy()
    firm_exist = (credit_e[:, None] * gens.capacity_mw.to_numpy()[:, None] * act).sum(0)
    if Ss:
        firm_exist = firm_exist + (FIRM_CREDIT["bess"] * st.power_mw.to_numpy()[:, None] * sact).sum(0)
    rm = lp.rows(Y, ">", peak * (1 + ex["reserve_margin"]) - firm_exist)
    rm_short = lp.var(Y, 0, np.inf, cost=float(ex.get("capacity_shortfall_usd_kw_yr", 600)) * 1000 * DF * step)
    lp.terms(rm, rm_short, 1)
    for j in range(Y):
        for jj in range(j + 1):
            lp.terms(rm[j], b[:, jj], cand.firm_credit.to_numpy())

    # ---------------- policy constraints
    fossil_e = gens.fuel.isin(FOSSIL_FUELS).to_numpy()
    fossil_c = cand.fuel.isin(FOSSIL_FUELS).to_numpy()[gen_c]
    share = float(ex.get("min_clean_share") or 0.0)
    if share > 0:
        for j, y in enumerate(years):
            if y >= int(ex.get("clean_target_year", years[0])):
                rr = lp.rows(1, "<", (1 - share) * (D[j] * w).sum())
                if fossil_e.any():
                    lp.terms(rr, ge[fossil_e][:, j, :], np.broadcast_to(w, ge[fossil_e][:, j, :].shape))
                if fossil_c.any():
                    lp.terms(rr, gc[fossil_c][:, j, :], np.broadcast_to(w, gc[fossil_c][:, j, :].shape))
    co2e = gens.co2_rate.to_numpy()
    co2cg = co2c[gen_c]
    if ex.get("co2_cap_mt"):
        rr = lp.rows(Y, "<", float(ex["co2_cap_mt"]) * 1e6)
        lp.terms(rr[None, :, None], ge, co2e[:, None, None] * w[None, None, :])
        lp.terms(rr[None, :, None], gc, co2cg[:, None, None] * w[None, None, :])

    if progress:
        progress(0.3, f"Solving LP ({lp.n:,} variables, {lp.m:,} constraints)")
    sol = lp.solve()
    if progress:
        progress(1.0, "Done")

    # ---------------- results
    bv = sol.val(b)
    builds = pd.DataFrame(bv, index=cand.id, columns=years)
    cum = builds.cumsum(axis=1)
    cap_exist = pd.DataFrame(gens.capacity_mw.to_numpy()[:, None] * act, index=gens.id, columns=years).groupby(gens.tech.to_numpy()).sum()
    if Ss:
        if "bess" not in cap_exist.index:
            cap_exist.loc["bess"] = 0.0
        cap_exist.loc["bess"] += (st.power_mw.to_numpy()[:, None] * sact).sum(0)
    cap_new = cum.groupby(cand.tech.to_numpy()).sum()
    capacity = cap_exist.add(cap_new, fill_value=0)
    gev, gcv = sol.val(ge), sol.val(gc)
    gen_e = pd.DataFrame((gev * w).sum(2) / 1000, index=gens.id, columns=years).groupby(gens.tech.to_numpy()).sum()
    gen_n = pd.DataFrame((gcv * w).sum(2) / 1000, index=cand.id[gen_c], columns=years).groupby(cand.tech.to_numpy()[gen_c]).sum()
    generation = gen_e.add(gen_n, fill_value=0)
    net_st = np.zeros(Y)
    if Ss:
        net_st += (((sol.val(sdis) - sol.val(sch)) * w).sum(2).sum(0)) / 1000
    if len(bess_c):
        net_st += (((sol.val(bdis) - sol.val(bch)) * w).sum(2).sum(0)) / 1000
    if Ss or len(bess_c):
        generation.loc["bess"] = net_st
    lam = sol.mu(bal) / wy
    price = (lam * D * w).sum(1) / (D * w).sum(1)
    cap_price = sol.mu(rm) / (DF * step) / 1000
    co2 = ((gev * w).sum(2) * co2e[:, None]).sum(0) / 1e6 + ((gcv * w).sum(2) * co2cg[:, None]).sum(0) / 1e6
    ensv = (sol.val(ens) * w).sum(1)
    fossil_gen = ((gev[fossil_e] * w).sum(2).sum(0) + (gcv[fossil_c] * w).sum(2).sum(0)) / 1000
    summary = pd.DataFrame({"peak_mw": peak, "demand_gwh": (D * w).sum(1) / 1000, "firm_existing_mw": firm_exist,
                            "new_build_mw": bv.sum(0), "avg_marginal_price_usd_mwh": price,
                            "capacity_price_usd_kw_yr": cap_price, "co2_mt": co2, "unserved_mwh": ensv,
                            "firm_capacity_shortfall_mw": sol.val(rm_short)}, index=years)
    summary["clean_share_pct"] = 100 * (1 - fossil_gen / summary.demand_gwh.to_numpy())
    hourly_new = {cand.id[ci]: gcv[k] for k, ci in enumerate(gen_c)}
    return {"years": years, "builds": builds, "cumulative": cum, "capacity": capacity, "generation": generation,
            "summary": summary, "npv_cost_musd": sol.obj / 1e6, "price_slices": pd.DataFrame(lam, index=years),
            "demand_slices": pd.DataFrame(D, index=years), "hourly_new": hourly_new,
            "candidates": cand, "runtime_s": time.time() - t0, "lp_size": (lp.n, lp.m)}


def _storage_dynamics(lp, ch, dis, soc, eta):
    """Daily-cyclic state of charge for (n, Y, 288) storage arrays (12 rep days x 24 h)."""
    n, Y, S = ch.shape
    r = lp.rows((n, Y, S), "=", 0)
    lp.terms(r, soc, 1)
    prev = soc.reshape(n, Y, 12, 24)
    prev = np.roll(prev, 1, axis=3).reshape(n, Y, S)        # hour 0 links to hour 23 of the same day
    lp.terms(r, prev, -1)
    lp.terms(r, ch, -np.broadcast_to(eta, (n, Y, S)))
    lp.terms(r, dis, 1 / np.broadcast_to(eta, (n, Y, S)))
