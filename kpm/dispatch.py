"""Multi-zone security-constrained economic dispatch / unit commitment.

Two-stage, PLEXOS-style approach:
  Stage 1 (mid-term):  whole year at coarse resolution -> reservoir trajectories
                       and marginal *water values* for the Tana cascade & Turkwel.
  Stage 2 (short-term): rolling weekly windows (+ look-ahead) at hourly resolution,
                       end-of-window water valued at the Stage 1 water value.

Each window is an LP (relaxed unit commitment) or MILP (integer commitment) with:
  zonal energy balance (duals -> locational marginal prices), transport-model
  transmission with losses, thermal min-stable-generation & start-up costs,
  geothermal minimum (venting) limits, VRE/take-or-pay curtailment, reservoir
  hydro energy balance with spill, battery storage, spinning reserve requirement,
  flexible exports, unserved energy at VoLL and over-generation slack.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import Inputs, plant_economics
from .lp import LPBuilder
from .profiles import build_timeseries

VALID_RES = (1, 2, 3, 4, 6, 8, 12, 24)
OVERGEN_PENALTY = 500.0
END_SHORTFALL_PENALTY = 400.0


@dataclass
class Static:
    gens: pd.DataFrame
    storage: pd.DataFrame
    reservoirs: pd.DataFrame
    lines: pd.DataFrame
    zones: list
    exports: pd.DataFrame
    op: dict

    def __post_init__(self):
        zi = {z: i for i, z in enumerate(self.zones)}
        g = self.gens
        self.gz = g.zone.map(zi).to_numpy()
        self.th = np.where(g.tech.isin(["thermal", "nuclear"]).to_numpy())[0]
        self.rc = np.where(g.reserve_capable.astype(bool).to_numpy())[0]
        self.rc_nonth = np.setdiff1d(self.rc, self.th)
        self.rc_th = np.intersect1d(self.rc, self.th)
        rmap = {r: i for i, r in enumerate(self.reservoirs.reservoir)}
        self.g_res = np.array([rmap.get(r, -1) for r in g.reservoir])
        self.sz = self.storage.zone.map(zi).to_numpy().astype(int)
        self.lf = self.lines.from_zone.map(zi).to_numpy()
        self.lt = self.lines.to_zone.map(zi).to_numpy()
        self.xz = self.exports.zone.map(zi).to_numpy().astype(int) if len(self.exports) else np.zeros(0, int)
        s = self.storage
        self.P = s.power_mw.to_numpy(float)
        self.E = (s.power_mw * s.duration_h).to_numpy(float)
        self.eta = np.sqrt(s.efficiency_rt.to_numpy(float))
        self.smax = (self.reservoirs.max_storage_gwh * 1000).to_numpy(float)
        self.smin = (self.reservoirs.min_storage_gwh * 1000).to_numpy(float)


def _agg(x: np.ndarray, b: int) -> np.ndarray:
    if b == 1:
        return x
    T = x.shape[0] // b * b
    return x[:T].reshape(T // b, b, *x.shape[1:]).mean(1)


def solve_window(st: Static, D, A, I, dt, state, end, mip=False):
    """Solve one dispatch window. D:(T,Z) A:(T,G) I:(T,R) in MW; dt hours per period."""
    T, Z = D.shape
    G, R, S, L, X = len(st.gens), len(st.reservoirs), len(st.storage), len(st.lines), len(st.exports)
    op, g_ = st.op, st.gens
    lp = LPBuilder()
    At = A.T
    minf = g_.min_gen_frac.to_numpy(float)
    lbg = np.where(np.isin(np.arange(G), st.th)[:, None], 0.0, minf[:, None] * At)
    g = lp.var((G, T), lb=lbg, ub=At, cost=g_.dispatch_cost.to_numpy(float)[:, None] * dt)

    # --- thermal unit commitment (relaxed or integer)
    nth = len(st.th)
    if nth:
        cap = g_.capacity_mw.to_numpy(float)[st.th]
        on = lp.var((nth, T), 0, 1, integer=mip)
        su = lp.var((nth, T), 0, 1, cost=(g_.startup_usd_mw.to_numpy(float)[st.th] * cap)[:, None])
        r = lp.rows((nth, T), "<", 0)
        lp.terms(r, g[st.th], 1); lp.terms(r, on, -At[st.th])
        r = lp.rows((nth, T), ">", 0)
        lp.terms(r, g[st.th], 1); lp.terms(r, on, -(minf[st.th, None] * At[st.th]))
        rhs = np.zeros((nth, T)); rhs[:, 0] = -state["on"]
        r = lp.rows((nth, T), ">", rhs)
        lp.terms(r, su, 1); lp.terms(r, on, -1); lp.terms(r[:, 1:], on[:, :-1], 1)

    # --- spinning reserve
    rv = lp.var((len(st.rc), T), 0, At[st.rc])
    pos = {gi: k for k, gi in enumerate(st.rc)}
    if len(st.rc_nonth):
        k = [pos[i] for i in st.rc_nonth]
        r = lp.rows((len(k), T), "<", At[st.rc_nonth])
        lp.terms(r, g[st.rc_nonth], 1); lp.terms(r, rv[k], 1)
    if len(st.rc_th):
        k = [pos[i] for i in st.rc_th]
        thpos = [int(np.where(st.th == i)[0][0]) for i in st.rc_th]
        r = lp.rows((len(k), T), "<", 0)
        lp.terms(r, g[st.rc_th], 1); lp.terms(r, rv[k], 1); lp.terms(r, on[thpos], -At[st.rc_th])
    req = np.maximum(op["largest_contingency_mw"], op["reserve_frac_of_load"] * D.sum(1))
    short = lp.var(T, 0, req, cost=op["reserve_shortfall_penalty"] * dt)
    rreq = lp.rows(T, ">", req)
    lp.terms(rreq[None, :], rv, 1); lp.terms(rreq, short, 1)

    # --- reservoir hydro
    if R:
        wv_cost = np.zeros((R, T))
        if end["mode"] == "value":
            wv_cost[:, -1] = -end["wv"]
        s = lp.var((R, T), st.smin[:, None], st.smax[:, None], cost=wv_cost)
        spill = lp.var((R, T), 0, np.inf, cost=0.01 * dt)
        rhs = dt * I.T.copy(); rhs[:, 0] += state["s"]
        rbal = lp.rows((R, T), "=", rhs)
        lp.terms(rbal, s, 1); lp.terms(rbal[:, 1:], s[:, :-1], -1); lp.terms(rbal, spill, dt)
        for gi in np.where(st.g_res >= 0)[0]:
            lp.terms(rbal[st.g_res[gi]], g[gi], dt)
        if end["mode"] == "target":
            es = lp.var(R, 0, np.inf, cost=END_SHORTFALL_PENALTY)
            r = lp.rows(R, ">", end["s"])
            lp.terms(r, s[:, -1], 1); lp.terms(r, es, 1)

    # --- battery storage
    if S:
        ch = lp.var((S, T), 0, st.P[:, None])
        dis = lp.var((S, T), 0, st.P[:, None], cost=0.1 * dt)
        soc = lp.var((S, T), 0, st.E[:, None])
        rs = lp.var((S, T), 0, st.P[:, None])
        rhs = np.zeros((S, T)); rhs[:, 0] = state["soc"]
        r = lp.rows((S, T), "=", rhs)
        lp.terms(r, soc, 1); lp.terms(r[:, 1:], soc[:, :-1], -1)
        lp.terms(r, ch, -dt * st.eta[:, None]); lp.terms(r, dis, dt / st.eta[:, None])
        r = lp.rows((S, T), "<", st.P[:, None] * np.ones(T))
        lp.terms(r, rs, 1); lp.terms(r, dis, 1); lp.terms(r, ch, -1)
        r = lp.rows((S, T), "<", 0)
        lp.terms(r, rs, 1); lp.terms(r, soc, -1)
        sres = st.storage.reserve_capable.astype(bool).to_numpy()
        if sres.any():
            lp.terms(rreq[None, :], rs[sres], 1)
        if end["mode"] == "target":
            r = lp.rows(S, ">", state["soc"] * 0.999)
            lp.terms(r, soc[:, -1], 1)

    # --- network, exports, slack
    hurdle = op["wheeling_hurdle_usd_mwh"] * dt
    capl = st.lines.capacity_mw.to_numpy(float)[:, None]
    fp = lp.var((L, T), 0, capl, cost=hurdle)
    fn = lp.var((L, T), 0, capl, cost=hurdle)
    x = lp.var((X, T), 0, st.exports.max_mw.to_numpy(float)[:, None],
               cost=-st.exports.price_usd_mwh.to_numpy(float)[:, None] * dt) if X else None
    ens = lp.var((Z, T), 0, D.T, cost=op["voll_usd_mwh"] * dt)
    ovg = lp.var((Z, T), 0, np.inf, cost=OVERGEN_PENALTY * dt)

    bal = lp.rows((Z, T), "=", D.T)
    lp.terms(bal[st.gz], g, 1)
    if S:
        lp.terms(bal[st.sz], dis, 1); lp.terms(bal[st.sz], ch, -1)
    keep = 1 - st.lines.loss_frac.to_numpy(float)[:, None]
    lp.terms(bal[st.lf], fp, -1); lp.terms(bal[st.lt], fp, keep)
    lp.terms(bal[st.lt], fn, -1); lp.terms(bal[st.lf], fn, keep)
    if X:
        lp.terms(bal[st.xz], x, -1)
    lp.terms(bal, ens, 1); lp.terms(bal, ovg, -1)

    sol = lp.solve(mip=mip)
    out = {"gen": sol.val(g).T, "price": sol.mu(bal).T / dt, "flow": (sol.val(fp) - sol.val(fn)).T,
           "ens": sol.val(ens).T, "overgen": sol.val(ovg).T, "reserve_short": sol.val(short),
           "reserve_price": sol.mu(rreq) / dt, "reserve": sol.val(rv).T, "obj": sol.obj,
           "line_cong": (np.abs(sol.val(fp) - sol.val(fn)) >= capl * 0.999).T}
    out["export"] = sol.val(x).T if X else np.zeros((T, 0))
    if nth:
        out["on"], out["su"] = sol.val(on).T, sol.val(su).T
    else:
        out["on"] = out["su"] = np.zeros((T, 0))
    if R:
        out["s"], out["spill"], out["wv"] = sol.val(s).T, sol.val(spill).T, -sol.mu(rbal).T
    else:
        out["s"] = out["spill"] = out["wv"] = np.zeros((T, 0))
    if S:
        out["ch"], out["dis"], out["soc"] = sol.val(ch).T, sol.val(dis).T, sol.val(soc).T
    else:
        out["ch"] = out["dis"] = out["soc"] = np.zeros((T, 0))
    return out


@dataclass
class DispatchResult:
    year: int
    index: pd.DatetimeIndex
    dt: float
    gens: pd.DataFrame
    storage: pd.DataFrame
    zones: list
    lines: pd.DataFrame
    exports: pd.DataFrame
    reservoirs: pd.DataFrame
    data: dict
    stage1: dict = field(default_factory=dict)
    runtime_s: float = 0.0

    def df(self, key, cols=None):
        cols = cols if cols is not None else {
            "gen": self.gens.id, "avail": self.gens.id, "price": self.zones, "ens": self.zones,
            "overgen": self.zones, "demand": self.zones, "flow": self.lines.line, "line_cong": self.lines.line,
            "export": self.exports.index, "s": self.reservoirs.reservoir, "spill": self.reservoirs.reservoir,
            "wv": self.reservoirs.reservoir, "ch": self.storage.id, "dis": self.storage.id, "soc": self.storage.id,
            "on": self.gens.id.iloc[self.data["th"]], "su": self.gens.id.iloc[self.data["th"]],
        }.get(key)
        v = self.data[key]
        if v.ndim == 1:
            return pd.Series(v, index=self.index, name=key)
        return pd.DataFrame(v, index=self.index, columns=list(cols))

    def gen_by_tech(self) -> pd.DataFrame:
        gdf = self.df("gen")
        out = gdf.T.groupby(self.gens.set_index("id").tech).sum().T
        if len(self.storage):
            out["bess"] = self.data["dis"].sum(1) - self.data["ch"].sum(1)
        return out


def run_dispatch(inp: Inputs, year: int | None = None, progress=None, ts: dict | None = None) -> DispatchResult:
    t0 = time.time()
    year = inp.year if year is None else year
    ts = ts or build_timeseries(inp, year)
    op = inp.cfg["operation"]
    res = int(op.get("resolution_hours", 1))
    if res not in VALID_RES:
        raise ValueError(f"resolution_hours must be one of {VALID_RES}")
    mip = str(op.get("unit_commitment", "lp")).lower() == "milp"
    gens = plant_economics(inp, ts["gens"])
    storage = inp.active_storage(year)
    exports = pd.DataFrame(inp.cfg.get("exports") or {}).T
    if len(exports):
        exports[["max_mw", "price_usd_mwh"]] = exports[["max_mw", "price_usd_mwh"]].astype(float)
    st = Static(gens, storage, inp.reservoirs, inp.lines, inp.zones.zone.tolist(), exports, op)

    D = ts["demand"][st.zones].to_numpy()
    A = ts["avail"][gens.id].to_numpy()
    I = ts["inflow"][st.reservoirs.reservoir].to_numpy()
    N = D.shape[0]
    s0 = np.clip(inp.reservoirs.initial_frac.to_numpy() * st.smax, st.smin, st.smax)
    soc0 = (storage.initial_soc_frac.to_numpy(float) * st.E) if len(storage) else np.zeros(0)
    s_end = s0.copy() if inp.cfg["hydrology"].get("end_storage_equals_start", True) else st.smin.copy()

    # ---------------- Stage 1: mid-term hydro-thermal coordination
    b1 = max(6, res)
    if progress:
        progress(0.02, "Stage 1: annual hydro-thermal coordination")
    st1 = solve_window(st, _agg(D, b1), _agg(A, b1), _agg(I, b1), b1,
                       {"s": s0, "soc": soc0, "on": np.zeros(len(st.th))},
                       {"mode": "target", "s": s_end}, mip=False)
    wv1 = np.maximum(st1["wv"], op.get("water_value_floor", 0.0))

    # ---------------- Stage 2: rolling short-term dispatch
    W, Lk = int(op.get("rolling_window_h", 168)), int(op.get("lookahead_h", 48))
    W, Lk = max(res, W // res * res), Lk // res * res
    state = {"s": s0, "soc": soc0, "on": np.zeros(len(st.th))}
    chunks, h0, starts = [], 0, list(range(0, N, W))
    for k, h0 in enumerate(starts):
        h1 = min(h0 + W + Lk, N)
        keep_h = min(W, N - h0)
        last = h1 >= N
        end = ({"mode": "target", "s": s_end} if last else
               {"mode": "value", "wv": wv1[min((h1 - 1) // b1, len(wv1) - 1)]})
        sl = slice(h0, h1)
        o = solve_window(st, _agg(D[sl], res), _agg(A[sl], res), _agg(I[sl], res), res, state, end, mip=mip)
        nk = max(1, keep_h // res)
        o = {kk: (v[:nk] if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] >= nk and kk != "obj" else v)
             for kk, v in o.items()}
        chunks.append(o)
        state = {"s": o["s"][-1] if len(st.reservoirs) else s0, "soc": o["soc"][-1] if len(storage) else soc0,
                 "on": np.round(o["on"][-1]) if mip else o["on"][-1] if len(st.th) else np.zeros(0)}
        if progress:
            progress(0.05 + 0.95 * (k + 1) / len(starts), f"Stage 2: week {k + 1}/{len(starts)}")
    data = {kk: np.concatenate([c[kk] for c in chunks]) for kk in chunks[0] if kk != "obj"}
    data["obj"] = sum(c["obj"] for c in chunks)
    T2 = data["gen"].shape[0]
    data["demand"] = _agg(D, res)[:T2]
    data["avail"] = _agg(A, res)[:T2]
    data["th"] = st.th
    idx = ts["index"][::res][:T2]
    return DispatchResult(year, idx, float(res), gens, storage, st.zones, inp.lines, exports, inp.reservoirs,
                          data, stage1={"s": st1["s"], "wv": wv1, "dt": b1}, runtime_s=time.time() - t0)
