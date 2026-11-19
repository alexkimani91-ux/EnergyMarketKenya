"""Plotly figure builders shared by the HTML report and the Streamlit dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TECH_COLORS = {
    "geothermal": "#B5523B", "hydro_res": "#2E6FAC", "hydro_ror": "#6FA8DC", "hydro_new": "#1B4F72",
    "wind": "#3E9C97", "solar": "#F2B134", "import": "#8E7CC3", "biomass": "#7A9A3B",
    "thermal": "#5D5D5D", "nuclear": "#A04E9E", "bess": "#E07B39", "unserved": "#D62728",
}
TECH_ORDER = ["geothermal", "nuclear", "hydro_ror", "import", "wind", "solar", "biomass", "hydro_res",
              "hydro_new", "bess", "thermal", "unserved"]
LAYOUT = dict(template="plotly_white", margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.15),
              font=dict(family="Segoe UI, Arial", size=12))


def _order(cols):
    return [t for t in TECH_ORDER if t in cols] + [c for c in cols if c not in TECH_ORDER]


def dispatch_stack(r, start=None, days=7):
    gt = r.gen_by_tech()
    gt["unserved"] = r.data["ens"].sum(1)
    idx = r.index
    if start is None:   # default: the week containing the system peak
        start = idx[int(np.argmax(r.data["demand"].sum(1)))].normalize() - pd.Timedelta(days=3)
    m = (idx >= pd.Timestamp(start)) & (idx < pd.Timestamp(start) + pd.Timedelta(days=days))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for t in _order(gt.columns):
        y = gt.loc[m, t].clip(lower=0)
        if y.abs().sum() < 1e-6:
            continue
        fig.add_trace(go.Scatter(x=idx[m], y=y, name=t, stackgroup="g", mode="none",
                                 fillcolor=TECH_COLORS.get(t, "#999")), secondary_y=False)
    if "bess" in gt:
        ch = gt.loc[m, "bess"].clip(upper=0)
        fig.add_trace(go.Scatter(x=idx[m], y=ch, name="bess charging", fill="tozeroy", mode="none",
                                 fillcolor="rgba(224,123,57,0.4)"), secondary_y=False)
    dem = r.data["demand"].sum(1)[m] + (r.data["export"].sum(1)[m] if r.data["export"].size else 0)
    fig.add_trace(go.Scatter(x=idx[m], y=dem, name="demand + exports", line=dict(color="black", width=2)), secondary_y=False)
    p = pd.DataFrame(r.data["price"], index=idx, columns=r.zones)
    fig.add_trace(go.Scatter(x=idx[m], y=p.loc[m, "NAIROBI"] if "NAIROBI" in p else p.loc[m].mean(axis=1),
                             name="Nairobi LMP", line=dict(color="#D62728", width=1.5, dash="dot")), secondary_y=True)
    fig.update_yaxes(title_text="MW", secondary_y=False)
    fig.update_yaxes(title_text="USD/MWh", secondary_y=True, showgrid=False)
    fig.update_layout(title=f"Hourly dispatch & price ({pd.Timestamp(start):%d %b %Y}, {days} days)", height=480, **LAYOUT)
    return fig


def monthly_energy(r):
    gt = r.gen_by_tech() * r.dt / 1000
    mo = gt.groupby(gt.index.month).sum()
    fig = go.Figure()
    for t in _order(mo.columns):
        fig.add_bar(x=mo.index, y=mo[t], name=t, marker_color=TECH_COLORS.get(t, "#999"))
    fig.update_layout(barmode="relative", title="Monthly generation by technology (GWh)", xaxis_title="month",
                      height=420, **LAYOUT)
    return fig


def price_heatmap(r, zone="NAIROBI"):
    p = pd.Series(r.data["price"][:, r.zones.index(zone)], index=r.index)
    h = p.groupby([p.index.dayofyear, p.index.hour]).mean().unstack()
    fig = go.Figure(go.Heatmap(z=h.T.values, x=h.index, y=h.columns, colorscale="Turbo",
                               zmin=0, zmax=float(np.percentile(p, 99)), colorbar=dict(title="USD/MWh")))
    fig.update_layout(title=f"Marginal price heat-map - {zone} (hour of day x day of year)", xaxis_title="day of year",
                      yaxis_title="hour", height=420, **LAYOUT)
    return fig


def price_duration(r):
    fig = go.Figure()
    for i, z in enumerate(r.zones):
        p = np.sort(r.data["price"][:, i])[::-1]
        fig.add_scatter(x=np.linspace(0, 100, len(p)), y=p, name=z, mode="lines")
    fig.update_layout(title="Price duration curves by zone", xaxis_title="% of time", yaxis_title="USD/MWh",
                      yaxis_type="log", height=420, **LAYOUT)
    return fig


def reservoir_levels(r):
    fig = go.Figure()
    for i, res in enumerate(r.reservoirs.reservoir):
        fig.add_scatter(x=r.index, y=r.data["s"][:, i] / 1000, name=f"{res} level", mode="lines")
        fig.add_scatter(x=r.index, y=np.full(len(r.index), r.reservoirs.max_storage_gwh.iloc[i]),
                        name=f"{res} max", line=dict(dash="dot", width=1))
    fig.update_layout(title="Reservoir storage (GWh energy-equivalent)", yaxis_title="GWh", height=400, **LAYOUT)
    return fig


def line_utilisation(r):
    f = pd.DataFrame(r.data["flow"], columns=r.lines.line)
    util = f.abs().div(r.lines.capacity_mw.values) * 100
    fig = go.Figure()
    for c in util:
        fig.add_box(y=util[c], name=c, boxpoints=False)
    fig.update_layout(title="Transmission corridor utilisation (% of capacity)", yaxis_title="%", height=400,
                      showlegend=False, **LAYOUT)
    return fig


def plant_economics_chart(fin):
    p = fin["plants"].sort_values("unit_cost_usd_mwh")
    p = p[p.generation_gwh > 0.5]
    fig = go.Figure()
    fig.add_bar(x=p.index, y=p.unit_cost_usd_mwh.clip(upper=600), name="PPA cost per MWh (capped 600)",
                marker_color=[TECH_COLORS.get(t, "#999") for t in p.tech])
    fig.add_scatter(x=p.index, y=p.capture_price_usd_mwh, name="market capture price", mode="markers",
                    marker=dict(color="black", size=8, symbol="diamond"))
    fig.update_layout(title="Plant cost to KPLC vs value at marginal price (USD/MWh)", height=450, **LAYOUT)
    return fig


def cost_breakdown(fin):
    p = fin["plants"]
    cols = {"capacity_payment_musd": "Capacity", "energy_payment_musd": "Energy", "deemed_energy_musd": "Deemed energy",
            "fuel_cost_musd": "Fuel", "startup_musd": "Start-up", "var_om_musd": "Var O&M", "carbon_musd": "Carbon"}
    g = p.groupby("tech")[list(cols)].sum().rename(columns=cols)
    fig = go.Figure()
    for c in g.columns:
        if g[c].abs().sum() > 1e-6:
            fig.add_bar(x=g.index, y=g[c], name=c)
    fig.update_layout(barmode="stack", title="Power purchase cost by technology & component (MUSD)", height=420, **LAYOUT)
    return fig


def tariff_waterfall(fin):
    t = fin["tariff"]
    fig = go.Figure(go.Waterfall(x=list(t.index) + ["Total"], y=list(t.values) + [0],
                                 measure=["relative"] * len(t) + ["total"],
                                 text=[f"{v:.2f}" for v in t.values] + [f"{t.sum():.2f}"], textposition="outside"))
    fig.update_layout(title="Average retail tariff build-up (KES/kWh)", height=450, showlegend=False, **LAYOUT)
    return fig


# ------------------------------------------------------------------ expansion
def expansion_capacity(x):
    c = x["capacity"]
    fig = go.Figure()
    for t in _order(c.index):
        fig.add_bar(x=c.columns, y=c.loc[t], name=t, marker_color=TECH_COLORS.get(t, "#999"))
    fig.add_scatter(x=x["summary"].index, y=x["summary"].peak_mw, name="peak demand", line=dict(color="black", width=3))
    fig.update_layout(barmode="stack", title="Installed capacity by technology (MW)", height=450, **LAYOUT)
    return fig


def expansion_generation(x):
    g = x["generation"]
    fig = go.Figure()
    for t in _order(g.index):
        fig.add_bar(x=g.columns, y=g.loc[t], name=t, marker_color=TECH_COLORS.get(t, "#999"))
    fig.update_layout(barmode="relative", title="Generation by technology (GWh)", height=450, **LAYOUT)
    return fig


def expansion_builds(x):
    b = x["builds"]
    b = b[b.sum(axis=1) > 0.5]
    fig = go.Figure()
    for cid in b.index:
        tech = x["candidates"].set_index("id").tech[cid]
        fig.add_bar(x=b.columns, y=b.loc[cid], name=cid, marker_color=TECH_COLORS.get(tech, "#999"),
                    marker_pattern_shape="/" if cid.startswith(("CCGT", "ETH")) else "")
    fig.update_layout(barmode="stack", title="New capacity commissioned per period (MW)", height=420, **LAYOUT)
    return fig


def expansion_prices(x):
    s = x["summary"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=s.index, y=s.co2_mt, name="CO2 (Mt)", marker_color="#BBBBBB", secondary_y=True)
    fig.add_scatter(x=s.index, y=s.avg_marginal_price_usd_mwh, name="energy price (USD/MWh)", secondary_y=False)
    fig.add_scatter(x=s.index, y=s.capacity_price_usd_kw_yr / 10, name="capacity price (USD/kW-yr / 10)", secondary_y=False)
    fig.update_layout(title="Long-run marginal prices & emissions", height=420, **LAYOUT)
    return fig


# ---------------------------------------------------------------- monte carlo
def mc_hist(mc, col="Avg retail tariff incl. taxes (KES/kWh)"):
    fig = go.Figure()
    for h, c in (("dry", "#B5523B"), ("normal", "#2E6FAC"), ("wet", "#3E9C97")):
        d = mc[mc.hydrology == h][col]
        if len(d):
            fig.add_histogram(x=d, name=h, marker_color=c, opacity=0.75, nbinsx=15)
    fig.update_layout(barmode="overlay", title=f"Distribution: {col}", height=400, **LAYOUT)
    return fig


def mc_scatter(mc, x="fuel_mult", y="Avg power purchase cost (US c/kWh sent-out)"):
    fig = go.Figure()
    for h, c in (("dry", "#B5523B"), ("normal", "#2E6FAC"), ("wet", "#3E9C97")):
        d = mc[mc.hydrology == h]
        fig.add_scatter(x=d[x], y=d[y], mode="markers", name=h, marker=dict(color=c, size=9))
    fig.update_layout(title=f"{y} vs {x}", xaxis_title=x, yaxis_title=y, height=400, **LAYOUT)
    return fig
