"""Kenya Power Market Model - interactive web dashboard.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from kpm import charts
from kpm.data import INPUT_DIR, load_inputs, validate
from kpm.dispatch import run_dispatch
from kpm.expansion import run_expansion
from kpm.finance import settle
from kpm.montecarlo import run_monte_carlo, summarise

st.set_page_config(page_title="Kenya Power Market Model", page_icon="⚡", layout="wide")
ss = st.session_state
ss.setdefault("runs", {})

# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚡ Kenya Power Market Model")
scen_files = sorted(p.name for p in INPUT_DIR.glob("*.yaml"))
scen = st.sidebar.selectbox("Scenario file", scen_files, index=scen_files.index("scenario_base.yaml") if "scenario_base.yaml" in scen_files else 0)
base = load_inputs(scenario=scen)
c = base.cfg

if ss.get("_scen") != scen:          # (re)load editable tables when scenario changes
    ss["_scen"] = scen
    for k in ("gens", "storage", "lines", "zones", "reservoirs", "inflows", "ror_cf", "candidates"):
        ss[f"base_{k}"] = getattr(base, k).copy()
        ss[f"tbl_{k}"] = getattr(base, k).copy()

sb = st.sidebar
sb.subheader("Market & system")
year = sb.number_input("Model year", 2024, 2045, int(c["model_year"]))
hyd = sb.selectbox("Hydrology", ["dry", "normal", "wet"], index=["dry", "normal", "wet"].index(c["hydrology"]["scenario"]))
infl = sb.slider("Inflow multiplier", 0.5, 1.5, float(c["hydrology"]["inflow_multiplier"]), 0.05)
with sb.expander("Demand", expanded=False):
    e0 = st.number_input("Base-year energy (GWh)", 5000.0, 60000.0, float(c["demand"]["base_energy_gwh"]), 100.0)
    p0 = st.number_input("Base-year peak (MW)", 1000.0, 10000.0, float(c["demand"]["base_peak_mw"]), 10.0)
    eg = st.slider("Energy growth (%/yr)", 0.0, 12.0, 100 * c["demand"]["energy_growth"], 0.1) / 100
    pg = st.slider("Peak growth (%/yr)", 0.0, 12.0, 100 * c["demand"]["peak_growth"], 0.1) / 100
    st.caption("Additional policy loads (GWh/yr)")
    xl = {}
    for k, v in (c["demand"].get("extra_loads") or {}).items():
        xl[k] = {**v, "annual_gwh": st.number_input(k.replace("_", " "), 0.0, 20000.0, float(v["annual_gwh"]), 10.0, key=f"xl_{k}")}
with sb.expander("Fuels, carbon & FX", expanded=False):
    fuels = {f: {"price": st.number_input(f"{f} price (USD/GJ)", 0.0, 100.0, float(v["price"]), 0.1, key=f"fp_{f}")}
             for f, v in c["fuels"].items() if f != "NONE"}
    carbon = st.number_input("Carbon price (USD/tCO2)", 0.0, 300.0, float(c.get("carbon_price_usd_t", 0)), 5.0)
    fx = st.number_input("FX (KES per USD)", 50.0, 300.0, float(c["finance"]["fx_kes_per_usd"]), 1.0)
with sb.expander("Operation & solver", expanded=False):
    voll = st.number_input("VoLL (USD/MWh)", 100.0, 20000.0, float(c["operation"]["voll_usd_mwh"]), 100.0)
    rfrac = st.slider("Spinning reserve (% of load)", 0.0, 20.0, 100 * c["operation"]["reserve_frac_of_load"], 0.5) / 100
    cont = st.number_input("Largest contingency (MW)", 0.0, 1000.0, float(c["operation"]["largest_contingency_mw"]), 10.0)
    res = st.selectbox("Time resolution (h)", [1, 2, 3, 4, 6], index=[1, 2, 3, 4, 6].index(int(c["operation"]["resolution_hours"])))
    uc = st.radio("Unit commitment", ["lp", "milp"], index=["lp", "milp"].index(c["operation"]["unit_commitment"]),
                  help="lp = relaxed (fast); milp = integer on/off decisions for thermal units (slower, exact)")
with sb.expander("Finance & tariff", expanded=False):
    loss = st.slider("T&D losses (% of sent-out)", 5.0, 35.0, 100 * c["finance"]["td_loss_frac"], 0.5) / 100
    trans = st.number_input("Transmission cost (USD/MWh)", 0.0, 50.0, float(c["finance"]["transmission_cost_usd_mwh"]), 0.5)
    dist = st.number_input("Distribution & retail (KES/kWh)", 0.0, 20.0, float(c["finance"]["distribution_cost_kes_kwh"]), 0.1)

overrides = {
    "model_year": int(year),
    "hydrology": {"scenario": hyd, "inflow_multiplier": infl},
    "demand": {"base_energy_gwh": e0, "base_peak_mw": p0, "energy_growth": eg, "peak_growth": pg, "extra_loads": xl},
    "fuels": fuels, "carbon_price_usd_t": carbon,
    "operation": {"voll_usd_mwh": voll, "reserve_frac_of_load": rfrac, "largest_contingency_mw": cont,
                  "resolution_hours": int(res), "unit_commitment": uc},
    "finance": {"fx_kes_per_usd": fx, "td_loss_frac": loss, "transmission_cost_usd_mwh": trans,
                "distribution_cost_kes_kwh": dist},
}


def current_inputs(extra=None):
    inp = load_inputs(scenario=scen, overrides=overrides)
    for k in ("gens", "storage", "lines", "zones", "reservoirs", "inflows", "ror_cf", "candidates"):
        df = ss[f"tbl_{k}"].copy()
        for col in ("profile", "reservoir"):
            if col in df:
                df[col] = df[col].fillna("").astype(str)
        setattr(inp, k, df.reset_index(drop=True))
    if extra:
        from kpm.data import deep_update
        deep_update(inp.cfg, extra)
    validate(inp)
    return inp


def bar(label):
    pb = st.progress(0.0, text=label)
    return lambda f, m: pb.progress(min(max(f, 0.0), 1.0), text=m)


def to_excel(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        for name, df in sheets.items():
            (df.to_frame() if isinstance(df, pd.Series) else df).to_excel(xw, sheet_name=name[:31])
    return buf.getvalue()


tabs = st.tabs(["📋 Inputs", "⚙️ Dispatch & prices", "💰 Finance & tariff", "🏗️ Capacity expansion",
                "🎲 Risk (Monte Carlo)", "📊 Scenario comparison", "ℹ️ Methodology"])

# ------------------------------------------------------------------ inputs
with tabs[0]:
    st.markdown("Edit any input table below - changes apply to the next run. Values are indicative public-domain "
                "estimates; replace with EPRA / KPLC / KenGen / KETRACO data.")
    names = {"gens": "Generation fleet (PPAs, technical data)", "storage": "Battery storage", "lines": "Transmission corridors",
             "zones": "Demand zones", "reservoirs": "Hydro reservoirs", "inflows": "Monthly reservoir inflows (GWh)",
             "ror_cf": "Run-of-river monthly capacity factors", "candidates": "Expansion candidates"}
    for k, label in names.items():
        with st.expander(label, expanded=(k == "gens")):
            ss[f"tbl_{k}"] = st.data_editor(ss[f"base_{k}"], num_rows="dynamic", use_container_width=True,
                                            key=f"ed_{scen}_{k}")
    try:
        inp_preview = current_inputs()
        g = inp_preview.active_gens()
        cap = g.groupby("tech").capacity_mw.sum().sort_values(ascending=False)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Installed capacity {year}", f"{g.capacity_mw.sum():,.0f} MW")
        from kpm.profiles import system_targets
        e, p = system_targets(inp_preview, int(year))
        m2.metric("Projected energy (organic)", f"{e:,.0f} GWh")
        m3.metric("Projected peak (organic)", f"{p:,.0f} MW")
        st.bar_chart(cap)
    except Exception as ex:  # noqa: BLE001
        st.error(f"Input problem: {ex}")

# ------------------------------------------------------------------ dispatch
with tabs[1]:
    col1, col2 = st.columns([1, 3])
    run_name = col1.text_input("Run name", f"{hyd}-{year}-{uc}")
    if col1.button("▶ Run dispatch", type="primary"):
        try:
            inp = current_inputs()
            r = run_dispatch(inp, progress=bar("Solving dispatch"))
            fin = settle(inp, r)
            ss["dispatch"], ss["fin"] = r, fin
            ss["runs"][run_name] = fin["kpis"]
            st.success(f"Solved in {r.runtime_s:.1f}s")
        except Exception as ex:  # noqa: BLE001
            st.exception(ex)
    if "dispatch" in ss:
        r, fin = ss["dispatch"], ss["fin"]
        k = fin["kpis"]
        a, b_, c_, d_, e_ = st.columns(5)
        a.metric("Avg LMP (USD/MWh)", f"{k['Load-weighted LMP (USD/MWh)']:.1f}")
        b_.metric("Purchase cost (USc/kWh)", f"{k['Avg power purchase cost (US c/kWh sent-out)']:.2f}")
        c_.metric("Retail tariff (KES/kWh)", f"{k['Avg retail tariff incl. taxes (KES/kWh)']:.2f}")
        d_.metric("Unserved energy (MWh)", f"{k['Unserved energy (MWh)']:,.0f}")
        e_.metric("Curtailed TOP (GWh)", f"{k['Curtailed take-or-pay energy (GWh)']:,.0f}")
        d0 = st.date_input("Window start", value=None, help="Leave empty to show the week of system peak")
        days = st.slider("Days shown", 1, 31, 7)
        st.plotly_chart(charts.dispatch_stack(r, d0, days), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.monthly_energy(r), use_container_width=True)
        zone = c2.selectbox("Zone for price heat-map", r.zones)
        c2.plotly_chart(charts.price_heatmap(r, zone), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.price_duration(r), use_container_width=True)
        c2.plotly_chart(charts.reservoir_levels(r), use_container_width=True)
        st.plotly_chart(charts.line_utilisation(r), use_container_width=True)
        st.download_button("⬇ Hourly results (Excel)", to_excel({
            "generation_MW": r.df("gen"), "by_tech_MW": r.gen_by_tech(), "LMP": r.df("price"),
            "flows_MW": r.df("flow"), "demand_MW": r.df("demand"), "reservoir_MWh": r.df("s")}),
            "kenya_dispatch_hourly.xlsx")

# ------------------------------------------------------------------ finance
with tabs[2]:
    if "fin" not in ss:
        st.info("Run the dispatch first.")
    else:
        fin = ss["fin"]
        c1, c2 = st.columns([1, 1])
        c1.subheader("Key indicators")
        c1.dataframe(fin["kpis"].to_frame("value").style.format("{:,.2f}"), use_container_width=True, height=720)
        c2.plotly_chart(charts.tariff_waterfall(fin), use_container_width=True)
        c2.plotly_chart(charts.cost_breakdown(fin), use_container_width=True)
        st.plotly_chart(charts.plant_economics_chart(fin), use_container_width=True)
        st.subheader("Plant-by-plant PPA settlement")
        st.dataframe(fin["plants"].round(2), use_container_width=True)
        if len(fin["storage"]):
            st.subheader("Storage")
            st.dataframe(fin["storage"].round(2), use_container_width=True)
        st.download_button("⬇ Settlement & tariff (Excel)", to_excel({"KPIs": fin["kpis"], "Tariff": fin["tariff"],
                           "Plants": fin["plants"], "Storage": fin["storage"]}), "kenya_settlement.xlsx")

# ------------------------------------------------------------------ expansion
with tabs[3]:
    ex = c["expansion"]
    cc = st.columns(6)
    y0 = cc[0].number_input("Start year", 2024, 2040, int(ex["start_year"]))
    y1 = cc[1].number_input("End year", 2026, 2060, int(ex["end_year"]))
    stp = cc[2].selectbox("Year step", [1, 2, 3, 5], index=[1, 2, 3, 5].index(int(ex["year_step"])))
    rm = cc[3].slider("Reserve margin %", 0, 40, int(100 * ex["reserve_margin"])) / 100
    clean = cc[4].slider("Min clean share %", 0, 100, int(100 * (ex.get("min_clean_share") or 0))) / 100
    cty = cc[5].number_input("Clean target from", 2026, 2060, int(ex.get("clean_target_year", 2030)))
    cc = st.columns(4)
    disc = cc[0].slider("Discount rate %", 3.0, 15.0, 100 * float(ex["discount_rate"]), 0.5) / 100
    co2cap = cc[1].number_input("CO2 cap (Mt/yr, 0 = none)", 0.0, 50.0, float(ex.get("co2_cap_mt") or 0), 0.1)
    xhyd = cc[2].selectbox("Planning hydrology", ["dry", "normal", "wet"], index=1)
    if cc[3].button("▶ Run expansion", type="primary"):
        try:
            inp = current_inputs({"expansion": {"start_year": int(y0), "end_year": int(y1), "year_step": int(stp),
                                                "reserve_margin": rm, "min_clean_share": clean, "clean_target_year": int(cty),
                                                "discount_rate": disc, "co2_cap_mt": co2cap or None, "hydrology": xhyd}})
            ss["exp"] = run_expansion(inp, progress=bar("Solving capacity expansion"))
        except Exception as ex_:  # noqa: BLE001
            st.exception(ex_)
    if "exp" in ss:
        x = ss["exp"]
        st.metric("NPV of system cost (new-build + operation)", f"{x['npv_cost_musd']:,.0f} MUSD",
                  help=f"LP size {x['lp_size'][0]:,} variables; solved in {x['runtime_s']:.1f}s")
        c1, c2 = st.columns(2)
        c1.plotly_chart(charts.expansion_capacity(x), use_container_width=True)
        c2.plotly_chart(charts.expansion_builds(x), use_container_width=True)
        c1.plotly_chart(charts.expansion_generation(x), use_container_width=True)
        c2.plotly_chart(charts.expansion_prices(x), use_container_width=True)
        st.dataframe(x["summary"].round(1), use_container_width=True)
        st.dataframe(x["builds"][x["builds"].sum(axis=1) > 0.5].round(0), use_container_width=True)
        st.download_button("⬇ Expansion plan (Excel)", to_excel({"Summary": x["summary"], "Builds": x["builds"],
                           "Capacity": x["capacity"], "Generation": x["generation"]}), "kenya_expansion.xlsx")

# ------------------------------------------------------------------ monte carlo
with tabs[4]:
    cc = st.columns(4)
    n = cc[0].number_input("Draws", 4, 500, 20)
    mres = cc[1].selectbox("Resolution (h)", [2, 3, 4, 6], index=1)
    st.caption("Uncertain: hydrology class (25% dry / 50% normal / 25% wet) and inflow volume, demand growth, "
               "fossil fuel prices, wind yield, geothermal steam availability and KES/USD.")
    if cc[2].button("▶ Run Monte Carlo", type="primary"):
        try:
            mc = run_monte_carlo(current_inputs(), n=int(n), resolution=int(mres), progress=bar("Monte Carlo"))
            ss["mc"] = mc
        except Exception as ex_:  # noqa: BLE001
            st.exception(ex_)
    if "mc" in ss:
        mc = ss["mc"]
        st.dataframe(summarise(mc).round(2), use_container_width=True)
        c1, c2 = st.columns(2)
        col = c1.selectbox("Distribution of", [k for k in mc.columns if mc[k].dtype != object and k != "draw"], index=0)
        c1.plotly_chart(charts.mc_hist(mc, col), use_container_width=True)
        xv = c2.selectbox("Driver", ["fuel_mult", "inflow_mult", "energy_growth", "wind_mult", "geo_avail_shift", "fx"])
        c2.plotly_chart(charts.mc_scatter(mc, xv, col), use_container_width=True)
        st.download_button("⬇ All draws (CSV)", mc.to_csv(index=False), "kenya_montecarlo.csv")

# ------------------------------------------------------------------ comparison
with tabs[5]:
    if ss["runs"]:
        comp = pd.DataFrame(ss["runs"])
        st.dataframe(comp.style.format("{:,.2f}"), use_container_width=True, height=720)
        if st.button("Clear comparison"):
            ss["runs"] = {}
    else:
        st.info("Every dispatch run is stored here (by run name) so scenarios can be compared side by side.")

# ------------------------------------------------------------------ methodology
with tabs[6]:
    readme = Path(__file__).with_name("README.md")
    st.markdown(readme.read_text(encoding="utf-8") if readme.exists() else "See README.md")
