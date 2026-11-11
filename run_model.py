"""Command-line entry point for the Kenya Power Market Model.

Examples
--------
  python run_model.py dispatch                         # hourly dispatch of model_year, base scenario
  python run_model.py dispatch --hydrology dry --year 2027 --hfo-price 15
  python run_model.py dispatch --milp                  # integer unit commitment
  python run_model.py expansion                        # least-cost capacity expansion plan
  python run_model.py montecarlo --n 30                # stochastic risk analysis
  python run_model.py all                              # everything, one HTML report
  python run_model.py dispatch --scenario my_case.yaml # any YAML in inputs/
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

from kpm.data import load_inputs
from kpm.dispatch import run_dispatch
from kpm.expansion import run_expansion
from kpm.finance import settle
from kpm.montecarlo import run_monte_carlo, summarise
from kpm.report import export_dispatch, export_expansion, html_report

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def progress(frac, msg):
    sys.stdout.write(f"\r  [{frac * 100:5.1f}%] {msg:<60}")
    sys.stdout.flush()
    if frac >= 1:
        sys.stdout.write("\n")


def main():
    ap = argparse.ArgumentParser(description="Kenya Power Market Model")
    ap.add_argument("mode", choices=["dispatch", "expansion", "montecarlo", "all"])
    ap.add_argument("--scenario", default="scenario_base.yaml")
    ap.add_argument("--inputs", default=None, help="input folder (default ./inputs)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--year", type=int)
    ap.add_argument("--hydrology", choices=["dry", "normal", "wet"])
    ap.add_argument("--resolution", type=int, help="dispatch time step in hours (1,2,3,4,6...)")
    ap.add_argument("--milp", action="store_true", help="integer unit commitment for thermal plant")
    ap.add_argument("--hfo-price", type=float, help="HFO price USD/GJ")
    ap.add_argument("--carbon-price", type=float)
    ap.add_argument("--fx", type=float, help="KES per USD")
    ap.add_argument("--n", type=int, default=20, help="Monte Carlo draws")
    a = ap.parse_args()

    ov = {}
    if a.year:
        ov["model_year"] = a.year
    if a.hydrology:
        ov.setdefault("hydrology", {})["scenario"] = a.hydrology
    if a.resolution:
        ov.setdefault("operation", {})["resolution_hours"] = a.resolution
    if a.milp:
        ov.setdefault("operation", {})["unit_commitment"] = "milp"
    if a.hfo_price is not None:
        ov.setdefault("fuels", {})["HFO"] = {"price": a.hfo_price}
    if a.carbon_price is not None:
        ov["carbon_price_usd_t"] = a.carbon_price
    if a.fx:
        ov.setdefault("finance", {})["fx_kes_per_usd"] = a.fx
    inp = load_inputs(a.inputs, a.scenario, ov)
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(inp.cfg.get("scenario_name", "scenario"))).strip("_")
    out = Path(a.out) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    r = fin = x = mc = mcs = None

    if a.mode in ("dispatch", "all"):
        print(f"Dispatch {inp.year} | hydrology={inp.cfg['hydrology']['scenario']} | "
              f"UC={inp.cfg['operation']['unit_commitment']} | step={inp.cfg['operation']['resolution_hours']}h")
        r = run_dispatch(inp, progress=progress)
        fin = settle(inp, r)
        export_dispatch(r, fin, out)
        print(fin["kpis"].round(2).to_string())
        print("\nTariff build-up (KES/kWh)\n" + fin["tariff"].round(2).to_string())
    if a.mode in ("expansion", "all"):
        print("\nCapacity expansion ...")
        x = run_expansion(inp, progress=progress)
        export_expansion(x, out)
        print(f"NPV system cost: {x['npv_cost_musd']:,.0f} MUSD   (LP {x['lp_size'][0]:,} vars, {x['runtime_s']:.1f}s)")
        print(x["builds"][x["builds"].sum(axis=1) > 0.5].round(0).to_string())
        print(x["summary"].round(1).to_string())
    if a.mode in ("montecarlo", "all"):
        print(f"\nMonte Carlo ({a.n} draws) ...")
        mc = run_monte_carlo(inp, n=a.n, progress=progress)
        mcs = summarise(mc)
        out.mkdir(parents=True, exist_ok=True)
        mc.to_csv(out / "montecarlo_draws.csv", index=False)
        mcs.to_csv(out / "montecarlo_summary.csv")
        print(mcs.round(2).to_string())
    rep = html_report(out / "report.html", f"Kenya Power Market Model - {inp.cfg.get('scenario_name')}",
                      r, fin, x, mc, mcs)
    print(f"\nResults written to {out.resolve()}\nReport: {rep.resolve()}")


if __name__ == "__main__":
    main()
