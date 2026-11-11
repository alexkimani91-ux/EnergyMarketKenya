# Kenya Power Market Model (kpm)

An open, Python-based production-cost, market and planning model of the Kenyan power system.
It runs from the command line or as an interactive web dashboard. It covers four questions:

| Module | Question it answers | Method |
|---|---|---|
| **Dispatch & prices** (`kpm/dispatch.py`) | How is the system operated hour by hour? What are the marginal prices, curtailment, reserves and unserved energy? | 6-zone security-constrained economic dispatch / unit commitment. Two stages: an annual hydro-thermal model, then rolling weekly hourly LP/MILP solves |
| **Finance & tariff** (`kpm/finance.py`) | What does KPLC pay each plant? What is the cost of supply and the retail tariff? | PPA settlement (capacity, energy and deemed-energy charges, fuel pass-through), then a build-up of FCC, levies and VAT |
| **Capacity expansion** (`kpm/expansion.py`) | What should be built, and when, at least cost? | Multi-year LP with 12 representative days, a reserve margin, build limits and optional clean-energy or CO₂ constraints |
| **Risk** (`kpm/montecarlo.py`) | How exposed are costs and tariffs to rainfall, fuel prices, demand, wind and FX? | Monte Carlo runs of the full dispatch and settlement |

The solver is **HiGHS**, which ships inside SciPy, so you do not need a commercial solver or a separate install.

---

## 1. Quick start

```bash
pip install -r requirements.txt
```

**Web dashboard** (recommended):

```bash
streamlit run app.py
```

Then open http://localhost:8501. You can edit every input table in the browser, run any module, compare scenarios and download the results to Excel.

**Command line:**

```bash
python run_model.py dispatch                          # hourly dispatch of model_year
python run_model.py dispatch --hydrology dry --hfo-price 16 --fx 135
python run_model.py dispatch --milp                   # integer unit commitment
python run_model.py expansion                         # least-cost build plan 2026-2040
python run_model.py montecarlo --n 50                 # risk analysis
python run_model.py all                               # everything plus one HTML report
python run_model.py dispatch --scenario my_case.yaml  # any scenario file in inputs/
```

Each run writes a folder under `outputs/`. It contains hourly CSVs, Excel workbooks and an interactive `report.html`.

Typical run times on a laptop: hourly dispatch for a full year takes about 12 s with the LP and about 6–7 min with MILP unit commitment. Expansion for 2026–2040 takes about 5 s. Each Monte Carlo draw takes about 4 s at 3-hour resolution.

---

## 2. Inputs (`inputs/`)

Every input is a plain CSV or YAML file. Edit them in Excel or a text editor, or in the dashboard's **Inputs** tab.

| File | Contents |
|---|---|
| `scenario_base.yaml` | Global assumptions: model year, demand and growth, load shape, policy loads (e-mobility, e-cooking, green H₂, data centres), hydrology scenario, wind/solar resource, fuel prices and emission factors, carbon price, VoLL, reserves, solver settings, exports, FX, T&D losses, transmission/distribution costs, levies and VAT, expansion settings |
| `generators.csv` | Every plant: zone, technology, fuel, MW, availability, minimum stable level, heat rate, variable O&M, **PPA capacity charge (USD/kW-month)**, **energy charge (USD/MWh)**, take-or-pay flag, reserve capability, start-up cost, resource profile, reservoir link, CO₂ rate, commissioning and retirement (PPA expiry) year |
| `storage.csv` | Battery systems: zone, MW, hours, round-trip efficiency, capacity charge |
| `zones.csv` | The 6 demand zones and their share of demand |
| `lines.csv` | Transmission corridors between zones: MW limit and loss factor |
| `reservoirs.csv` | Tana cascade (Masinga…Kiambere) and Turkwel: storage limits in GWh-equivalent and initial level |
| `hydro_inflows.csv` | Monthly inflow energy (GWh) for **dry, normal and wet** years |
| `ror_cf.csv` | Monthly capacity factors for run-of-river hydro (Sondu Miriu, small hydros, new hydro) |
| `candidates.csv` | Expansion options: geothermal, wind, solar, 4-h BESS, High Grand Falls, Karura, LNG CCGT, gas peakers, nuclear, extra Ethiopia/EAPP imports. Each has capex, FOM, VOM, heat rate, life, firm credit, build limits and earliest year |
| `profiles/` *(optional)* | Measured data to replace the synthetic series. `demand.csv` holds one MW column per zone for the base year. `<profile>.csv` holds a column `cf` with 8760 rows, e.g. `turkana_wind.csv` |

To make a new scenario, create a small YAML file that starts with `inherits: scenario_base.yaml` and lists only the values that change. Two examples are included:
- `scenario_dry_highfuel.yaml`: drought, an oil price spike and a weaker shilling;
- `scenario_clean_2030.yaml`: 100 % clean electricity from 2030, a carbon price, and e-mobility, e-cooking and green H₂ loads.

Run a scenario with `--scenario <file>`, or select it in the dashboard.

> **Data note.** The values shipped with the model are indicative estimates taken from public sources (EPRA statistics, KPLC annual reports, the LCPDP, KenGen and IPP disclosures, press reports). They are calibrated to a plausible 2025/26 system: about 3.3 GW installed, a peak of about 2.5–2.6 GW and about 15 TWh sent out. **Replace them with official data before using the results for decisions.** Confidential PPA tariffs in particular are approximations.

---

## 3. What the dispatch model represents

**Spatial.** There are 6 zones: Nairobi, Coast, Mt Kenya/North-East, Western, North Rift and South Rift. They are connected by 7 transmission corridors, each with a thermal limit and a loss factor. The dual of each zone's energy balance gives that zone's **locational marginal price (LMP)**.

**Plant types.**
- **Geothermal** (Olkaria I–V, wellheads, Orpower 4, Menengai): must-run down to a minimum. Below that, steam is vented, which counts as curtailment.
- **Reservoir hydro**: the Tana cascade (Masinga, Kamburu, Gitaru, Kindaruma, Kiambere) and Turkwel. Water is tracked as a stored-energy balance with inflows, spill and minimum and maximum levels.
- **Run-of-river hydro**: Sondu Miriu/Sang'oro and small hydros, following monthly hydrology.
- **Wind**: LTWP, Kipeto and Ngong. Synthetic hourly profiles reproduce Turkana's night-time low-level jet and its seasonal pattern.
- **Solar**: Garissa, Malindi, Eldosol and others. Output follows an equatorial day shape, with more cloud in the long rains (MAM) and short rains (OND).
- **Thermal HFO/gasoil**: Kipevu III, Rabai, Thika, Triumph, Gulf and the Muhoroni GT. Each has a minimum stable level, start-up costs, and **relaxed (LP) or integer (MILP) unit commitment**.
- **Imports**: the Ethiopia PPA over the Sodo–Suswa HVDC is take-or-pay. Uganda supplies spot imports. Exports to Uganda and Tanzania are flexible and priced.
- **Batteries**: charge and discharge limits, state of charge, efficiency, and they can supply reserve.

**Constraints.** The model also enforces:
- a spinning reserve of max(largest contingency, x % of load);
- unserved energy, charged at VoLL;
- over-generation, as a slack variable.

**Two-stage hydro coordination.** Stage 1 solves the whole year in 6-hour blocks. It finds the reservoir trajectory and the **marginal water value**. Stage 2 then solves 168-hour windows with a 48-hour look-ahead at hourly resolution. Water left at the end of each window is valued at the Stage 1 water value. This is the standard medium-term/short-term split used by commercial tools.

**Take-or-pay economics.** Kenya's PPAs mostly pay for energy whether or not it is used. The model keeps two costs separate:
- the *dispatch* cost, i.e. the short-run marginal cost. Contracted take-or-pay energy is already paid for, so its dispatch cost is close to zero.
- the *settlement* cost, i.e. what KPLC actually pays, including **deemed energy** for curtailed output.

This separation shows the real Kenyan issues: night-time curtailment, very low off-peak marginal prices, and expensive idle thermal capacity.

## 4. Finance and tariff

For each plant the model computes:
- capacity payment, energy payment and deemed energy;
- fuel cost, start-up and variable O&M, plus carbon;
- CO₂;
- **market revenue at its own LMP**, which gives the capture price and the margin a plant would earn in a competitive market.

These roll up into the system power purchase cost, which is then divided by **units sold**, i.e. sent-out energy after T&D losses. Transmission, distribution, the REP levy, the EPRA and WRA levies and VAT are added on top. The result is the average **retail tariff (KES/kWh)** and the **Fuel Energy Cost charge**.

## 5. Capacity expansion

The model solves for the cheapest combination of new builds and operation over the horizon. It uses 12 representative days, one per month, each weighted by the number of days in the month. Existing reservoir hydro has a monthly storage balance. New hydro has monthly energy limits. Batteries cycle daily. Investment costs are converted to annual payments (capital recovery factor at the discount rate) and discounted.

The constraints are:
- reserve margin on firm capacity, with a technology capacity credit;
- build-rate limits, total resource potential and earliest commissioning year;
- an optional minimum clean-energy share;
- an optional CO₂ cap.

The model reports:
- the build plan;
- capacity and generation by technology and year;
- long-run energy prices and capacity prices (from the duals);
- emissions and NPV.

*Limitations:* the expansion model treats Kenya as a single node with no transmission limits, and it does not represent extreme peak days. The reserve margin is what ensures adequacy. Test any plan in the hourly dispatch model for the relevant years by adding the new plants to `generators.csv`.

## 6. Monte Carlo

Each draw samples:
- the hydrology class: 25 % dry, 50 % normal, 25 % wet;
- inflow volume (lognormal);
- demand growth;
- fossil fuel prices (lognormal);
- wind yield;
- geothermal steam availability;
- KES/USD.

The full dispatch and settlement then run for that draw. The model reports P5–P95 for cost of supply, tariff, FCC, thermal share, unserved energy, curtailment, deemed energy payments, CO₂ and LMP. The uncertainty settings are in `kpm/montecarlo.py` (`DEFAULT_UNCERTAINTY`).

## 7. Extending the model

- **New plant**: add a row to `generators.csv`. It is used automatically from its commissioning year.
- **New zone or line**: add rows to `zones.csv` and `lines.csv`. The demand shares must sum to 1.
- **Measured profiles**: put CSVs in `inputs/profiles/`.
- **New constraint**: the LP builder (`kpm/lp.py`) works on whole blocks of variables at once. See `solve_window()` in `dispatch.py` for examples.
- **Commercial solver**: replace `LPBuilder.solve()` with a Gurobi, CPLEX or HiGHS (`highspy`) call.

## 8. Project layout

```
kenya_power_model/
├── app.py              Streamlit web dashboard
├── run_model.py        command-line runner
├── requirements.txt
├── inputs/             all input data (CSV + YAML)
├── kpm/
│   ├── data.py         loading, validation, plant economics
│   ├── profiles.py     demand / wind / solar / hydro time-series
│   ├── lp.py           sparse LP/MILP builder (SciPy-HiGHS)
│   ├── dispatch.py     zonal SCED / unit commitment, two-stage hydro
│   ├── finance.py      PPA settlement and tariff
│   ├── expansion.py    least-cost capacity expansion
│   ├── montecarlo.py   stochastic risk analysis
│   ├── charts.py       Plotly figures
│   └── report.py       CSV / Excel / HTML export
└── outputs/            results of each run
```
