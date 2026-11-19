"""Input loading, validation and derived plant economics."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INPUT_DIR = Path(__file__).resolve().parent.parent / "inputs"

THERMAL_TECHS = {"thermal", "nuclear"}
FOSSIL_FUELS = {"HFO", "GASOIL", "LNG"}
# default firm-capacity credits used in planning reserve margin (existing fleet)
FIRM_CREDIT = {"geothermal": 0.90, "hydro_res": 0.80, "hydro_ror": 0.35, "hydro_new": 0.60, "wind": 0.10,
               "solar": 0.0, "thermal": 0.90, "nuclear": 0.90, "biomass": 0.50, "import": 0.70, "bess": 0.90}


@dataclass
class Inputs:
    cfg: dict
    zones: pd.DataFrame
    lines: pd.DataFrame
    gens: pd.DataFrame
    storage: pd.DataFrame
    reservoirs: pd.DataFrame
    inflows: pd.DataFrame
    ror_cf: pd.DataFrame
    candidates: pd.DataFrame
    input_dir: Path

    def copy(self) -> "Inputs":
        return copy.deepcopy(self)

    @property
    def year(self) -> int:
        return int(self.cfg["model_year"])

    def active_gens(self, year: int | None = None) -> pd.DataFrame:
        y = self.year if year is None else year
        g = self.gens
        return g[(g.commission_year <= y) & (g.retire_year > y) & (g.capacity_mw > 0)].reset_index(drop=True)

    def active_storage(self, year: int | None = None) -> pd.DataFrame:
        y = self.year if year is None else year
        s = self.storage
        return s[(s.commission_year <= y) & (s.retire_year > y) & (s.power_mw > 0)].reset_index(drop=True)


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _read_scenario(d: Path, name: str, depth: int = 0) -> dict:
    """Read a scenario YAML. A file may start from another via `inherits: other.yaml`."""
    with open(d / name, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent and depth < 5:
        return deep_update(_read_scenario(d, parent, depth + 1), cfg)
    return cfg


def load_inputs(input_dir: str | Path | None = None, scenario: str = "scenario_base.yaml",
                overrides: dict | None = None) -> Inputs:
    d = Path(input_dir) if input_dir else INPUT_DIR
    cfg = _read_scenario(d, scenario)
    deep_update(cfg, overrides or {})
    rd = lambda n: pd.read_csv(d / n, comment="#")
    inp = Inputs(cfg=cfg, zones=rd("zones.csv"), lines=rd("lines.csv"), gens=rd("generators.csv"),
                 storage=rd("storage.csv"), reservoirs=rd("reservoirs.csv"), inflows=rd("hydro_inflows.csv"),
                 ror_cf=rd("ror_cf.csv"), candidates=rd("candidates.csv"), input_dir=d)
    for col in ("profile", "reservoir"):
        inp.gens[col] = inp.gens[col].fillna("").astype(str)
    inp.candidates["profile"] = inp.candidates["profile"].fillna("").astype(str)
    validate(inp)
    return inp


def validate(inp: Inputs) -> None:
    zones = set(inp.zones.zone)
    errs = []
    for df, name in ((inp.gens, "generators"), (inp.storage, "storage")):
        bad = set(df.zone) - zones
        if bad:
            errs.append(f"{name}: unknown zones {bad}")
    bad = (set(inp.lines.from_zone) | set(inp.lines.to_zone)) - zones
    if bad:
        errs.append(f"lines: unknown zones {bad}")
    res = set(inp.reservoirs.reservoir)
    bad = {r for r in inp.gens.reservoir if r} - res
    if bad:
        errs.append(f"generators: unknown reservoirs {bad}")
    fuels = set(inp.cfg["fuels"])
    bad = (set(inp.gens.fuel) | set(inp.candidates.fuel)) - fuels
    if bad:
        errs.append(f"unknown fuels {bad}")
    if inp.gens.id.duplicated().any():
        errs.append("duplicate generator ids")
    share = inp.zones.demand_share.sum()
    if abs(share - 1) > 1e-3:
        errs.append(f"zone demand shares sum to {share:.3f} (must be 1)")
    known = set(inp.cfg.get("vre_profiles", {})) | set(inp.ror_cf.profile)
    bad = {p for p in inp.gens.profile if p} - known
    bad = {p for p in bad if not (inp.input_dir / "profiles" / f"{p}.csv").exists()}
    if bad:
        errs.append(f"unknown profiles {bad}")
    if errs:
        raise ValueError("Input validation failed:\n  " + "\n  ".join(errs))


def plant_economics(inp: Inputs, gens: pd.DataFrame) -> pd.DataFrame:
    """Add fuel cost, emission rate and *dispatch* (short-run) cost to a generator table.

    Dispatch cost != settlement cost: energy under take-or-pay contracts is sunk,
    so it is dispatched at ~zero marginal cost and paid for anyway (deemed energy).
    """
    g = gens.copy()
    fuels, cp, op = inp.cfg["fuels"], inp.cfg.get("carbon_price_usd_t", 0.0), inp.cfg["operation"]
    price = g.fuel.map(lambda f: fuels[f]["price"])
    ef = g.fuel.map(lambda f: fuels[f]["ef"])
    g["fuel_cost_usd_mwh"] = g.heat_rate_gj_mwh * price
    g["co2_rate"] = np.where(g.heat_rate_gj_mwh > 0, g.heat_rate_gj_mwh * ef, 0.0) + g.co2_t_mwh
    carbon = g.co2_rate * cp
    top = g.take_or_pay.astype(bool)
    curtail = np.where(g.tech == "geothermal", op["geothermal_curtail_cost"], op["vre_curtail_cost"])
    g["dispatch_cost"] = np.where(
        top, g.var_om_usd_mwh + carbon - curtail,
        g.fuel_cost_usd_mwh + g.var_om_usd_mwh + g.energy_charge_usd_mwh + carbon)
    return g
