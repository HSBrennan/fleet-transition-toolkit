"""Data models for fleet scenarios and analytics outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional


class EnergySource(str, Enum):
    GRID = "Grid Electricity"
    RENEWABLE = "Renewable Electricity"
    HYDROGEN = "Green Hydrogen"
    LIQUID = "Liquid Fuels"


@dataclass
class EnergyMix:
    grid_share: float
    renewable_share: float
    hydrogen_share: float
    liquid_share: float

    def normalised(self) -> "EnergyMix":
        total = self.grid_share + self.renewable_share + self.hydrogen_share + self.liquid_share
        if total <= 0:
            raise ValueError("Energy mix shares must sum to a positive value.")
        return EnergyMix(
            grid_share=self.grid_share / total,
            renewable_share=self.renewable_share / total,
            hydrogen_share=self.hydrogen_share / total,
            liquid_share=self.liquid_share / total,
        )

    def as_dict(self) -> Dict[EnergySource, float]:
        mix = self.normalised()
        return {
            EnergySource.GRID: mix.grid_share,
            EnergySource.RENEWABLE: mix.renewable_share,
            EnergySource.HYDROGEN: mix.hydrogen_share,
            EnergySource.LIQUID: mix.liquid_share,
        }


@dataclass
class FleetScenario:
    name: str
    vehicle_type: str
    quantity: int
    annual_mileage_km: float
    duty_cycle_hours_per_day: float
    depot_location: str
    energy_mix: EnergyMix
    notes: Optional[str] = None

    def total_distance_km(self) -> float:
        return float(self.quantity) * float(self.annual_mileage_km)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["energy_mix"] = self.energy_mix.as_dict()
        return data


@dataclass
class EnergyDemandResult:
    total_kwh: float
    energy_by_source_kwh: Dict[EnergySource, float]
    total_liquid_litres: float
    hydrogen_kg: float


@dataclass
class EconomicResult:
    capex_total: float
    opex_total: float
    tco_per_km: float
    sensitivity: Dict[str, Dict[str, float]]


@dataclass
class EnvironmentalResult:
    total_emissions_kg: float
    intensity_per_km: float
    abatement_vs_baseline_kg: float
    sensitivity: Dict[str, Dict[str, float]]


@dataclass
class ScenarioAnalysis:
    scenario: FleetScenario
    energy: EnergyDemandResult
    economics: EconomicResult
    environment: EnvironmentalResult

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario.to_dict(),
            "energy": {
                "total_kwh": self.energy.total_kwh,
                "energy_by_source_kwh": {k.value: v for k, v in self.energy.energy_by_source_kwh.items()},
                "total_liquid_litres": self.energy.total_liquid_litres,
            },
            "economics": {
                "capex_total": self.economics.capex_total,
                "opex_total": self.economics.opex_total,
                "tco_per_km": self.economics.tco_per_km,
                "sensitivity": self.economics.sensitivity,
            },
            "environment": {
                "total_emissions_kg": self.environment.total_emissions_kg,
                "intensity_per_km": self.environment.intensity_per_km,
                "abatement_vs_baseline_kg": self.environment.abatement_vs_baseline_kg,
                "sensitivity": self.environment.sensitivity,
            },
        }


def summarise_scenarios(analyses: List[ScenarioAnalysis]) -> str:
    if not analyses:
        return "No scenarios defined."

    lines: List[str] = []
    for analysis in analyses:
        scenario = analysis.scenario
        lines.append(
            (
                f"Scenario '{scenario.name}' for {scenario.quantity} {scenario.vehicle_type}(s) "
                f"driving {scenario.annual_mileage_km:,.0f} km/year each. "
                f"TCO per km: {analysis.economics.tco_per_km:,.2f} USD; "
                f"Emissions intensity: {analysis.environment.intensity_per_km:,.3f} kg/km."
            )
        )
    return " \n".join(lines)

