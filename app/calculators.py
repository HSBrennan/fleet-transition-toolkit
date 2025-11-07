"""Core analytics for the fleet transition toolkit."""

from __future__ import annotations

from typing import Dict

import math

from . import constants
from .models import (
    EconomicResult,
    EnergyDemandResult,
    EnergySource,
    EnvironmentalResult,
    FleetScenario,
    ScenarioAnalysis,
)


def _ensure_vehicle_type(vehicle_type: str) -> str:
    if vehicle_type not in constants.VEHICLE_ENERGY_INTENSITY:
        raise ValueError(f"Unsupported vehicle type: {vehicle_type}")
    return vehicle_type


def compute_energy_demand(scenario: FleetScenario) -> EnergyDemandResult:
    vehicle_type = _ensure_vehicle_type(scenario.vehicle_type)
    intensity = constants.VEHICLE_ENERGY_INTENSITY[vehicle_type]
    total_km = scenario.total_distance_km()

    electric_intensity_kwh_per_km = intensity["electric_kwh_per_km"]
    baseline_liquid_l_per_km = intensity["liquid_l_per_100km"] / 100.0

    energy_mix = scenario.energy_mix.as_dict()

    total_electric_kwh = total_km * electric_intensity_kwh_per_km
    energy_by_source: Dict[EnergySource, float] = {}

    # Electricity pathways
    for source in (EnergySource.GRID, EnergySource.RENEWABLE):
        share = energy_mix[source]
        energy_by_source[source] = total_electric_kwh * share

    # Hydrogen: convert to kg using energy density
    hydrogen_share = energy_mix[EnergySource.HYDROGEN]
    hydrogen_kwh = total_electric_kwh * hydrogen_share
    hydrogen_density = constants.ENERGY_DENSITY[EnergySource.HYDROGEN.value]
    hydrogen_kg = hydrogen_kwh / hydrogen_density if hydrogen_density else 0.0
    energy_by_source[EnergySource.HYDROGEN] = hydrogen_kwh

    # Liquid fuels: use baseline fuel intensity to estimate litres
    liquid_share = energy_mix[EnergySource.LIQUID]
    total_liquid_litres = total_km * baseline_liquid_l_per_km * liquid_share
    liquid_density = constants.ENERGY_DENSITY[EnergySource.LIQUID.value]
    liquid_kwh = total_liquid_litres * liquid_density
    energy_by_source[EnergySource.LIQUID] = liquid_kwh

    total_kwh = sum(energy_by_source.values())

    return EnergyDemandResult(
        total_kwh=total_kwh,
        energy_by_source_kwh=energy_by_source,
        total_liquid_litres=total_liquid_litres,
        hydrogen_kg=hydrogen_kg,
    )


def compute_economic_analysis(
    scenario: FleetScenario, energy: EnergyDemandResult
) -> EconomicResult:
    vehicle_type = _ensure_vehicle_type(scenario.vehicle_type)
    capex_info = constants.VEHICLE_CAPEX[vehicle_type]
    quantity = scenario.quantity
    total_km = max(scenario.total_distance_km(), 1.0)

    capex_total = capex_info["capex"] * quantity
    residual_value = capex_total * capex_info["residual_fraction"]
    amortised_capex = (capex_total - residual_value) / constants.DEFAULT_ANALYSIS_YEARS

    fixed_opex = constants.VEHICLE_FIXED_OPEX[vehicle_type] * quantity

    # Variable energy costs
    grid_cost = energy.energy_by_source_kwh[EnergySource.GRID] * constants.ENERGY_PRICES[EnergySource.GRID.value]
    renewable_cost = (
        energy.energy_by_source_kwh[EnergySource.RENEWABLE]
        * constants.ENERGY_PRICES[EnergySource.RENEWABLE.value]
    )
    hydrogen_cost = energy.hydrogen_kg * constants.ENERGY_PRICES[EnergySource.HYDROGEN.value]
    liquid_cost = energy.total_liquid_litres * constants.ENERGY_PRICES[EnergySource.LIQUID.value]

    variable_opex = grid_cost + renewable_cost + hydrogen_cost + liquid_cost
    opex_total = fixed_opex + variable_opex

    annualised_total_cost = amortised_capex + opex_total
    tco_per_km = annualised_total_cost / total_km

    sensitivity_band = constants.SENSITIVITY_BANDS
    energy_factor = sensitivity_band["energy_price"]
    capex_factor = sensitivity_band["capex"]

    sensitivity = {
        "energy_price": {
            "low_opex": fixed_opex + variable_opex * (1 - energy_factor),
            "high_opex": fixed_opex + variable_opex * (1 + energy_factor),
        },
        "capex": {
            "low_capex": ((capex_total * (1 - capex_factor)) - residual_value)
            / constants.DEFAULT_ANALYSIS_YEARS,
            "high_capex": ((capex_total * (1 + capex_factor)) - residual_value)
            / constants.DEFAULT_ANALYSIS_YEARS,
        },
    }

    sensitivity["energy_price"]["tco_low"] = (
        sensitivity["energy_price"]["low_opex"] + amortised_capex
    ) / total_km
    sensitivity["energy_price"]["tco_high"] = (
        sensitivity["energy_price"]["high_opex"] + amortised_capex
    ) / total_km

    sensitivity["capex"]["tco_low"] = (
        sensitivity["capex"]["low_capex"] + opex_total
    ) / total_km
    sensitivity["capex"]["tco_high"] = (
        sensitivity["capex"]["high_capex"] + opex_total
    ) / total_km

    return EconomicResult(
        capex_total=capex_total,
        opex_total=opex_total,
        tco_per_km=tco_per_km,
        sensitivity=sensitivity,
    )


def compute_environmental_analysis(
    scenario: FleetScenario, energy: EnergyDemandResult
) -> EnvironmentalResult:
    vehicle_type = _ensure_vehicle_type(scenario.vehicle_type)
    total_km = max(scenario.total_distance_km(), 1.0)

    emissions = 0.0
    emissions += (
        energy.energy_by_source_kwh[EnergySource.GRID]
        * constants.EMISSIONS_FACTORS[EnergySource.GRID.value]
    )
    emissions += (
        energy.energy_by_source_kwh[EnergySource.RENEWABLE]
        * constants.EMISSIONS_FACTORS[EnergySource.RENEWABLE.value]
    )
    emissions += energy.hydrogen_kg * constants.EMISSIONS_FACTORS[EnergySource.HYDROGEN.value]
    emissions += energy.total_liquid_litres * constants.EMISSIONS_FACTORS[EnergySource.LIQUID.value]

    intensity_per_km = emissions / total_km
    baseline_emissions = constants.BASELINE_INTENSITY[vehicle_type] * total_km
    abatement = baseline_emissions - emissions

    emissions_factor = constants.SENSITIVITY_BANDS["emissions"]
    sensitivity = {
        "emissions_factor": {
            "low": intensity_per_km * (1 - emissions_factor),
            "high": intensity_per_km * (1 + emissions_factor),
        }
    }

    return EnvironmentalResult(
        total_emissions_kg=emissions,
        intensity_per_km=intensity_per_km,
        abatement_vs_baseline_kg=abatement,
        sensitivity=sensitivity,
    )


def run_full_analysis(scenario: FleetScenario) -> ScenarioAnalysis:
    energy = compute_energy_demand(scenario)
    economics = compute_economic_analysis(scenario, energy)
    environment = compute_environmental_analysis(scenario, energy)
    return ScenarioAnalysis(
        scenario=scenario,
        energy=energy,
        economics=economics,
        environment=environment,
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}" if not math.isnan(value) else "N/A"


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"

