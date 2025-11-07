import pytest

from app import calculators, models


def build_scenario(**overrides) -> models.FleetScenario:
    mix = models.EnergyMix(grid_share=40, renewable_share=30, hydrogen_share=20, liquid_share=10)
    scenario = models.FleetScenario(
        name="Test Fleet",
        vehicle_type="Car",
        quantity=100,
        annual_mileage_km=30000,
        duty_cycle_hours_per_day=8.0,
        depot_location="Test City",
        energy_mix=mix,
        notes="",
    )
    for key, value in overrides.items():
        setattr(scenario, key, value)
    return scenario


def test_energy_demand_breakdown():
    scenario = build_scenario()
    energy = calculators.compute_energy_demand(scenario)

    assert energy.total_kwh > 0
    assert abs(sum(energy.energy_by_source_kwh.values()) - energy.total_kwh) < 1e-6
    assert energy.total_liquid_litres >= 0
    assert energy.hydrogen_kg >= 0


def test_economic_analysis_tco_positive():
    scenario = build_scenario()
    energy = calculators.compute_energy_demand(scenario)
    economics = calculators.compute_economic_analysis(scenario, energy)

    assert economics.capex_total > 0
    assert economics.opex_total > 0
    assert economics.tco_per_km > 0


def test_environmental_abatement():
    scenario = build_scenario()
    analysis = calculators.run_full_analysis(scenario)

    assert analysis.environment.abatement_vs_baseline_kg != 0
    assert analysis.environment.total_emissions_kg >= 0


def test_energy_mix_normalises():
    mix = models.EnergyMix(grid_share=10, renewable_share=10, hydrogen_share=0, liquid_share=0)
    normalised = mix.normalised()
    total = sum(normalised.as_dict().values())
    assert pytest.approx(total, rel=1e-9) == 1.0

