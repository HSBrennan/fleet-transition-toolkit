"""Baseline assumptions and lookup tables used across the fleet toolkit."""

from __future__ import annotations

from typing import Dict, Final, Mapping


VEHICLE_TYPES: Final = [
    "Car",
    "Light Duty Van",
    "Heavy Goods Vehicle",
    "Bus",
    "Motorbike",
]


# Energy consumption (kWh per km) and baseline fuel intensity (L per 100 km)
VEHICLE_ENERGY_INTENSITY: Final[Mapping[str, Dict[str, float]]] = {
    "Car": {
        "electric_kwh_per_km": 0.18,
        "liquid_l_per_100km": 6.5,
    },
    "Light Duty Van": {
        "electric_kwh_per_km": 0.25,
        "liquid_l_per_100km": 9.8,
    },
    "Heavy Goods Vehicle": {
        "electric_kwh_per_km": 1.3,
        "liquid_l_per_100km": 32.0,
    },
    "Bus": {
        "electric_kwh_per_km": 1.8,
        "liquid_l_per_100km": 45.0,
    },
    "Motorbike": {
        "electric_kwh_per_km": 0.06,
        "liquid_l_per_100km": 3.5,
    },
}


# Capex per vehicle (USD) and residual value fraction at year 10
VEHICLE_CAPEX: Final[Mapping[str, Dict[str, float]]] = {
    "Car": {"capex": 45000.0, "residual_fraction": 0.35},
    "Light Duty Van": {"capex": 65000.0, "residual_fraction": 0.30},
    "Heavy Goods Vehicle": {"capex": 310000.0, "residual_fraction": 0.25},
    "Bus": {"capex": 520000.0, "residual_fraction": 0.20},
    "Motorbike": {"capex": 12000.0, "residual_fraction": 0.40},
}


# Fuel prices in USD per unit (kWh for electricity / H2, litre for liquids)
ENERGY_PRICES: Final[Mapping[str, float]] = {
    "Grid Electricity": 0.16,
    "Renewable Electricity": 0.20,
    "Green Hydrogen": 8.50,
    "Liquid Fuels": 1.70,
}


# Opex overhead per vehicle per year (USD)
VEHICLE_FIXED_OPEX: Final[Mapping[str, float]] = {
    "Car": 1200.0,
    "Light Duty Van": 1800.0,
    "Heavy Goods Vehicle": 5400.0,
    "Bus": 6200.0,
    "Motorbike": 600.0,
}


# Emissions factors (kg CO2e per unit)
EMISSIONS_FACTORS: Final[Mapping[str, float]] = {
    "Grid Electricity": 0.32,  # kg CO2e per kWh
    "Renewable Electricity": 0.05,
    "Green Hydrogen": 0.8,  # kg CO2e per kg (converted from kt)
    "Liquid Fuels": 2.68,  # kg CO2e per litre of diesel/petrol equivalent
}


# Energy density conversions
ENERGY_DENSITY: Final[Mapping[str, float]] = {
    "Green Hydrogen": 55.0,  # kWh per kg
    "Liquid Fuels": 9.7,  # kWh per litre (diesel equivalent)
}


# Baseline emissions intensity (kg CO2e per km) for current fleet
BASELINE_INTENSITY: Final[Mapping[str, float]] = {
    "Car": 0.192,
    "Light Duty Van": 0.301,
    "Heavy Goods Vehicle": 1.12,
    "Bus": 1.64,
    "Motorbike": 0.115,
}


SENSITIVITY_BANDS: Final = {
    "energy_price": 0.25,
    "capex": 0.20,
    "emissions": 0.30,
}


DEFAULT_ANALYSIS_YEARS: Final = 10

