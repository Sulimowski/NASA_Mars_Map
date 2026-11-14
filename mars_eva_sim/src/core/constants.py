"""Physical constants and unit conversions.

All constants are SI. The conversion helpers at the bottom are the only place
where non-SI units (km/h, degrees Celsius, psi, beats per minute, mSv) appear;
they are meant for configuration parsing, input adapters and the reporting layer.
"""

from __future__ import annotations

import math
from typing import Final

# --- gravity ---------------------------------------------------------------
G_MARS_M_S2: Final[float] = 3.711
G_EARTH_M_S2: Final[float] = 9.80665

# --- radiation heat transfer -----------------------------------------------
STEFAN_BOLTZMANN_W_M2_K4: Final[float] = 5.670374419e-8

# --- respiration -----------------------------------------------------------
# Metabolic energy released per cubic metre of O2 consumed (20.2 kJ per litre, STP).
O2_ENERGY_EQUIVALENT_J_M3: Final[float] = 2.02e7
O2_DENSITY_STP_KG_M3: Final[float] = 1.429
CO2_DENSITY_STP_KG_M3: Final[float] = 1.977

# --- water -----------------------------------------------------------------
WATER_TRIPLE_POINT_PA: Final[float] = 611.657
WATER_TRIPLE_POINT_K: Final[float] = 273.16
WATER_LATENT_HEAT_VAPORIZATION_J_KG: Final[float] = 2.43e6
WATER_LATENT_HEAT_SUBLIMATION_J_KG: Final[float] = 2.834e6

# --- molar masses, kg/mol --------------------------------------------------
MOLAR_MASS_O2_KG_MOL: Final[float] = 0.031998
MOLAR_MASS_CO2_KG_MOL: Final[float] = 0.044010
MOLAR_MASS_N2_KG_MOL: Final[float] = 0.028014
MOLAR_MASS_H2O_KG_MOL: Final[float] = 0.018015

# --- time ------------------------------------------------------------------
SOL_S: Final[float] = 88_775.244
EARTH_DAY_S: Final[float] = 86_400.0
EARTH_YEAR_S: Final[float] = 365.25 * EARTH_DAY_S
EARTH_MONTH_S: Final[float] = EARTH_YEAR_S / 12.0
HOUR_S: Final[float] = 3_600.0
MINUTE_S: Final[float] = 60.0

# --- Mars atmosphere -------------------------------------------------------
MARS_REFERENCE_PRESSURE_PA: Final[float] = 610.0
MARS_SCALE_HEIGHT_M: Final[float] = 11_100.0
MARS_MEAN_SOLAR_FLUX_W_M2: Final[float] = 590.0
MARS_OBLIQUITY_RAD: Final[float] = math.radians(25.19)

# --- temperature -----------------------------------------------------------
ZERO_CELSIUS_K: Final[float] = 273.15


# --- unit conversions (adapters and reporting only) ------------------------
def kmh_to_ms(speed_kmh: float) -> float:
    """km/h -> m/s."""
    return speed_kmh / 3.6


def ms_to_kmh(speed_m_s: float) -> float:
    """m/s -> km/h."""
    return speed_m_s * 3.6


def celsius_to_kelvin(temp_c: float) -> float:
    """Degrees Celsius -> kelvin."""
    return temp_c + ZERO_CELSIUS_K


def kelvin_to_celsius(temp_k: float) -> float:
    """Kelvin -> degrees Celsius."""
    return temp_k - ZERO_CELSIUS_K


def bpm_to_hz(beats_per_minute: float) -> float:
    """Heart rate, beats per minute -> Hz."""
    return beats_per_minute / 60.0


def hz_to_bpm(frequency_hz: float) -> float:
    """Heart rate, Hz -> beats per minute."""
    return frequency_hz * 60.0


def psi_to_pa(pressure_psi: float) -> float:
    """Pounds per square inch -> pascal."""
    return pressure_psi * 6_894.757


def ml_kg_min_to_m3_kg_s(vo2_ml_kg_min: float) -> float:
    """Oxygen uptake, ml O2 per kg per minute -> m^3 O2 per kg per second."""
    return vo2_ml_kg_min * 1.0e-6 / 60.0


def wh_to_j(energy_wh: float) -> float:
    """Watt-hours -> joules."""
    return energy_wh * 3_600.0
