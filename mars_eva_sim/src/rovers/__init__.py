"""Mars rovers: profiles of the vehicles and their engineering limits (algorithms R1-R3).

Only rovers that are operating on Mars or are officially planned to land are
modelled: Curiosity, Perseverance and Rosalind Franklin.

    profiles             what each rover is: mass, wheels, limits, power system
    rover_mobility       R1: clearance, tilt, wheel slip (Bekker-Wong), wheel-walking
    rover_power_thermal  R2: drive power, heaters, MMRTG and solar generation, battery
    rover_radiation      R3: electronics upsets during a solar particle event
    types                state and step records used by the route search
"""
