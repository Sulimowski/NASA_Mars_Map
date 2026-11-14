"""Mars EVA simulator: astronaut physiology, suit life support and resource-constrained routing.

Layout
------
core/        physical constants, tunable configuration, data schemas, exceptions
providers/   contracts for terrain and environment data, synthetic testbeds, site adapter
physiology/  algorithms A1-A12 (one module per algorithm)
routing/     algorithms B1-B5 (edge physics, shelter maps, Pareto A*, mission plan, reachability)
reporting/   route profiles and raster layers in map/report units

Every quantity inside ``core``, ``physiology`` and ``routing`` is in SI units.
"""

__version__ = "0.1.0"
