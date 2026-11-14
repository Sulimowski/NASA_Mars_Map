# Radiation and temperature maps

Maps of the environment an astronaut walks through, for a chosen sol and time of day.
Built on top of the site from [DATA_CONTRACT.md](DATA_CONTRACT.md); nothing in the contract changes.

```bash
.venv\Scripts\python scripts\make_env_maps.py            # data/gale_stub/env_preview.png
.venv\Scripts\python scripts\make_env_maps.py --sol 23   # a dust-storm sol
```

```python
from mars_map import Environment, load_stub

site = load_stub()
env  = Environment(site)                       # ~3 s, do it once

dose = env.dose_rate(sol=17)                   # µSv per Earth hour
temp = env.temperature(sol=0, ltst_h=13.0)     # °C, ground
lit  = env.sunlit(sol=0, ltst_h=8.0)           # bool, False = shadow or night
sun  = env.direct_sunlight(sol=0, ltst_h=8.0)  # W/m², direct beam after dust

r, c = site.to_pixel(*site.start)
print(dose[r, c], temp[r, c])
```

Every map has the shape of the site layers and is indexed the same way (`map[row, col]`).

## Time

| Name | Meaning |
|---|---|
| `sol` | a sol from `site.series["sol"]` |
| `ltst_h` | local true solar time, 0..24 Mars hours, 12 = noon. One Mars hour = 1.0275 Earth hours |
| `env.ls(sol)` | season as solar longitude Ls, degrees. The stub has no season, so sol 0 is pinned to Ls 150 (`Environment(site, ls_at_sol0=...)` to change) |

Dose rate is per **Earth** hour, like `eva_max_hours` in `MissionParams`.

## What the site data gives, and what the model adds

| | From `site.series` (one number per sol) | Added by the model (varies over the map) |
|---|---|---|
| Radiation | dose per sol, including solar particle events | less dose under cliffs and in craters (blocked sky), slightly less on low ground (more air) |
| Temperature | minimum and maximum per sol | time of day; sand swings more than bedrock (`thermal_inertia`); Sun-facing slopes warmer, shadows colder |
| Light | dust optical depth `tau` | Sun position for the season and hour, cast shadows, dimming of the direct beam by dust |

Open flat ground with the site's median thermal inertia reproduces the series exactly.

## Reading the radiation map

On a 2 km site radiation is nearly uniform: the model changes it by a few percent, up to
about 20 % right under the mesa cliff. What changes it a lot is the **sol**: the particle
event on sols 17 to 18 multiplies the whole map by 4. For a route this means the dose is
mostly `rate × time outside`, and the big lever is picking the sol, not the path.

## Assumptions to replace with real numbers

All of them are fields of `RadiationParams` and `TemperatureParams`.

- How much dose a blocked sky removes (`shielding_efficiency = 0.5`) and how dose depends on
  the air column (`column_sensitivity = -0.5`): order-of-magnitude values in the spirit of
  MSL/RAD results, not fitted to data.
- How strongly the daily temperature swing depends on thermal inertia (`inertia_exponent = -0.35`).
- The daily curve: warmest one hour after noon, coldest at sunrise.
- The series temperatures are treated as **ground** temperature.

## Known gaps

- Every cell peaks at the same hour; an east-facing slope really warms earlier.
- The season changes the Sun's path and strength but not the series min/max. Real per-sol
  data will carry the seasonal trend by itself.
- `direct_sunlight` leaves out light scattered by dust, so it understates total light on
  dusty sols.
- No air temperature, wind chill or suit model yet: these are ground and sky conditions only.
