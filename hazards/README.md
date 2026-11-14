# Hazards

Models that say when a place on Mars is dangerous to be outside: dust devils first, then
regional dust storms and solar particle events. The results go into the map as numbers per
sol and per hour for each region.

Self-contained: nothing here imports `mars_map`. Downloads and outputs live in `hazards/data/`,
which is not in git.

```bash
.venv\Scripts\pip install -r hazards\requirements.txt
.venv\Scripts\python -m hazards.vortex_rates          # labels: how often vortices hit a station
.venv\Scripts\python -m hazards.train_dust_devils     # train, test, forecast for our region
```

The first run downloads about 1.2 GB (Curiosity pressure, see below) and asks the Mars Climate
Database web server for 96 days of climate; later runs use the cache.

## Product

`hazards/data/gale_boxwork_dust_devils.csv`: expected dust devil encounters per Earth hour at our
map region, for every season bin (30° of Ls) and local solar hour, pressure drop ≥ 0.8 Pa.

At Gale boxwork this comes to 0.4 to 1.0 vortices per sol: lowest around Ls 150, highest in the
dusty season (Ls 240–300), peak hour 12–13 at about one vortex per 6 hours. Night is zero.

## Data

| Station | Place | Labels | Vortices ≥ 0.8 Pa | Sols | Season |
|---|---|---|---|---|---|
| InSight | Elysium Planitia, 4.5 N 135.6 E | catalog, Lorenz et al., NASA PDS, doi:10.17189/71eb-n087 | 853 | 15–390 | Ls 305 → 129 |
| Perseverance | Jezero crater, 18.4 N 77.5 E | catalog, Hueso et al. 2023, doi:10.5281/zenodo.7315863 | 421 by day | 15–415 | Ls 13 → 213 |
| Curiosity | Gale crater, Mount Sharp | **our detector** on raw REMS pressure (NASA PDS), every 6th sol | 22 by day | 4176–4843 | one full year |

Model inputs come from the **Mars Climate Database** 6.2 (LMD / ESA / CNES), climatology
scenario, 2 m above the ground, for the middle of each season bin and hour: heat flux from the
ground, boundary layer height, updraft speed, ground minus air temperature, wind, pressure, dust.

Only daytime (8–18 h) counts. At night the pressure at Gale and Jezero also shows short drops,
but they are slope winds, turbulence or sensor noise, not convective vortices.

## Model

A small neural network (2 × 8 tanh units, numpy, Poisson loss with listening time as exposure,
8 members averaged) that **corrects a physics index** instead of predicting from scratch:

    log(rate) = log(c · heat flux · boundary layer height) + network(heat flux, boundary layer height)

How it got there, each step tested on a station the model did not see:

1. All seven inputs, rate from scratch: worse than a plain hour-of-day table. Wind and ground–air
   temperature difference take values at one station that the other never had.
2. Two inputs (heat flux, boundary layer height), rate from scratch: good in one direction,
   wild in the other (it extrapolated on heat flux values Jezero never had).
3. Two inputs, correcting the physics index, strong L2: never worse than the physics index.

Daytime deviance explained (1 = perfect, 0 = no better than the station's own average):

| Test | network | physics index | hour-of-day table |
|---|---|---|---|
| InSight → Perseverance | **0.19** | −0.10 | 0.05 |
| Perseverance → InSight | 0.44 | 0.44 | 0.23 |
| both → Curiosity (Gale) | −0.30 | −0.26 | −0.77 |
| Gale after one level factor | 0.25 | 0.30 | 0.29 |
| Gale season shape (correlation per sol) | 0.70 | 0.75 | 0.00 |

What this says, plainly:

- **Gale is three times quieter than any model expects.** All of them predict about 63 vortices
  where the detector found 22. A deep crater suppresses the convective layer, and the climate
  database is too coarse to see the crater. So for our region the level comes from Curiosity's
  own count (factor 0.35) and the model gives the shape over hours and seasons.
- **The shape over the year at Gale is right.** With the level fixed, the physics-based models
  follow the seasonal cycle (correlation 0.7–0.75): fewer vortices around Ls 150, most in the
  dusty season. A table that knows only the hour cannot do that.
- **The network barely beats plain physics.** It helps in one of the three tests and ties in the
  others. With two training stations there is not enough variety for it to learn more; the
  physics index alone is almost as good. More stations (or the full InSight mission) are what
  would let it earn its place.
- **For a new place without a station** the forecast is uncalibrated: in a deep crater it may be
  about three times too high.

## Files

| File | What it does |
|---|---|
| `marstime.py` | `mars_ls(utc)`: Mars season from an Earth date (Allison & McEwen 2000), agrees with the catalogs to 0.3° |
| `sites.py` | Stations and our region with coordinates |
| `vortices.py` | Catalog loaders (InSight, Perseverance) into one event format, recording time, `rate_table` |
| `rems.py` | Curiosity: download REMS pressure, detect vortices, recording time per hour |
| `mcd.py` | Mars Climate Database client with a cache: hourly climate of any point and season |
| `dataset.py` | Training table: station × season × hour, counts, listening hours, climate inputs |
| `model.py` | The network, the two baselines, scores |
| `vortex_rates.py` | Pictures of the raw rates per station |
| `train_dust_devils.py` | Tests, pictures, the product CSV |

## Known weak points

- **Perseverance recording time is estimated** from the catalog (`alternating_coverage`), so its
  rates are lower bounds. This alone explains part of the 0.5–1.4 level error between InSight and Jezero.
- **The detectors are not the same.** Two catalogs and our own detector measure the drop
  slightly differently, so 0.8 Pa is not exactly the same cut at the three stations. Our running
  median of 61 s may shave the deepest part off wide vortices at Gale.
- **Curiosity: every 6th sol of one Mars year** (112 sols, 337 daytime hours). 22 vortices is few;
  the season curve at Gale has large error bars.
- **Climatology, not weather.** The inputs are the average climate of a season, so the model
  says how many vortices to expect on a typical sol, not on a particular one.
- InSight is assumed to record all day on every sol (close to true for sols 15–390, not checked).
- Two InSight rows (sol 197) have a date nine months off in the source; the loader rebuilds it.

## Next

1. The full InSight mission (Onodera et al. 2023, about 13 000 events) and the newer Jezero
   catalog: more seasons and more events per station.
2. Real MEDA recording schedule instead of the estimate.
3. Dust storms (Mars Dust Activity Database) and solar particle events, in the same shape:
   expected events per sol for a region.
