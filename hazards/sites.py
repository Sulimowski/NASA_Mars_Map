"""Places the hazard models know about: the stations that measured, and our map regions."""

SITES = {
    # Stations with vortex observations (the training and test labels).
    "insight": {"name": "InSight", "place": "Elysium Planitia", "lat": 4.502, "lon": 135.623},
    "perseverance": {"name": "Perseverance", "place": "Jezero crater", "lat": 18.445, "lon": 77.451},
    # Curiosity moves; two positions bracket the data we use.
    "curiosity_bradbury": {"name": "Curiosity", "place": "Gale crater floor (Bradbury, sols 1-700)",
                           "lat": -4.589, "lon": 137.441},
    "curiosity_boxwork": {"name": "Curiosity", "place": "Gale crater, Mount Sharp boxwork (our map)",
                          "lat": -4.812, "lon": 137.383},
}

# Our map regions (same names and centres as mars_map/grid.py REGIONS). `calibrate` names the
# station nearby whose own vortex count sets the level of the forecast there; None = no station,
# the forecast is the model alone.
REGIONS = {
    "gale_boxwork": {"lat": -4.812, "lon": 137.383, "calibrate": "curiosity"},
    "gale_mount_sharp": {"lat": -4.75, "lon": 137.38, "calibrate": "curiosity"},
    "jezero_delta": {"lat": 18.458, "lon": 77.402, "calibrate": "perseverance"},
    "nili_planum": {"lat": 18.432, "lon": 77.238, "calibrate": "perseverance"},
}
for _name, _r in REGIONS.items():                 # regions are places too: climate is fetched the same way
    SITES.setdefault(f"region_{_name}", {"name": _name, "place": "map region", "lat": _r["lat"], "lon": _r["lon"]})
