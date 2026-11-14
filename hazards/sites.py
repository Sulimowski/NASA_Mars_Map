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
