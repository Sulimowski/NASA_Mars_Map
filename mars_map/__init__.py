from .site import (DATA_DIR, LAYER_SPEC, PROJECT_ROOT, TERRAIN_CLASSES,
                   MissionParams, Site, Target, slope_deg)

STUB_DIR = DATA_DIR / "gale_stub"


def load_stub() -> Site:
    """Load the synthetic test site; generate it first if it is missing."""
    if not (STUB_DIR / "site.json").exists():
        from .synthetic import make_gale_like_site
        print(f"[mars_map] {STUB_DIR} not found, generating stub site...")
        make_gale_like_site().save(STUB_DIR)
    return Site.load(STUB_DIR)


__all__ = ["Site", "Target", "MissionParams", "LAYER_SPEC", "TERRAIN_CLASSES",
           "slope_deg", "load_stub", "PROJECT_ROOT", "DATA_DIR", "STUB_DIR"]
