"""Generate the synthetic Gale-like test site.

    python scripts/make_stub_site.py                 # data/gale_stub, 2 km at 2 m/px
    python scripts/make_stub_site.py --res 1         # 1 m/px (4x bigger)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mars_map.preview import render  # noqa: E402
from mars_map.synthetic import make_gale_like_site  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gale_stub")
    ap.add_argument("--size", type=float, default=2000.0, help="site width/height, m")
    ap.add_argument("--res", type=float, default=2.0, help="metres per pixel")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    site = make_gale_like_site(size_m=args.size, resolution_m=args.res, seed=args.seed)
    folder = site.save(args.out)
    png = render(site, folder / "preview.png")

    rows, cols = site.shape
    print(f"saved {folder}  ({rows}x{cols} px, {site.resolution_m} m/px)")
    print(f"preview {png}")
    for t in site.targets:
        r, c = site.to_pixel(t.x_m, t.y_m)
        cls = int(site.layers["terrain_class"][r, c])
        print(f"  {t.name:22s} px=({r},{c}) slope={site.layers['slope'][r, c]:.1f} class={cls}")


if __name__ == "__main__":
    main()
