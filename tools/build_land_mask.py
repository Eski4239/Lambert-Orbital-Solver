"""
Build data/land_mask_1deg.json: a 1-degree land/sea grid rasterised from
Natural Earth's 1:110m land polygons (public domain), used to colour the
3D Earth. Run once; the output is committed so the app works offline.

    python tools/build_land_mask.py
"""

import json
import os
import urllib.request

import numpy as np
from matplotlib.path import Path

URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
       "master/geojson/ne_110m_land.geojson")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "land_mask_1deg.json")


def main():
    geo = json.load(urllib.request.urlopen(URL, timeout=60))
    lats = np.arange(90, -91, -1)       # 181 rows, north to south
    lons = np.arange(-180, 181, 1)      # 361 columns
    LON, LAT = np.meshgrid(lons, lats)
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    mask = np.zeros(len(pts), dtype=bool)
    for feat in geo["features"]:
        g = feat["geometry"]
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            inside = Path(poly[0]).contains_points(pts)
            for hole in poly[1:]:
                inside &= ~Path(hole).contains_points(pts)
            mask |= inside
    mask = mask.reshape(LAT.shape)
    mask[lats <= -78, :] = True  # Antarctic interior (polygon edge at -90 is degenerate)
    rows = ["".join("1" if v else "0" for v in row) for row in mask]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "Natural Earth 1:110m land (public domain)",
                   "lat_start": 90, "lon_start": -180, "step_deg": 1, "rows": rows}, f)
    print(f"wrote {OUT}: land fraction {mask.mean():.3f}")


if __name__ == "__main__":
    main()
