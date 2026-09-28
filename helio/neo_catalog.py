"""
Near-Earth asteroid catalog from the JPL Small-Body Database (SBDB).

A snapshot is bundled in data/neo_asteroids.csv.gz so the app works
offline; fetch_catalog() refreshes it from the SBDB Query API
(https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html).

Elements are osculating heliocentric elements referred to the ecliptic and
mean equinox of J2000 (the same frame as helio.planets), at each object's
own epoch (TDB Julian Date). Angles in degrees, a/q/Q/MOID in AU.

Orbit classes (JPL definitions, q = perihelion, Q = aphelion distance):
    ATE  Aten    a < 1.0 AU, Q > 0.983 AU
    APO  Apollo  a > 1.0 AU, q < 1.017 AU
    AMO  Amor    a > 1.0 AU, 1.017 < q < 1.3 AU
    IEO  Atira   Q < 0.983 AU (orbit entirely inside Earth's)
A potentially hazardous asteroid (PHA) has Earth MOID <= 0.05 AU and
H <= 22 (roughly 140 m or larger).
"""

import gzip
import io
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.constants import AU

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SNAPSHOT = os.path.join(DATA_DIR, "neo_asteroids.csv.gz")
API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
FIELDS = ["spkid", "full_name", "pdes", "name", "class", "pha", "H", "diameter",
          "a", "e", "i", "om", "w", "ma", "epoch", "moid", "q", "ad", "per_y"]

CLASS_NAMES = {"ATE": "Aten", "APO": "Apollo", "AMO": "Amor", "IEO": "Atira"}
LUNAR_DISTANCE_AU = 384400.0 / AU


def fetch_catalog(path=SNAPSHOT, timeout=120):
    """Download all near-Earth asteroids from SBDB and save a gzip CSV."""
    query = urllib.parse.urlencode({"fields": ",".join(FIELDS), "sb-group": "neo",
                                    "sb-kind": "a", "full-prec": "true"})
    with urllib.request.urlopen(f"{API}?{query}", timeout=timeout) as resp:
        payload = json.load(resp)
    df = pd.DataFrame(payload["data"], columns=payload["fields"])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        f.write(f"# JPL SBDB near-Earth asteroids, retrieved {stamp}, {len(df)} objects\n")
        df.to_csv(f, index=False)
    return load_catalog(path)


def snapshot_info(path=SNAPSHOT):
    """First-line header of the snapshot, e.g. retrieval date and count."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.readline().lstrip("# ").strip()


def load_catalog(path=SNAPSHOT):
    """Catalog as a DataFrame with numeric columns and a few derived ones."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, comment="#", dtype={"pdes": str, "name": str, "spkid": str})
    for col in ("H", "diameter", "a", "e", "i", "om", "w", "ma", "epoch", "moid", "q", "ad", "per_y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["a", "e", "i", "om", "w", "ma", "epoch"])
    df = df[(df["e"] < 1) & (df["a"] > 0)].reset_index(drop=True)
    df["display"] = df["full_name"].str.strip()
    df["pha"] = df["pha"].eq("Y")
    df["class_name"] = df["class"].map(CLASS_NAMES).fillna(df["class"])
    df["moid_ld"] = df["moid"] / LUNAR_DISTANCE_AU
    return df


def filter_catalog(df, text="", classes=None, pha_only=False):
    """Rows matching a name/designation search, orbit classes and PHA flag."""
    mask = np.ones(len(df), dtype=bool)
    text = (text or "").strip().lower()
    if text:
        mask &= df["display"].str.lower().str.contains(text, regex=False).to_numpy()
    if classes is not None:
        mask &= df["class"].isin(classes).to_numpy()
    if pha_only:
        mask &= df["pha"].to_numpy()
    return df[mask]


def estimated_diameter_km(H, albedo=0.14):
    """Diameter (km) from absolute magnitude H for an assumed geometric albedo."""
    return 1329.0 / np.sqrt(albedo) * 10 ** (-0.2 * np.asarray(H, dtype=float))


def elements_of(row):
    """(a_km, e, i, raan, argp, M0, epoch_jd) of one catalog row."""
    return (row["a"] * AU, row["e"], row["i"], row["om"], row["w"], row["ma"], row["epoch"])
