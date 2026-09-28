"""
Scenario presets for the Earth-orbit page.

The assignment and Vallado cases are the published inputs. The others are
synthetic: a known reference orbit is propagated, observed from the
assignment's ground station (core.frames.eci_to_aer) at times when it is
above 10 deg elevation, and rounded to the assignment's precision (0.01 deg,
1 m). Solving them should therefore recover the reference orbit to within
the rounding error, which the fit-error column makes visible.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from core.constants import MU_EARTH, R_EARTH
from core.frames import eci_to_aer
from core.kepler import coe_to_rv, propagate_rv
from core.time_utils import gmst_degrees

STATION_ECEF = (1344.143, 6068.601, 1429.311)
TIME_FMT = "%Y-%m-%d %H:%M:%S"
_EPOCH = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)


def _observe(a, e, i, raan, argp, nu, n_obs, min_gap_s, search_s, step_s=60.0,
             max_span_s=None):
    """
    First n_obs visible times, at least min_gap_s apart and (if given) all
    within max_span_s of the first, as table rows. The span limit keeps
    LEO observations inside one pass: the zero-revolution Lambert solver
    cannot connect observations more than one orbit apart.
    """
    r0, v0 = coe_to_rv(a, e, i, raan, argp, nu, MU_EARTH)
    rows, last, first = [], None, None
    for t in np.arange(0.0, search_s, step_s):
        if max_span_s is not None and first is not None and t - first > max_span_s:
            rows, last, first = [], None, None
        if last is not None and t - last < min_gap_s:
            continue
        when = _EPOCH + timedelta(seconds=float(t))
        r, _ = propagate_rv(r0, v0, float(t), MU_EARTH)
        az, el, rng = eci_to_aer(r, STATION_ECEF, gmst_degrees(when))
        if el < 10.0:
            continue
        first = t if first is None else first
        rows.append({"time": when.strftime(TIME_FMT), "az": f"{az:.2f}",
                     "el": f"{el:.2f}", "range": f"{rng:.3f}"})
        last = t
        if len(rows) == n_obs:
            return rows
    raise RuntimeError("Preset orbit not visible often enough from the station.")


def _vectors(r1, r2, tof, epoch=""):
    return {"r1": [f"{x:.5f}" for x in r1], "r2": [f"{x:.5f}" for x in r2],
            "tof": f"{tof:g}", "epoch": epoch}


def build_presets():
    presets = {
        "assignment": {
            "label": "Assignment data (2 observations)",
            "mode": "obs",
            "rows": [
                {"time": "2023-04-02 00:30:00", "az": "132.67", "el": "32.44", "range": "16945.450"},
                {"time": "2023-04-02 03:00:00", "az": "123.08", "el": "50.06", "range": "37350.340"},
            ],
        },
        "vallado": {
            "label": "Vallado Example 5-5 (vectors)",
            "mode": "vec",
            "vectors": _vectors([15945.34, 0.0, 0.0], [12214.83899, 10249.46731, 0.0], 4560.0),
        },
        "leo": {
            "label": "Low Earth orbit, ISS-like (3 obs, one pass)",
            "mode": "obs",
            "rows": _observe(R_EARTH + 420, 0.0005, 51.6, 250.0, 90.0, 10.0,
                             n_obs=3, min_gap_s=120, search_s=2 * 86400,
                             step_s=30.0, max_span_s=600),
        },
        "gto": {
            "label": "Geostationary transfer orbit (3 obs)",
            "mode": "obs",
            "rows": _observe((R_EARTH + 250 + R_EARTH + 35786) / 2, 0.73, 6.0, 80.0, 180.0, 60.0,
                             n_obs=3, min_gap_s=1800, search_s=86400),
        },
        "molniya": {
            "label": "Molniya orbit (4 obs)",
            "mode": "obs",
            "rows": _observe(26600.0, 0.74, 63.4, 40.0, 270.0, 120.0,
                             n_obs=4, min_gap_s=2400, search_s=86400),
        },
        "geo": {
            "label": "Geostationary satellite (3 obs)",
            "mode": "obs",
            "rows": _observe(42164.0, 0.0002, 0.05, 0.0, 0.0,
                             (77.5 + gmst_degrees(_EPOCH)) % 360,
                             n_obs=3, min_gap_s=4 * 3600, search_s=86400),
        },
    }
    return presets


PRESETS = build_presets()
DEFAULT_PRESET = "assignment"
