"""
Tests for the Earth-orbit scope: inverse frame transforms, the solve
pipeline (core/solve.py), UI input parsing, and the scenario presets.

Run: python -m pytest tests
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import MU_EARTH                                          # noqa: E402
from core.frames import (aer_to_eci, ecef_to_geodetic, ecef_to_latlon,       # noqa: E402
                         eci_to_aer, eci_to_ecef)
from core.solve import Observation, solve_from_observations, solve_orbit     # noqa: E402

STATION = (1344.143, 6068.601, 1429.311)


# ----------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------
def test_eci_to_aer_inverts_aer_to_eci():
    rng = np.random.default_rng(0)
    for _ in range(200):
        az, el, r, g = rng.uniform(0, 360), rng.uniform(-89, 89), rng.uniform(300, 5e4), rng.uniform(0, 360)
        back = eci_to_aer(aer_to_eci(az, el, r, STATION, g), STATION, g)
        np.testing.assert_allclose(back, (az, el, r), rtol=1e-9, atol=1e-8)


def test_eci_to_ecef_is_rotation_by_minus_gmst():
    r = np.array([7000.0, 0.0, 100.0])
    np.testing.assert_allclose(eci_to_ecef(r, 90.0), [0.0, -7000.0, 100.0], atol=1e-9)


def test_ecef_to_latlon_matches_iterative_geodetic():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(100, 3))
    pts = pts / np.linalg.norm(pts, axis=1)[:, None] * rng.uniform(6380, 45000, size=(100, 1))
    lat, lon, alt = ecef_to_latlon(pts)
    for p, la, lo, al in zip(pts, lat, lon, alt):
        la2, lo2, al2 = ecef_to_geodetic(*p)
        assert la == pytest.approx(la2, abs=1e-6)
        assert lo == pytest.approx(lo2, abs=1e-9)
        assert al == pytest.approx(al2, abs=1e-3)


# ----------------------------------------------------------------------
# Solve pipeline
# ----------------------------------------------------------------------
def _assignment_obs():
    utc = timezone.utc
    return [Observation(datetime(2023, 4, 2, 0, 30, tzinfo=utc), 132.67, 32.44, 16945.450),
            Observation(datetime(2023, 4, 2, 3, 0, tzinfo=utc), 123.08, 50.06, 37350.340)]


def test_pipeline_reproduces_assignment_answer():
    sol, residuals = solve_from_observations(_assignment_obs(), STATION)
    assert sol.elements["a"] == pytest.approx(28196.775768, abs=1e-5)
    assert sol.elements["e"] == pytest.approx(0.767944, abs=1e-6)
    assert residuals == pytest.approx([0.0, 0.0], abs=1e-6)
    assert sol.is_closed and not sol.warnings


def test_pipeline_order_independent():
    obs = _assignment_obs()
    sol_a, _ = solve_from_observations(obs, STATION, 0, 1)
    sol_b, _ = solve_from_observations(obs[::-1], STATION, 0, 1)
    np.testing.assert_allclose(sol_a.v1, sol_b.v1)


def test_pipeline_rejects_bad_pairs():
    with pytest.raises(ValueError):
        solve_from_observations(_assignment_obs(), STATION, 0, 0)
    with pytest.raises(ValueError):
        solve_orbit([7000, 0, 0], [0, 7000, 0], -10.0)


def test_warning_when_trajectory_hits_earth():
    # Vallado Example 5-5: a pure Lambert geometry exercise whose conic
    # dips below the Earth's surface (perigee radius ~3,190 km).
    sol = solve_orbit([15945.34, 0, 0], [12214.83899, 10249.46731, 0], 4560.0)
    assert sol.rp_km < 6378.137
    assert any("intersects the Earth" in w for w in sol.warnings)


def test_state_at_matches_lambert_endpoint():
    sol, _ = solve_from_observations(_assignment_obs(), STATION)
    r, v = sol.state_at(sol.tof_s)
    np.testing.assert_allclose(r, sol.r2, rtol=1e-9)
    np.testing.assert_allclose(v, sol.v2, rtol=1e-7)


# ----------------------------------------------------------------------
# UI parsing and presets
# ----------------------------------------------------------------------
def test_parse_observations_flags_each_bad_cell():
    from app.earth_page import parse_observations
    rows = [{"time": "2023-04-02 00:30:00", "az": "400", "el": "32", "range": "-5"},
            {"time": "yesterday", "az": "10", "el": "abc", "range": "1000"}]
    obs, errors = parse_observations(rows)
    assert obs is None
    assert {(i, c) for i, c, _ in errors} == {(0, "az"), (0, "range"), (1, "time"), (1, "el")}


def test_parse_time_formats():
    from app.earth_page import parse_time
    want = datetime(2023, 4, 2, 3, 0, tzinfo=timezone.utc)
    for text in ("2023-04-02 03:00:00", "2023-04-02T03:00:00Z", "2023-04-02 03:00", " 2023-04-02 03:00:00.000 "):
        assert parse_time(text) == want


@pytest.mark.parametrize("key", ["leo", "gto", "molniya", "geo"])
def test_presets_recover_reference_orbit(key):
    """Synthetic presets must solve back to the orbit they were generated from."""
    from app.earth_page import parse_observations
    from app.presets import PRESETS, REFERENCE_ORBITS, STATIONS
    a_ref, e_ref, i_ref = REFERENCE_ORBITS[key][:3]
    preset = PRESETS[key]
    obs, errors = parse_observations(preset["rows"])
    assert not errors
    sol, residuals = solve_from_observations(obs, STATIONS[preset["station"]], 0, len(obs) - 1)
    el = sol.elements
    assert el["a"] == pytest.approx(a_ref, rel=2e-3)
    assert el["e"] == pytest.approx(e_ref, abs=2e-3)
    assert el["i"] == pytest.approx(i_ref, abs=0.05)
    assert max(residuals) < 10.0  # km: rounding to 0.01 deg / 1 m only


def test_default_station_is_madrid():
    from app.earth_page import station_label
    from app.presets import DEFAULT_PRESET, PRESETS, STATIONS
    station = STATIONS[PRESETS[DEFAULT_PRESET]["station"]]
    assert station_label(station) == "Madrid, Spain · 40.42°N 3.70°W"
    assert station_label((1344.143, 6068.601, 1429.311)).startswith("Assignment station")
