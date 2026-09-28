"""
pytest suite for the orbital-mechanics core and heliocentric helpers.

Complements tests/run_stress_tests.py (edge cases / input validation) with
independent cross-checks: every Lambert solution is propagated forward by
its time of flight with core.kepler and must land on r2; Kepler propagation
is checked against a textbook example; planet positions against JPL
Horizons.

Run: python -m pytest tests
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import AU, MU_EARTH, MU_SUN, DAY                 # noqa: E402
from core.elements import rv_to_elements                              # noqa: E402
from core.kepler import (coe_to_rv, orbit_curve, position_at,         # noqa: E402
                         propagate_rv, solve_kepler, true_from_mean)
from core.lamsolbert import lamsolbert                                # noqa: E402
from helio.planets import planet_state                                # noqa: E402


# ----------------------------------------------------------------------
# Lambert
# ----------------------------------------------------------------------
def test_lambert_vallado_5_5():
    r1 = np.array([15945.34, 0.0, 0.0])
    r2 = np.array([12214.83899, 10249.46731, 0.0])
    v1, v2 = lamsolbert(r1, r2, 76 * 60.0, mu=MU_EARTH)
    np.testing.assert_allclose(v1, [2.058913, 2.915965, 0.0], atol=1e-5)
    np.testing.assert_allclose(v2, [-3.451565, 0.910315, 0.0], atol=1e-5)


def test_assignment_answer_unchanged():
    """The v1 assignment result must not move with v2 solver changes."""
    from assignment_answer import solve_assignment
    el = solve_assignment()["elements"]
    assert el["a"] == pytest.approx(28196.775768, abs=1e-5)
    assert el["e"] == pytest.approx(0.767944, abs=1e-6)
    assert el["i"] == pytest.approx(20.315296, abs=1e-6)


def _random_geometry(rng, scale, r_range, tof_range):
    r1 = scale * rng.uniform(*r_range) * np.array([1.0, 0.0, 0.0])
    th = np.radians(rng.uniform(5, 355))
    r2 = scale * rng.uniform(*r_range) * np.array(
        [np.cos(th), np.sin(th), rng.uniform(-0.3, 0.3)])
    return r1, r2, rng.uniform(*tof_range)


@pytest.mark.parametrize("mu, scale, r_range, tof_range", [
    (MU_EARTH, 6378.137, (1.1, 8.0), (600.0, 40000.0)),   # Earth orbits
    (MU_SUN, AU, (0.6, 2.5), (20 * DAY, 900 * DAY)),       # heliocentric
])
def test_lambert_solution_reaches_r2(mu, scale, r_range, tof_range):
    """Propagating (r1, v1) forward by dt must land on r2, and in < 1 rev."""
    rng = np.random.default_rng(42)
    for _ in range(300):
        r1, r2, dt = _random_geometry(rng, scale, r_range, tof_range)
        try:
            v1, v2 = lamsolbert(r1, r2, dt, mu=mu)
        except RuntimeError:
            continue  # dt too short for any conic: legitimately no solution
        r2_prop, v2_prop = propagate_rv(r1, v1, dt, mu)
        assert np.linalg.norm(r2_prop - r2) / np.linalg.norm(r2) < 1e-8
        assert np.linalg.norm(v2_prop - v2) / np.linalg.norm(v2) < 1e-6
        a = rv_to_elements(r1, v1, mu)["a"]
        if 0 < a < np.inf:
            assert dt < 2 * np.pi * np.sqrt(a**3 / mu), "not the zero-revolution solution"


def test_lambert_transfer_over_180_deg():
    """Earth (1 AU) to 1.5 AU, 250 deg ahead: long transfer, prograde."""
    r1 = np.array([AU, 0.0, 0.0])
    th = np.radians(250.0)
    r2 = 1.5 * AU * np.array([np.cos(th), np.sin(th), 0.0])
    v1, _ = lamsolbert(r1, r2, 400 * DAY, mu=MU_SUN)
    assert v1[1] > 0, "prograde transfer must move counter-clockwise"
    r2_prop, _ = propagate_rv(r1, v1, 400 * DAY, MU_SUN)
    np.testing.assert_allclose(r2_prop, r2, rtol=1e-9, atol=1.0)


# ----------------------------------------------------------------------
# Kepler / propagation
# ----------------------------------------------------------------------
def test_solve_kepler_residual():
    M = np.linspace(0, 2 * np.pi, 200, endpoint=False)  # 2 pi wraps to 0
    for e in (0.0, 0.1, 0.5, 0.9, 0.99, 0.999):
        E = solve_kepler(M, e)
        np.testing.assert_allclose(E - e * np.sin(E), M, atol=1e-11)


def test_true_from_mean_known_value():
    # Circular orbit: true anomaly == mean anomaly.
    assert true_from_mean(123.0, 0.0) == pytest.approx(123.0)
    # Vallado Example 2-1: M = 235.4 deg, e = 0.4 -> E = 220.512074 deg,
    # which gives nu = 2 atan(sqrt(1.4/0.6) tan(E/2)) = 207.6 deg (mod 360).
    E = np.degrees(solve_kepler(np.radians(235.4), 0.4))
    assert E == pytest.approx(220.512074, abs=1e-5)


def test_propagate_vallado_2_4():
    """Vallado Example 2-4 (Kepler problem): 40 minutes forward."""
    r0 = np.array([1131.340, -2282.343, 6672.423])
    v0 = np.array([-5.64305, 4.30333, 2.42879])
    r, v = propagate_rv(r0, v0, 40 * 60.0, MU_EARTH)
    np.testing.assert_allclose(r, [-4219.7527, 4363.0292, -3958.7666], atol=1e-2)
    np.testing.assert_allclose(v, [3.689866, -1.916735, -6.112511], atol=1e-5)


def test_propagate_round_trip_and_full_period():
    rng = np.random.default_rng(7)
    for _ in range(100):
        a = rng.uniform(7000, 60000)
        e = rng.uniform(0, 0.95)
        r0, v0 = coe_to_rv(a, e, rng.uniform(0, 180), rng.uniform(0, 360),
                           rng.uniform(0, 360), rng.uniform(0, 360), MU_EARTH)
        dt = rng.uniform(-50000, 50000)
        r1, v1 = propagate_rv(r0, v0, dt, MU_EARTH)
        # 1e-8 relative (~0.1 m at GEO distance) is the double-precision
        # floor of the Stumpff-function evaluation over a full revolution.
        r_back, v_back = propagate_rv(r1, v1, -dt, MU_EARTH)
        assert np.linalg.norm(r_back - r0) / np.linalg.norm(r0) < 1e-8
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH)
        r_per, _ = propagate_rv(r0, v0, period, MU_EARTH)
        assert np.linalg.norm(r_per - r0) / np.linalg.norm(r0) < 1e-8


def test_propagate_hyperbolic_matches_energy():
    r0 = np.array([7000.0, 0.0, 0.0])
    v0 = np.array([0.0, 12.0, 1.0])  # above escape speed
    r, v = propagate_rv(r0, v0, 20000.0, MU_EARTH)
    energy0 = v0 @ v0 / 2 - MU_EARTH / np.linalg.norm(r0)
    energy1 = v @ v / 2 - MU_EARTH / np.linalg.norm(r)
    assert energy1 == pytest.approx(energy0, rel=1e-8)
    h0 = np.cross(r0, v0)
    assert np.linalg.norm(np.cross(r, v) - h0) / np.linalg.norm(h0) < 1e-8


def test_coe_rv_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(200):
        a, e = rng.uniform(7000, 50000), rng.uniform(0.01, 0.9)
        i, raan, argp, nu = (rng.uniform(1, 179), rng.uniform(0, 360),
                             rng.uniform(0, 360), rng.uniform(0, 360))
        r, v = coe_to_rv(a, e, i, raan, argp, nu, MU_EARTH)
        el = rv_to_elements(r, v, MU_EARTH)
        assert el["a"] == pytest.approx(a, rel=1e-9)
        assert el["e"] == pytest.approx(e, rel=1e-8)
        assert el["i"] == pytest.approx(i, abs=1e-8)
        for key, want in (("raan", raan), ("argp", argp), ("nu", nu)):
            diff = (el[key] - want + 180) % 360 - 180
            assert abs(diff) < 1e-6, key


def test_position_at_agrees_with_propagate_rv():
    a, e, i, raan, argp, M0 = 1.8 * AU, 0.4, 12.0, 80.0, 250.0, 30.0
    epoch = 2461000.5
    r0, v0 = position_at(a, e, i, raan, argp, M0, epoch, epoch, MU_SUN)
    r1, _ = position_at(a, e, i, raan, argp, M0, epoch, epoch + 321.0, MU_SUN)
    r1_prop, _ = propagate_rv(r0, v0, 321.0 * DAY, MU_SUN)
    np.testing.assert_allclose(r1, r1_prop, rtol=1e-9)


def test_orbit_curve_passes_through_state():
    r0, v0 = coe_to_rv(20000.0, 0.3, 40.0, 10.0, 70.0, 123.0, MU_EARTH)
    pts = orbit_curve(r0, v0, MU_EARTH, n_points=3601)
    assert np.min(np.linalg.norm(pts.T - r0, axis=1)) < 50.0  # km, grid spacing


# ----------------------------------------------------------------------
# Planets (reference vectors: JPL Horizons, heliocentric ecliptic J2000,
# TDB, retrieved 2026-09-28)
# ----------------------------------------------------------------------
HORIZONS = {
    ("Earth", 2451545.0): [-26502576.9, 144693955.6, -170.4],
    ("Earth", 2461311.5): [149459377.9, 11823438.3, -1622.1],
    ("Mars", 2451545.0): [208048140.6, -2007052.6, -5156289.0],
    ("Mars", 2461311.5): [25051769.4, 231164558.9, 4230089.8],
    ("Venus", 2461311.5): [106503076.9, -21241222.5, -6436906.3],
}


@pytest.mark.parametrize("name, jd", list(HORIZONS))
def test_planet_positions_vs_horizons(name, jd):
    r, _ = planet_state(name, jd)
    assert np.linalg.norm(r - HORIZONS[(name, jd)]) < 0.001 * AU


def test_planet_state_vectorised_over_dates():
    jds = np.array([2451545.0, 2461311.5])
    r, v = planet_state("Earth", jds)
    assert r.shape == (2, 3) and v.shape == (2, 3)
    np.testing.assert_allclose(r[1], planet_state("Earth", jds[1])[0])
