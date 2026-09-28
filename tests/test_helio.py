"""
Tests for the heliocentric (NEO) scope: catalog, propagation of catalog
elements, porkchop computation.

Run: python -m pytest tests
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.constants import AU, DAY, MU_SUN                        # noqa: E402
from core.kepler import coe_to_rv, perifocal_basis, solve_kepler, position_at  # noqa: E402
from helio.neo_catalog import elements_of, filter_catalog, load_catalog  # noqa: E402
from helio.planets import in_valid_range, planet_state            # noqa: E402
from helio.porkchop import (closest_approach, compute_porkchop, refine_best,  # noqa: E402
                            solve_transfer, target_state)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


# ----------------------------------------------------------------------
# Catalog
# ----------------------------------------------------------------------
def test_catalog_loads_and_classes_follow_definitions(catalog):
    assert len(catalog) > 40000
    q, Q, a = catalog["q"], catalog["ad"], catalog["a"]
    cls = catalog["class"]
    assert ((a[cls == "ATE"] < 1.0) & (Q[cls == "ATE"] > 0.983)).all()
    assert ((a[cls == "APO"] > 1.0) & (q[cls == "APO"] < 1.017)).all()
    assert ((q[cls == "AMO"] > 1.017) & (q[cls == "AMO"] < 1.3)).all()
    assert (Q[cls == "IEO"] < 0.983).all()
    # PHA definition: MOID <= 0.05 AU and H <= 22. JPL sets the flag when the
    # object is classified; later H refinements have moved ~1% of PHAs just
    # past 22 (up to 22.6), so H is checked with that allowance.
    pha = catalog[catalog["pha"]]
    assert (pha["moid"] <= 0.05).all()
    assert (pha["H"] <= 22.6).all() and (pha["H"] <= 22.0).mean() > 0.98


def test_catalog_q_Q_consistent_with_a_e(catalog):
    np.testing.assert_allclose(catalog["q"], catalog["a"] * (1 - catalog["e"]), rtol=1e-6)


def test_filter_catalog(catalog):
    assert filter_catalog(catalog, "apophis")["pdes"].tolist() == ["99942"]
    atens = filter_catalog(catalog, classes=["ATE"], pha_only=True)
    assert len(atens) > 0 and set(atens["class"]) == {"ATE"} and atens["pha"].all()


# ----------------------------------------------------------------------
# Propagation of catalog elements vs JPL Horizons (heliocentric ecliptic
# J2000, TDB, JD 2461311.5 = 2026-09-29; retrieved 2026-09-28). Errors come
# from ignoring planetary perturbations over the ~111 days since the epoch.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pdes, ref", [
    ("433", [-16705511.3, -249535388.9, -29515951.9]),      # Eros
    ("99942", [-48134399.8, -120250236.2, 5280395.9]),      # Apophis
])
def test_catalog_propagation_vs_horizons(catalog, pdes, ref):
    row = catalog[catalog["pdes"] == pdes].iloc[0]
    r, _ = position_at(*elements_of(row), 2461311.5, MU_SUN)
    assert np.linalg.norm(r - ref) < 1e-4 * AU  # < 15,000 km


def test_perifocal_basis_matches_coe_to_rv():
    a, e, i, raan, argp, M = 2.1 * AU, 0.35, 17.0, 120.0, 40.0, 200.0
    E = solve_kepler(np.radians(M), e)
    nu = np.degrees(2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)))
    r_ref, _ = coe_to_rv(a, e, i, raan, argp, nu, MU_SUN)
    P, Q = perifocal_basis(i, raan, argp)
    r = a * (np.cos(E) - e) * P + a * np.sqrt(1 - e**2) * np.sin(E) * Q
    np.testing.assert_allclose(r, r_ref, rtol=1e-12)


# ----------------------------------------------------------------------
# Porkchop
# ----------------------------------------------------------------------
def test_porkchop_minimum_near_hohmann():
    """
    Target on a circular, coplanar 1.524 AU orbit (Mars-like). The best
    rendezvous should be close to the Hohmann transfer: v_inf 2.94 km/s at
    departure + 2.65 km/s at arrival = 5.59 km/s, TOF ~ 259 days. Earth's
    real orbit is slightly eccentric, so allow a small margin.
    """
    target = (1.524 * AU, 0.0, 0.0, 0.0, 0.0, 0.0, 2461000.5)
    pc = compute_porkchop(target, 2461000.5, 800, 150, 350, n_dep=60, n_tof=40)
    dep, tof, dv = pc.best()
    assert dv == pytest.approx(5.59, abs=0.25)
    assert tof == pytest.approx(259, abs=35)


def test_transfer_path_connects_earth_and_target(catalog):
    row = catalog[catalog["pdes"] == "99942"].iloc[0]
    tr = solve_transfer(elements_of(row), 2461500.5, 200.0)
    path = tr.path(50)
    np.testing.assert_allclose(path[0], tr.r1, atol=1.0)
    np.testing.assert_allclose(path[-1], tr.r2, rtol=1e-8, atol=100.0)
    assert tr.dv_total == pytest.approx(tr.vinf_dep + tr.vinf_arr)


def test_porkchop_matches_single_transfer(catalog):
    row = catalog[catalog["pdes"] == "433"].iloc[0]
    pc = compute_porkchop(elements_of(row), 2461400.5, 300, 100, 400, n_dep=6, n_tof=5)
    tr = solve_transfer(elements_of(row), pc.dep_jd[2], pc.tof_days[3])
    assert pc.c3[3, 2] == pytest.approx(tr.c3, rel=1e-9)
    assert pc.vinf_arr[3, 2] == pytest.approx(tr.vinf_arr, rel=1e-9)


def test_refine_best_improves_on_grid_and_nears_hohmann():
    target = (1.524 * AU, 0.0, 0.0, 0.0, 0.0, 0.0, 2461000.5)
    pc = compute_porkchop(target, 2461000.5, 800, 150, 350, n_dep=30, n_tof=20)
    grid = pc.best()
    dep, tof, dv = refine_best(target, pc)
    assert dv <= grid[2]
    assert dv == pytest.approx(5.59, abs=0.1)   # Hohmann: 2.94 + 2.65 km/s
    assert tof == pytest.approx(259, abs=15)


def test_closest_approach_apophis_2029(catalog):
    """
    Two-body model of Apophis' 2029 Earth flyby. Actual: ~38,000 km from
    Earth's centre on 2029-04-13 21:46 UTC. Ignoring Earth's gravity on the
    asteroid (and measuring to the Earth-Moon barycentre) our model gives
    ~30,600 km a few hours later; guard that result against regressions and
    check the refinement against a brute-force scan.
    """
    row = catalog[catalog["pdes"] == "99942"].iloc[0]
    el = elements_of(row)
    jd, dist = closest_approach(el, 2461311.5, 2461311.5 + 4 * 365)
    assert jd == pytest.approx(2462240.4, abs=0.5)     # 2029-04-13/14
    assert 20000 < dist < 45000
    fine = np.arange(jd - 0.05, jd + 0.05, 1e-4)
    d = np.linalg.norm(target_state(el, fine)[0] - planet_state("Earth", fine)[0], axis=-1)
    assert dist <= d.min() + 1.0


def test_ephemeris_validity_range():
    assert in_valid_range([2461311.5, 2470000.5])
    assert not in_valid_range(2470172.5)   # 2051-01-01
    assert not in_valid_range(2378495.5)   # 1799-12-31
