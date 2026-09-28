"""
Approximate planet positions from JPL's Keplerian elements and rates.

Source: E. M. Standish, "Keplerian Elements for Approximate Positions of
the Major Planets", JPL/SSD, Table 1 (valid 1800 AD - 2050 AD). Elements
are referred to the mean ecliptic and equinox of J2000, the same frame as
JPL SBDB small-body elements, so planets and NEOs can be plotted and
differenced together directly.

Stated accuracy over that interval: under ~0.1 deg in heliocentric
longitude for the inner planets (Earth's error is at the 0.01-deg level),
which is ample for plotting and for porkchop plots (departure Δv
sensitivity to an Earth position error of this size is small).

"Earth" here is the Earth-Moon barycenter, as in Standish's table.
"""

from dataclasses import dataclass

import numpy as np

from core.constants import AU, MU_SUN, JD_J2000
from core.kepler import coe_to_rv, true_from_mean

# Standish Table 1 is fitted to 1800-01-01 .. 2050-12-31.
VALID_FROM_JD = 2378496.5
VALID_TO_JD = 2470171.5


def in_valid_range(jd):
    """True if every date in jd lies inside the ephemeris' fitted interval."""
    jd = np.asarray(jd, dtype=float)
    return bool(np.all((jd >= VALID_FROM_JD) & (jd <= VALID_TO_JD)))


@dataclass(frozen=True)
class PlanetElements:
    name: str
    color: str
    radius_km: float
    # (value at J2000, rate per Julian century):
    a_au: tuple
    e: tuple
    i_deg: tuple
    L_deg: tuple        # mean longitude
    varpi_deg: tuple    # longitude of perihelion
    node_deg: tuple     # longitude of ascending node


PLANETS = {
    p.name: p for p in (
        PlanetElements("Mercury", "#9ca3af", 2439.7,
                       (0.38709927, 0.00000037), (0.20563593, 0.00001906),
                       (7.00497902, -0.00594749), (252.25032350, 149472.67411175),
                       (77.45779628, 0.16047689), (48.33076593, -0.12534081)),
        PlanetElements("Venus", "#e6c27a", 6051.8,
                       (0.72333566, 0.00000390), (0.00677672, -0.00004107),
                       (3.39467605, -0.00078890), (181.97909950, 58517.81538729),
                       (131.60246718, 0.00268329), (76.67984255, -0.27769418)),
        PlanetElements("Earth", "#3b82f6", 6371.0,
                       (1.00000261, 0.00000562), (0.01671123, -0.00004392),
                       (-0.00001531, -0.01294668), (100.46457166, 35999.37244981),
                       (102.93768193, 0.32327364), (0.0, 0.0)),
        PlanetElements("Mars", "#dc6b4a", 3389.5,
                       (1.52371034, 0.00001847), (0.09339410, 0.00007882),
                       (1.84969142, -0.00813131), (-4.55343205, 19140.30268499),
                       (-23.94362959, 0.44441088), (49.55953891, -0.29257343)),
        PlanetElements("Jupiter", "#d9a066", 69911.0,
                       (5.20288700, -0.00011607), (0.04838624, -0.00013253),
                       (1.30439695, -0.00183714), (34.39644051, 3034.74612775),
                       (14.72847983, 0.21252668), (100.47390909, 0.20469106)),
    )
}


def planet_elements(name, jd):
    """
    Classical elements of `name` at Julian Date(s) `jd`:
    (a_km, e, i_deg, raan_deg, argp_deg, M_deg), each broadcast to jd's shape.
    """
    p = PLANETS[name]
    T = (np.asarray(jd, dtype=float) - JD_J2000) / 36525.0

    def at(pair):
        return pair[0] + pair[1] * T

    a = at(p.a_au) * AU
    e = at(p.e)
    i = at(p.i_deg)
    L = at(p.L_deg)
    varpi = at(p.varpi_deg)
    node = at(p.node_deg)
    argp = varpi - node
    M = np.mod(L - varpi, 360.0)
    return a, e, i, node, argp, M


def planet_state(name, jd):
    """
    Heliocentric ecliptic-J2000 position and velocity of `name` at jd.
    Returns (r, v) in km and km/s; shape (3,) for scalar jd, (N, 3) for N dates.
    """
    a, e, i, node, argp, M = planet_elements(name, jd)
    nu = true_from_mean(M, e)
    return coe_to_rv(a, e, i, node, argp, nu, MU_SUN)


def planet_orbit(name, jd, n_points=361):
    """Points (3, N) in km along the planet's osculating orbit at jd."""
    a, e, i, node, argp, _ = planet_elements(name, jd)
    nu = np.linspace(0.0, 360.0, n_points)
    r, _ = coe_to_rv(a, e, i, node, argp, nu, MU_SUN)
    return r.T
