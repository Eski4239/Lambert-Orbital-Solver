"""
Orbit-determination pipeline shared by the UI, scripts and tests.

observations (UTC, Az, El, Range) + station  ->  ECI positions  ->
Lambert (lamsolbert) between a chosen pair  ->  orbital elements, derived
quantities, and a consistency check against every other observation.

No UI code here: app/ only formats what this returns.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from core.constants import MU_EARTH, OMEGA_EARTH, R_EARTH
from core.elements import rv_to_elements
from core.frames import aer_to_eci
from core.kepler import propagate_rv
from core.lamsolbert import lamsolbert
from core.time_utils import gmst_degrees


@dataclass
class Observation:
    time: datetime  # timezone-aware UTC
    az_deg: float
    el_deg: float
    range_km: float


@dataclass
class OrbitSolution:
    r1: np.ndarray
    v1: np.ndarray
    r2: np.ndarray
    v2: np.ndarray
    tof_s: float
    epoch: datetime | None      # UTC time of r1 (None if unknown)
    gmst0_deg: float            # GMST at epoch (0 if epoch unknown)
    elements: dict              # from rv_to_elements
    period_s: float | None      # None for open (hyperbolic/parabolic) orbits
    rp_km: float                # periapsis radius
    ra_km: float | None         # apoapsis radius, None for open orbits
    warnings: list = field(default_factory=list)

    @property
    def is_closed(self):
        return self.period_s is not None

    @property
    def perigee_alt_km(self):
        return self.rp_km - R_EARTH

    @property
    def apogee_alt_km(self):
        return None if self.ra_km is None else self.ra_km - R_EARTH

    def gmst_at(self, t_s):
        """GMST (deg) t_s seconds after the epoch (sidereal rate)."""
        return self.gmst0_deg + np.degrees(OMEGA_EARTH * np.asarray(t_s))

    def state_at(self, t_s):
        """ECI state t_s seconds after the epoch (two-body propagation)."""
        return propagate_rv(self.r1, self.v1, float(t_s), MU_EARTH)


def observations_to_eci(observations, station_ecef_km):
    """ECI position (km) of each observation, each with its own GMST."""
    return [aer_to_eci(o.az_deg, o.el_deg, o.range_km, station_ecef_km,
                       gmst_degrees(o.time)) for o in observations]


def solve_orbit(r1, r2, tof_s, prograde=True, epoch=None, mu=MU_EARTH):
    """Lambert between two ECI positions, plus elements and derived values."""
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    if tof_s <= 0:
        raise ValueError("Time of flight must be positive: the second point "
                         "must come after the first.")
    v1, v2 = lamsolbert(r1, r2, tof_s, mu=mu, prograde=prograde)
    el = rv_to_elements(r1, v1, mu=mu)

    a, e = el["a"], el["e"]
    p = el["h"] ** 2 / mu
    rp = p / (1 + e)
    if np.isfinite(a) and e < 1:
        period = 2 * np.pi * np.sqrt(a**3 / mu)
        ra = a * (1 + e)
    else:
        period, ra = None, None

    warnings = []
    if rp < R_EARTH:
        warnings.append(f"Perigee is {R_EARTH - rp:,.0f} km below the Earth's "
                        f"surface: this trajectory intersects the Earth.")
    if period is None:
        warnings.append("Open orbit (e ≥ 1): the object escapes Earth.")

    return OrbitSolution(
        r1=r1, v1=v1, r2=r2, v2=v2, tof_s=float(tof_s), epoch=epoch,
        gmst0_deg=gmst_degrees(epoch) if epoch is not None else 0.0,
        elements=el, period_s=period, rp_km=rp, ra_km=ra, warnings=warnings,
    )


def solve_from_observations(observations, station_ecef_km, i=0, j=1, prograde=True):
    """
    Solve using observations i and j, then report how far the resulting
    orbit passes from every observation (km) at that observation's time.
    Returns (solution, residuals_km list aligned with `observations`).
    """
    if i == j:
        raise ValueError("Choose two different observations.")
    if not (0 <= i < len(observations) and 0 <= j < len(observations)):
        raise ValueError("Observation choice is out of range.")
    first, second = sorted((i, j), key=lambda k: observations[k].time)
    positions = observations_to_eci(observations, station_ecef_km)
    t0 = observations[first].time
    tof = (observations[second].time - t0).total_seconds()
    sol = solve_orbit(positions[first], positions[second], tof,
                      prograde=prograde, epoch=t0)

    residuals = []
    for obs, r in zip(observations, positions):
        r_pred, _ = sol.state_at((obs.time - t0).total_seconds())
        residuals.append(float(np.linalg.norm(r_pred - r)))
    return sol, residuals
