"""
Porkchop plots: Earth -> target transfers over a grid of departure dates
and times of flight, each solved with the project's Lambert solver
(core.lamsolbert) using the Sun's gravitational parameter.

For each cell:
    r1, vE = Earth's heliocentric state at departure (helio.planets)
    r2, vT = target's state at arrival (two-body propagation of its elements)
    v1, v2 = lamsolbert(r1, r2, tof, mu=MU_SUN)
    departure v-infinity = |v1 - vE|  ->  launch energy C3 = v_inf^2
    arrival v-infinity   = |v2 - vT|  (relative speed at the asteroid)
    total delta-v        = departure + arrival v-infinity, i.e. an impulsive
                           rendezvous measured from/to hyperbolic excess
                           speed (Earth escape and capture burns excluded).

Cells where Lambert has no zero-revolution solution (or the transfer angle
is ~180 deg, where the transfer plane is undefined) are NaN.
"""

from dataclasses import dataclass

import numpy as np

from core.constants import DAY, MU_SUN
from core.kepler import position_at, propagate_rv
from core.lamsolbert import lamsolbert
from helio.planets import planet_state


@dataclass
class Porkchop:
    dep_jd: np.ndarray        # (n_dep,)
    tof_days: np.ndarray      # (n_tof,)
    c3: np.ndarray            # (n_tof, n_dep) km^2/s^2
    vinf_arr: np.ndarray      # (n_tof, n_dep) km/s
    dv_total: np.ndarray      # (n_tof, n_dep) km/s

    def best(self, metric="dv_total"):
        """(dep_jd, tof_days, value) of the grid minimum of `metric`."""
        z = getattr(self, metric)
        if np.all(np.isnan(z)):
            return None
        j, i = np.unravel_index(np.nanargmin(z), z.shape)
        return float(self.dep_jd[i]), float(self.tof_days[j]), float(z[j, i])


def target_state(elements, jd):
    """Heliocentric state of a catalog object: elements = (a_km, e, i, raan, argp, M0, epoch)."""
    a, e, i, raan, argp, M0, epoch = elements
    return position_at(a, e, i, raan, argp, M0, epoch, jd, MU_SUN)


def compute_porkchop(elements, dep_start_jd, dep_span_days, tof_min_days, tof_max_days,
                     n_dep=70, n_tof=70, prograde=True):
    dep = np.linspace(dep_start_jd, dep_start_jd + dep_span_days, n_dep)
    tof = np.linspace(tof_min_days, tof_max_days, n_tof)
    r_e, v_e = planet_state("Earth", dep)                      # (n_dep, 3)
    r_t, v_t = target_state(elements, dep[None, :] + tof[:, None])  # (n_tof, n_dep, 3)

    c3 = np.full((n_tof, n_dep), np.nan)
    vinf_arr = np.full((n_tof, n_dep), np.nan)
    for j in range(n_tof):
        dt = tof[j] * DAY
        for i in range(n_dep):
            try:
                v1, v2 = lamsolbert(r_e[i], r_t[j, i], dt, mu=MU_SUN, prograde=prograde)
            except (ValueError, RuntimeError):
                continue
            c3[j, i] = np.sum((v1 - v_e[i]) ** 2)
            vinf_arr[j, i] = np.linalg.norm(v2 - v_t[j, i])
    return Porkchop(dep, tof, c3, vinf_arr, np.sqrt(c3) + vinf_arr)


@dataclass
class Transfer:
    dep_jd: float
    tof_days: float
    r1: np.ndarray
    v1: np.ndarray
    r2: np.ndarray
    v2: np.ndarray
    v_earth: np.ndarray
    v_target: np.ndarray

    @property
    def vinf_dep(self):
        return float(np.linalg.norm(self.v1 - self.v_earth))

    @property
    def c3(self):
        return self.vinf_dep ** 2

    @property
    def vinf_arr(self):
        return float(np.linalg.norm(self.v2 - self.v_target))

    @property
    def dv_total(self):
        return self.vinf_dep + self.vinf_arr

    def path(self, n=150):
        """Heliocentric points (n, 3) km along the transfer arc."""
        ts = np.linspace(0.0, self.tof_days * DAY, n)
        return np.array([propagate_rv(self.r1, self.v1, t, MU_SUN)[0] for t in ts])


def solve_transfer(elements, dep_jd, tof_days, prograde=True):
    """One Earth -> target transfer, for the transfer the user picks on the plot."""
    r1, v_e = planet_state("Earth", dep_jd)
    r2, v_t = target_state(elements, dep_jd + tof_days)
    v1, v2 = lamsolbert(r1, r2, tof_days * DAY, mu=MU_SUN, prograde=prograde)
    return Transfer(dep_jd, tof_days, r1, v1, r2, v2, v_e, v_t)


def refine_best(elements, pc, metric="dv_total"):
    """
    Continuous minimum of `metric` near the grid minimum: Nelder-Mead over
    (departure date, time of flight), starting from the best grid cell with
    a simplex one grid step wide, kept inside the grid's bounds. The grid
    alone is only as fine as its spacing (~10 days x ~7 days by default).
    Returns (dep_jd, tof_days, value).
    """
    from scipy.optimize import minimize

    best = pc.best(metric)
    if best is None:
        return None
    d0, t0, v0 = best
    dd = pc.dep_jd[1] - pc.dep_jd[0] if len(pc.dep_jd) > 1 else 1.0
    dt = pc.tof_days[1] - pc.tof_days[0] if len(pc.tof_days) > 1 else 1.0
    lo_d, hi_d, lo_t, hi_t = pc.dep_jd[0], pc.dep_jd[-1], pc.tof_days[0], pc.tof_days[-1]

    def f(x):
        d, t = d0 + x[0], x[1]  # departure as days from the grid best (see closest_approach)
        if not (lo_d <= d <= hi_d and lo_t <= t <= hi_t):
            return 1e9
        try:
            return getattr(solve_transfer(elements, d, t), metric)
        except (ValueError, RuntimeError):
            return 1e9

    simplex = [[0.0, t0], [dd, t0], [0.0, t0 + dt]]
    res = minimize(f, [0.0, t0], method="Nelder-Mead",
                   options={"initial_simplex": simplex, "xatol": 0.05, "fatol": 1e-4, "maxiter": 150})
    if res.fun < v0:
        return float(d0 + res.x[0]), float(res.x[1]), float(res.fun)
    return best


def closest_approach(elements, jd_start, jd_end, step_days=1.0):
    """
    Minimum distance between the target and Earth (Earth-Moon barycentre)
    over [jd_start, jd_end]: sampled every step_days, then refined around
    the smallest sample with a bounded scalar minimisation.
    Returns (jd, distance_km).
    """
    from scipy.optimize import minimize_scalar

    def dist(jd):
        return np.linalg.norm(target_state(elements, jd)[0] - planet_state("Earth", jd)[0], axis=-1)

    jds = np.arange(jd_start, jd_end + step_days, step_days)
    d = dist(jds)
    k = int(np.argmin(d))
    # Optimise over days from the best sample, not raw JD: the bounded
    # method's tolerance includes ~sqrt(eps)*|x|, i.e. ~0.04 days at
    # JD ~ 2.46e6, long enough for a close flyby to move by ~20,000 km.
    ref = jds[k]
    lo, hi = jds[max(k - 1, 0)] - ref, jds[min(k + 1, len(jds) - 1)] - ref
    res = minimize_scalar(lambda x: float(dist(ref + x)), bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-6})
    if res.fun < d[k]:
        return float(ref + res.x), float(res.fun)
    return float(ref), float(d[k])
