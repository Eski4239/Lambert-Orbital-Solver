"""
Two-body (Keplerian) propagation: the inverse direction of elements.py.

- solve_kepler / true_from_mean: Kepler's equation M = E - e sin E.
- coe_to_rv: classical elements -> position/velocity (Vallado Alg. 10,
  "coe2rv"), vectorised so a whole NEO catalog can be positioned at once.
- position_at: elements + mean anomaly at epoch -> position at any date.
- propagate_rv: universal-variable propagation of a state vector (Vallado
  Alg. 8, "kepler"); works for elliptic, parabolic and hyperbolic orbits.
  Used to check Lambert solutions independently and to animate orbits.
- orbit_curve: points along the full ellipse (or the hyperbola's arc)
  through a state vector, for plotting.

Angles are degrees at the API boundary, lengths km, times s, unless noted.
"""

import numpy as np

from core.lamsolbert import _stumpff_C, _stumpff_S


def solve_kepler(M, e, tol=1e-12, max_iter=50):
    """
    Solve Kepler's equation M = E - e sin(E) for the eccentric anomaly E
    (radians), elliptic orbits only (0 <= e < 1). M and e may be arrays.

    Newton iteration from E0 = M + e sin(M) (or pi for e > 0.8, where that
    guess can overshoot), which converges in a handful of steps for every
    e < 1.
    """
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    if np.any(e < 0) or np.any(e >= 1):
        raise ValueError("solve_kepler requires 0 <= e < 1 (elliptic orbits).")

    M = np.mod(M, 2 * np.pi)
    E = np.where(e > 0.8, np.pi, M + e * np.sin(M))
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            return E
    raise RuntimeError("Kepler's equation did not converge.")


def true_from_mean(M_deg, e):
    """Mean anomaly (deg) -> true anomaly (deg), elliptic orbits."""
    E = solve_kepler(np.radians(M_deg), e)
    e = np.asarray(e, dtype=float)
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return np.degrees(np.mod(nu, 2 * np.pi))


def _perifocal_to_inertial(i, raan, argp):
    """
    Rotation matrices (shape (..., 3, 3)) taking perifocal PQW vectors to
    the inertial frame: R = Rz(-raan) Rx(-i) Rz(-argp). Angles in radians.
    """
    ci, si = np.cos(i), np.sin(i)
    cO, sO = np.cos(raan), np.sin(raan)
    cw, sw = np.cos(argp), np.sin(argp)
    R = np.empty(np.broadcast(i, raan, argp).shape + (3, 3))
    R[..., 0, 0] = cO * cw - sO * sw * ci
    R[..., 0, 1] = -cO * sw - sO * cw * ci
    R[..., 0, 2] = sO * si
    R[..., 1, 0] = sO * cw + cO * sw * ci
    R[..., 1, 1] = -sO * sw + cO * cw * ci
    R[..., 1, 2] = -cO * si
    R[..., 2, 0] = sw * si
    R[..., 2, 1] = cw * si
    R[..., 2, 2] = ci
    return R


def coe_to_rv(a, e, i, raan, argp, nu, mu):
    """
    Classical orbital elements -> inertial position and velocity.

    a in km (negative for hyperbolic orbits), angles in degrees. Every
    argument may be a scalar or an array (broadcast together). Returns
    (r, v) with shape (..., 3) in km and km/s, in whatever inertial frame
    the angles were measured in (ECI for Earth orbits, ecliptic J2000 for
    heliocentric ones).
    """
    a, e = np.asarray(a, dtype=float), np.asarray(e, dtype=float)
    i, raan, argp, nu = (np.radians(np.asarray(x, dtype=float)) for x in (i, raan, argp, nu))

    p = a * (1 - e**2)
    if np.any(p <= 0):
        raise ValueError("Semi-latus rectum must be positive (check sign of a vs e).")

    r_mag = p / (1 + e * np.cos(nu))
    r_pqw = np.stack([r_mag * np.cos(nu), r_mag * np.sin(nu), np.zeros_like(r_mag)], axis=-1)
    sq = np.sqrt(mu / p)
    v_pqw = np.stack([-sq * np.sin(nu), sq * (e + np.cos(nu)), np.zeros_like(r_mag)], axis=-1)

    R = _perifocal_to_inertial(i, raan, argp)
    r = np.einsum("...ij,...j->...i", R, r_pqw)
    v = np.einsum("...ij,...j->...i", R, v_pqw)
    return r, v


def position_at(a, e, i, raan, argp, M0, epoch_jd, jd, mu):
    """
    Position/velocity of elliptic bodies at Julian Date `jd`, given their
    elements and mean anomaly M0 (deg) at `epoch_jd`. Vectorised over
    bodies (arrays of elements) and/or dates (broadcasting rules apply).
    """
    a = np.asarray(a, dtype=float)
    n = np.sqrt(mu / a**3)  # rad/s
    M = np.radians(M0) + n * (np.asarray(jd) - np.asarray(epoch_jd)) * 86400.0
    nu = true_from_mean(np.degrees(M), e)
    return coe_to_rv(a, e, i, raan, argp, nu, mu)


def propagate_rv(r0, v0, dt, mu, tol=1e-10, max_iter=100):
    """
    Propagate a state vector by dt seconds (may be negative) using the
    universal-variable formulation and Lagrange f/g coefficients.

    Returns (r, v) as np.ndarray, shape (3,).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    if dt == 0:
        return r0.copy(), v0.copy()

    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    sqmu = np.sqrt(mu)
    rv = np.dot(r0, v0)
    alpha = 2 / r0_mag - v0_mag**2 / mu  # 1/a

    def time_and_radius(chi):
        psi = chi**2 * alpha
        C, S = _stumpff_C(psi), _stumpff_S(psi)
        r = chi**2 * C + rv / sqmu * chi * (1 - psi * S) + r0_mag * (1 - psi * C)
        t = (chi**3 * S + rv / sqmu * chi**2 * C + r0_mag * chi * (1 - psi * S)) / sqmu
        return t, r

    # t(chi) is strictly increasing (dt/dchi = r / sqrt(mu) > 0), so the root
    # is bracketed and solved by Newton with a bisection safeguard. Plain
    # Newton (Vallado Alg. 8) can overshoot badly on near-parabolic orbits
    # that pass very close to the central body.
    if alpha > 1e-12:
        # Whole revolutions don't change the state; one revolution spans
        # chi in [0, 2 pi sqrt(a)].
        period = 2 * np.pi / np.sqrt(mu * alpha**3)
        dt = np.mod(dt, period)
        if dt == 0:
            return r0.copy(), v0.copy()
        lo, hi = 0.0, 2 * np.pi / np.sqrt(alpha)
        chi = sqmu * dt * alpha
    else:
        lo, hi = 0.0, np.sign(dt) * np.sqrt(abs(dt) * sqmu / r0_mag)
        while (time_and_radius(hi)[0] - dt) * np.sign(dt) < 0:
            lo, hi = hi, 2 * hi
        lo, hi = min(lo, hi), max(lo, hi)
        chi = 0.5 * (lo + hi)
    if not lo < chi < hi:
        chi = 0.5 * (lo + hi)

    for _ in range(max_iter + 100):
        t, r = time_and_radius(chi)
        if t < dt:
            lo = chi
        else:
            hi = chi
        chi_new = chi + (dt - t) * sqmu / r
        if not lo < chi_new < hi:
            chi_new = 0.5 * (lo + hi)
        # Relative test: heliocentric chi is ~1e5 sqrt(km), where an
        # absolute 1e-10 is below double precision and never satisfied.
        if abs(chi_new - chi) < tol * max(1.0, abs(chi)):
            chi = chi_new
            break
        chi = chi_new
    else:
        raise RuntimeError("Universal-variable propagation did not converge.")

    psi = chi**2 * alpha
    C, S = _stumpff_C(psi), _stumpff_S(psi)
    f = 1 - chi**2 / r0_mag * C
    g = dt - chi**3 / sqmu * S
    r_vec = f * r0 + g * v0
    r_mag = np.linalg.norm(r_vec)
    gdot = 1 - chi**2 / r_mag * C
    fdot = sqmu / (r_mag * r0_mag) * chi * (psi * S - 1)
    v_vec = fdot * r0 + gdot * v0
    return r_vec, v_vec


def orbit_curve(r0, v0, mu, n_points=361, max_radius=None):
    """
    Points (shape (3, N)) along the conic through state (r0, v0).

    Built from the eccentricity vector (periapsis direction) and angular
    momentum vector (orbit normal) so it is always well-defined, including
    for near-circular and near-equatorial orbits where RAAN/argp are not.
    Elliptic orbits give the full closed ellipse. Hyperbolic/parabolic
    orbits give the branch out to `max_radius` (default 4 |r0|).
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    r0_mag = np.linalg.norm(r0)
    h_vec = np.cross(r0, v0)
    w_hat = h_vec / np.linalg.norm(h_vec)
    e_vec = np.cross(v0, h_vec) / mu - r0 / r0_mag
    e = np.linalg.norm(e_vec)
    p = np.dot(h_vec, h_vec) / mu

    if e > 1e-8:
        p_hat = e_vec / e
    else:
        p_hat = r0 / r0_mag
    q_hat = np.cross(w_hat, p_hat)

    if e < 1:
        nu = np.linspace(0, 2 * np.pi, n_points)
    else:
        r_cap = max_radius if max_radius is not None else 4 * r0_mag
        # r(nu) = p / (1 + e cos nu) = r_cap  ->  cos nu = (p/r_cap - 1)/e
        nu_max = np.arccos(np.clip((p / r_cap - 1) / e, -1, 1))
        nu = np.linspace(-nu_max, nu_max, n_points)

    r = p / (1 + e * np.cos(nu))
    return (r * np.cos(nu))[None, :] * p_hat[:, None] + (r * np.sin(nu))[None, :] * q_hat[:, None]
