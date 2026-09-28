"""
LamSolbert: Lambert's problem solver using the universal-variable formulation.

Given two position vectors r1, r2 and a time-of-flight dt between them,
find the velocity vectors v1, v2 for the orbit connecting them.

This uses the universal-variable approach (works for elliptical, parabolic,
and hyperbolic transfer orbits alike, unlike the classical p-iteration
method which can fail to converge near certain geometries), following
Vallado, "Fundamentals of Astrodynamics and Applications", Algorithm 58.

Convention: "prograde" (default) means the transfer sweeps through the
short way (delta true anomaly < 180 deg) when the orbit is assumed
counter-clockwise as seen from +Z; this is the standard short-way transfer.
Long-way transfer (prograde=False conceptually swaps to dtheta > 180 deg)
is not implemented -- only short-way, per the assignment's "sufficient"
allowance.
"""

import numpy as np


def _stumpff_C(z: float) -> float:
    """Stumpff function C(z)."""
    if z > 1e-8:
        sz = np.sqrt(z)
        return (1 - np.cos(sz)) / z
    elif z < -1e-8:
        sz = np.sqrt(-z)
        return (1 - np.cosh(sz)) / z
    else:
        # Series expansion near z = 0 avoids 0/0.
        return 1 / 2 - z / 24 + z**2 / 720


def _stumpff_S(z: float) -> float:
    """Stumpff function S(z)."""
    if z > 1e-8:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / sz**3
    elif z < -1e-8:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / sz**3
    else:
        return 1 / 6 - z / 120 + z**2 / 5040


def lamsolbert(r1_vec, r2_vec, dt, mu=398600.4418, prograde=True,
                tol=1e-8, max_iter=100):
    """
    LamSolbert: solve Lambert's problem via the universal-variable formulation.

    Parameters
    ----------
    r1_vec, r2_vec : array-like, shape (3,)
        Initial and final position vectors [km], same frame (e.g. ECI).
    dt : float
        Time of flight from r1 to r2 [s]. Must be > 0.
    mu : float
        Gravitational parameter [km^3/s^2]. Defaults to Earth.
    prograde : bool
        If True, assumes the transfer angle dtheta is measured such that
        the motion is counter-clockwise about +Z (short-way transfer for
        the "prograde" sense). If False, uses the retrograde sense
        (dtheta = 360 - dtheta). Long-way (dtheta > 180 within the same
        sense) is not supported.
    tol : float
        Convergence tolerance on the universal Kepler time equation [s].
    max_iter : int
        Maximum Newton iterations before falling back to bisection.

    Returns
    -------
    v1_vec, v2_vec : np.ndarray, shape (3,)
        Velocity vectors [km/s] at r1 and r2.
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    if dt <= 0:
        raise ValueError("Time of flight dt must be positive.")

    r1 = np.linalg.norm(r1_vec)
    r2 = np.linalg.norm(r2_vec)

    cross_r1r2 = np.cross(r1_vec, r2_vec)
    cos_dtheta = np.dot(r1_vec, r2_vec) / (r1 * r2)
    cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)

    # Determine transfer angle direction based on prograde/retrograde sense
    # (Vallado Sec. 5.3): for prograde, dtheta is in [0, 360) increasing
    # counter-clockwise about +Z; if cross_r1r2_z < 0 the "short way" is
    # actually the reflex angle.
    if prograde:
        if cross_r1r2[2] >= 0:
            dtheta = np.arccos(cos_dtheta)
        else:
            dtheta = 2 * np.pi - np.arccos(cos_dtheta)
    else:
        if cross_r1r2[2] < 0:
            dtheta = np.arccos(cos_dtheta)
        else:
            dtheta = 2 * np.pi - np.arccos(cos_dtheta)

    # dtheta == 0 exactly (e.g. r1 == r2) makes 1 - cos(dtheta) exactly 0,
    # which numpy would otherwise report as a "divide by zero" /"invalid
    # value" RuntimeWarning on this line -- that is an expected, handled
    # edge case (caught by the isfinite check just below), not a real
    # numerical problem, so it's suppressed locally rather than left to
    # print a spurious warning on every such call.
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.sin(dtheta) * np.sqrt(r1 * r2 / (1 - np.cos(dtheta)))

    # When dtheta is exactly 0 (e.g. r1 == r2), 1 - cos(dtheta) is exactly
    # 0, so the sqrt(...) term is +inf while sin(dtheta) is exactly 0 --
    # this is a 0 * inf product, which numpy evaluates to NaN rather than
    # 0. The original "if A == 0" check below does not catch this (NaN ==
    # 0 is False), so it used to fall through into the root-finder, which
    # would then fail with an unrelated-looking "failed to converge"
    # RuntimeError instead of this function's own clear, specific
    # degenerate-geometry error. Checking for non-finite A up front closes
    # that gap.
    if not np.isfinite(A) or A == 0:
        raise ValueError(
            "Lambert geometry degenerate (A=0 or undefined): transfer angle is "
            "0 or 180 deg (e.g. r1 and r2 are parallel/antiparallel, or "
            "identical), no unique solution exists."
        )

    def y_of_z(z):
        return r1 + r2 + A * (z * _stumpff_S(z) - 1) / np.sqrt(_stumpff_C(z))

    def time_of_flight(z):
        y = y_of_z(z)
        if y < 0:
            return None  # invalid region for this A; caller must adjust bracket
        chi = np.sqrt(y / _stumpff_C(z))
        t = (chi**3 * _stumpff_S(z) + A * np.sqrt(y)) / np.sqrt(mu)
        return t

    # --- Find a bracket [z_lo, z_hi] with time_of_flight(z_lo) < dt < time_of_flight(z_hi) ---
    # z=0 corresponds to the parabolic transfer time; z>0 ellipse, z<0 hyperbola.
    z_lo, z_hi = -4 * np.pi**2, 4 * np.pi**2
    # Expand z_hi until y(z_hi) stays positive and t(z_hi) exceeds dt (or cap tries).
    z = 0.0
    t_z = time_of_flight(z)
    tries = 0
    while t_z is None or t_z > dt:
        # Need smaller (more negative or less positive) z: shrink toward hyperbolic side.
        z_hi = z if t_z is not None and t_z > dt else z_hi
        z -= 0.1
        t_z = time_of_flight(z)
        tries += 1
        if tries > 1000 or z < z_lo:
            z = z_lo
            break

    # Newton's method on z using dt(z) - dt = 0, with derivative from Vallado Eq. 5-43/5-44.
    z = 0.0
    converged = False
    for _ in range(max_iter):
        y = y_of_z(z)
        if y < 0:
            z += 0.1
            continue
        Cz = _stumpff_C(z)
        Sz = _stumpff_S(z)
        chi = np.sqrt(y / Cz)
        t = (chi**3 * Sz + A * np.sqrt(y)) / np.sqrt(mu)

        if abs(t - dt) < tol:
            converged = True
            break

        if abs(z) > 1e-6:
            dCdz = (1 - z * Sz - 2 * Cz) / (2 * z)
            dSdz = (Cz - 3 * Sz) / (2 * z)
        else:
            # Series-based derivatives near z = 0.
            dCdz = -1 / 24
            dSdz = -1 / 120

        dtdz = (chi**3 * (dSdz - 3 * Sz * dCdz / (2 * Cz)) +
                (A / 8) * (3 * Sz * np.sqrt(y) / Cz + A / chi)) / np.sqrt(mu)

        if dtdz == 0 or not np.isfinite(dtdz):
            break

        z_new = z - (t - dt) / dtdz
        z = z_new

    if not converged:
        # Fallback: robust bisection on time_of_flight(z) - dt.
        z_lo, z_hi = -4 * np.pi**2, 4 * np.pi**2 - 1e-6
        # Search for a valid bracket where f is defined and crosses dt.
        zs = np.linspace(z_lo, z_hi, 2000)
        bracket = None
        prev_z, prev_t = None, None
        for zc in zs:
            tc = time_of_flight(zc)
            if tc is None:
                prev_z, prev_t = zc, tc
                continue
            if prev_t is not None and (prev_t - dt) * (tc - dt) < 0:
                bracket = (prev_z, zc)
                break
            prev_z, prev_t = zc, tc
        if bracket is None:
            raise RuntimeError("Lambert solver failed to converge (no valid bracket found).")
        z_lo, z_hi = bracket
        for _ in range(200):
            z_mid = 0.5 * (z_lo + z_hi)
            t_mid = time_of_flight(z_mid)
            if t_mid is None:
                z_lo = z_mid
                continue
            if abs(t_mid - dt) < tol:
                z = z_mid
                converged = True
                break
            if t_mid < dt:
                z_lo = z_mid
            else:
                z_hi = z_mid
        z = 0.5 * (z_lo + z_hi)

    y = y_of_z(z)
    Cz = _stumpff_C(z)
    f = 1 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r2

    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (gdot * r2_vec - r1_vec) / g

    return v1_vec, v2_vec
