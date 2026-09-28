"""
LamSolbert: Lambert's problem solver using the universal-variable formulation.

Given two position vectors r1, r2 and a time-of-flight dt between them,
find the velocity vectors v1, v2 for the orbit connecting them.

This uses the universal-variable approach (works for elliptical, parabolic,
and hyperbolic transfer orbits alike, unlike the classical p-iteration
method which can fail to converge near certain geometries), following
Vallado, "Fundamentals of Astrodynamics and Applications", Algorithm 58.

Convention: the direction of motion picks the transfer angle. "prograde"
(default) means counter-clockwise as seen from +Z, so the transfer angle
dtheta is whichever of the two geometric angles between r1 and r2 that
sense of motion sweeps: less than 180 deg when (r1 x r2)_z >= 0, more than
180 deg otherwise (Curtis, Algorithm 5.2). prograde=False takes the
clockwise sense. Transfers over 180 deg are therefore handled. Only the
zero-revolution solution is found (the transfer completes less than one
revolution); multi-revolution Lambert is not implemented.

Verification: Vallado Example 5-5 (verify.py), plus tests/test_core.py,
which propagates each solution forward by dt with core.kepler and checks
it reaches r2, over random Earth-orbit and heliocentric geometries.
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

    # A time tolerance below the spacing of doubles near dt can never be
    # met: heliocentric transfers have dt ~ 1e7-1e8 s, where that spacing
    # is already ~1e-8 s, so Newton would never "converge" and every call
    # fell through to the (correct but ~50x slower) bisection fallback.
    # Floor tol at a few ulps of dt. For Earth-orbit cases (dt ~ 1e4 s)
    # the floor is ~1e-11 s, below the default 1e-8, so results there are
    # unchanged.
    tol = max(tol, 8 * np.finfo(float).eps * dt)

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

    # --- Root-find t(z) = dt: safeguarded Newton (Newton-bisection) ---
    # z = 0 is the parabolic transfer, z > 0 elliptic, z < 0 hyperbolic.
    # For the single-revolution transfer t(z) increases monotonically with
    # z, from ~0 (the edge of the y(z) < 0 region, which only exists when
    # A < 0, i.e. transfer angles > 180 deg) up to +inf as z -> 4 pi^2.
    # So the root is kept inside a bracket [z_lo, z_hi] with
    # t(z_lo) < dt < t(z_hi), treating y(z) < 0 as "t below dt". Each step
    # takes the Newton update (derivative from Vallado Eq. 5-43/5-44) when it
    # stays inside the bracket, and bisects otherwise. This finds the same
    # root as plain Newton, but cannot diverge or stall, which matters for
    # transfers > 180 deg: there y(0) < 0 and the old fixed z += 0.1 walk
    # used up the iteration budget, falling back to a slow grid search.
    def residual(z):
        t = time_of_flight(z)
        return -np.inf if t is None else t - dt

    # Approach the z = 4 pi^2 asymptote in decades: getting too close makes
    # 1 - cos(sqrt(z)) round to exactly 0 (so C(z) = 0), which breaks y(z).
    gap = 1.0
    while residual(4 * np.pi**2 - gap) <= 0:
        gap /= 10
        if gap < 1e-8:
            raise RuntimeError(
                "Lambert solver failed to converge: time of flight exceeds the "
                "single-revolution limit (multi-revolution transfers not supported).")
    z_hi = 4 * np.pi**2 - gap
    z_lo = 0.0
    step = 1.0
    while residual(z_lo) > 0:
        # Too slow even at the parabolic limit: the transfer is hyperbolic.
        z_hi = z_lo
        z_lo = -step
        step *= 2
        if step > 1e7:
            raise RuntimeError(
                "Lambert solver failed to converge (no valid bracket found): "
                "time of flight is shorter than any hyperbolic transfer allows.")

    z = z_lo if y_of_z(z_lo) >= 0 else 0.5 * (z_lo + z_hi)
    for _ in range(max_iter + 200):
        y = y_of_z(z)
        if y < 0:
            z_lo = z
            z = 0.5 * (z_lo + z_hi)
            continue
        Cz = _stumpff_C(z)
        Sz = _stumpff_S(z)
        chi = np.sqrt(y / Cz)
        t = (chi**3 * Sz + A * np.sqrt(y)) / np.sqrt(mu)

        if abs(t - dt) < tol:
            break
        if t < dt:
            z_lo = z
        else:
            z_hi = z

        if abs(z) > 1e-6:
            dCdz = (1 - z * Sz - 2 * Cz) / (2 * z)
            dSdz = (Cz - 3 * Sz) / (2 * z)
        else:
            # Series-based derivatives near z = 0.
            dCdz = -1 / 24
            dSdz = -1 / 120

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dtdz = (chi**3 * (dSdz - 3 * Sz * dCdz / (2 * Cz)) +
                    (A / 8) * (3 * Sz * np.sqrt(y) / Cz + A / chi)) / np.sqrt(mu)
            z_new = z - (t - dt) / dtdz

        if not np.isfinite(z_new) or not (z_lo < z_new < z_hi):
            z_new = 0.5 * (z_lo + z_hi)
        if z_new == z:
            break  # bracket collapsed to machine precision
        z = z_new
    else:
        raise RuntimeError("Lambert solver failed to converge (iteration limit reached).")

    y = y_of_z(z)
    Cz = _stumpff_C(z)
    f = 1 - y / r1
    g = A * np.sqrt(y / mu)
    gdot = 1 - y / r2

    v1_vec = (r2_vec - f * r1_vec) / g
    v2_vec = (gdot * r2_vec - r1_vec) / g

    return v1_vec, v2_vec
