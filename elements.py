"""
Classical orbital elements from a position/velocity state vector.

Standard vis-viva / angular-momentum / eccentricity-vector formulation
(Vallado, Algorithm 9, "rv2coe"). Handles the classic degeneracies:

- Near-circular orbits (e ~ 0): argument of perigee is undefined (no
  well-defined perigee direction) -- reported as None with a note.
- Near-equatorial orbits (i ~ 0 or i ~ 180): RAAN is undefined (no
  well-defined ascending node) -- reported as None with a note.

All angles are returned in degrees, a in km, e dimensionless.
"""

import numpy as np

ECC_TOL = 1e-8
INC_TOL = 1e-8  # radians; ~5.7e-7 deg


def rv_to_elements(r_vec, v_vec, mu=398600.4418) -> dict:
    """
    Convert a Cartesian state vector (r_vec [km], v_vec [km/s]) at some
    epoch into classical orbital elements.

    Returns a dict with keys:
        a (km), e, i (deg), raan (deg or None), argp (deg or None),
        nu (deg, true anomaly), h (angular momentum magnitude, km^2/s),
        e_vec (eccentricity vector, np.ndarray, km^3/s^2-normalized i.e.
        dimensionless direction with magnitude e), h_vec (angular momentum
        vector, np.ndarray, km^2/s), notes (list of str).

        e_vec and h_vec are frame-independent and always well-defined
        (unlike raan/argp, which can be None) -- they are the quantities
        to use for reconstructing the orbit's geometry in degenerate
        (near-circular / near-equatorial) cases.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    notes = []

    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)

    n_vec = np.cross([0, 0, 1], h_vec)  # node vector, points to ascending node
    n = np.linalg.norm(n_vec)

    e_vec = ((v**2 - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)

    energy = v**2 / 2 - mu / r
    if abs(e - 1.0) > ECC_TOL:
        a = -mu / (2 * energy)
    else:
        a = np.inf
        notes.append("Orbit is parabolic (e ~ 1); semi-major axis is undefined (inf).")

    i = np.degrees(np.arccos(np.clip(h_vec[2] / h, -1.0, 1.0)))

    # RAAN (undefined for near-equatorial orbits: n ~ 0)
    if n > INC_TOL:
        raan = np.degrees(np.arccos(np.clip(n_vec[0] / n, -1.0, 1.0)))
        if n_vec[1] < 0:
            raan = 360.0 - raan
    else:
        raan = None
        notes.append("Orbit is near-equatorial (i ~ 0 or 180 deg); RAAN is undefined.")

    # Argument of perigee (undefined for near-circular orbits: e ~ 0)
    if e > ECC_TOL and n > INC_TOL:
        argp = np.degrees(np.arccos(np.clip(np.dot(n_vec, e_vec) / (n * e), -1.0, 1.0)))
        if e_vec[2] < 0:
            argp = 360.0 - argp
    elif e <= ECC_TOL:
        argp = None
        notes.append("Orbit is near-circular (e ~ 0); argument of perigee is undefined.")
    else:
        argp = None
        notes.append("Orbit is near-equatorial; argument of perigee is undefined "
                      "(no ascending node reference).")

    # True anomaly: angle between eccentricity vector and position vector.
    if e > ECC_TOL:
        cos_nu = np.clip(np.dot(e_vec, r_vec) / (e * r), -1.0, 1.0)
        nu = np.degrees(np.arccos(cos_nu))
        if np.dot(r_vec, v_vec) < 0:
            nu = 360.0 - nu
    else:
        # Near-circular: measure true anomaly from ascending node (or from
        # x-axis if also near-equatorial), i.e. the "argument of latitude".
        if n > INC_TOL:
            cos_nu = np.clip(np.dot(n_vec, r_vec) / (n * r), -1.0, 1.0)
            nu = np.degrees(np.arccos(cos_nu))
            if r_vec[2] < 0:
                nu = 360.0 - nu
        else:
            cos_nu = np.clip(r_vec[0] / r, -1.0, 1.0)
            nu = np.degrees(np.arccos(cos_nu))
            if r_vec[1] < 0:
                nu = 360.0 - nu
        notes.append("True anomaly measured as argument of latitude / true longitude "
                      "due to near-circular and/or near-equatorial geometry.")

    return {
        "a": a,
        "e": e,
        "i": i,
        "raan": raan,
        "argp": argp,
        "nu": nu,
        "h": h,
        "e_vec": e_vec,
        "h_vec": h_vec,
        "notes": notes,
    }
