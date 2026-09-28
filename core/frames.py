"""
Coordinate frame conversions: station ECEF -> geodetic, and topocentric
AER (azimuth, elevation, range) -> geocentric equatorial (ECI) position.

Frame/convention notes (read this before touching the math):

- ECEF: Earth-Centered, Earth-Fixed, X toward (0N,0E), Z toward the north
  pole, Y completing the right-handed set. Rotates with the Earth.
- Geodetic lat/lon/alt: WGS84 ellipsoid. Latitude is geodetic (normal to
  the ellipsoid), not geocentric.
- SEZ (topocentric horizon frame): South-East-Zenith, right-handed, origin
  at the station. This is the natural frame for Az/El/Range: an object at
  azimuth Az (measured clockwise from local North) and elevation El
  (measured up from the local horizontal plane) at slant range Range has
  SEZ components:
      S = -Range*cos(El)*cos(Az)
      E =  Range*cos(El)*sin(Az)
      Z =  Range*sin(El)
- SEZ -> ECEF: rotate by (90 - lat) about the Y-axis then by lon about Z
  (equivalently, a single rotation matrix built from lat/lon, see
  `sez_to_ecef_rotation`), then translate by the station's ECEF position.
- ECEF -> ECI: rotate about Z by +GMST (Greenwich Mean Sidereal Time) at
  the *epoch of that specific observation* -- NOT a single GMST for both
  observations, since they are 2.5 hours apart (see time_utils.py).

GUESS/AMBIGUITY FLAG: the assignment gives a single ECEF station position
(X, Y, Z) without stating a specific epoch it is valid at. ECEF coordinates
are nominally time-invariant (station is fixed on the rotating Earth, so
its ECEF coords don't change with time, aside from tectonic drift which is
negligible here) -- so this ECEF position is used as-is for both
observation epochs. This is the standard/expected interpretation, but is
flagged here since the assignment refers to it as "the Greenwich/reference
epoch," which could otherwise be (mis)read as requiring some epoch-specific
correction.
"""

import numpy as np

R_EARTH = 6378.137  # km, WGS84 semi-major axis
F_EARTH = 1.0 / 298.257223563  # WGS84 flattening
E2_EARTH = F_EARTH * (2 - F_EARTH)  # WGS84 first eccentricity squared


def ecef_to_geodetic(x, y, z, tol=1e-12, max_iter=50):
    """
    Convert ECEF (km) to WGS84 geodetic latitude/longitude/altitude.

    Uses the standard iterative (Bowring-style) algorithm: solve for
    latitude iteratively since the ellipsoid normal's radius of curvature
    N depends on latitude itself.

    Returns
    -------
    lat_deg, lon_deg, alt_km
    """
    lon = np.arctan2(y, x)

    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - E2_EARTH))  # initial guess (geocentric-ish)

    for _ in range(max_iter):
        N = R_EARTH / np.sqrt(1 - E2_EARTH * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat_new = np.arctan2(z, p * (1 - E2_EARTH * N / (N + alt)))
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new

    N = R_EARTH / np.sqrt(1 - E2_EARTH * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N

    return np.degrees(lat), np.degrees(lon), alt


def _sez_from_aer(az_deg, el_deg, rng):
    """Az/El/Range -> SEZ topocentric Cartesian vector (km)."""
    az = np.radians(az_deg)
    el = np.radians(el_deg)

    s = -rng * np.cos(el) * np.cos(az)
    e = rng * np.cos(el) * np.sin(az)
    z = rng * np.sin(el)

    return np.array([s, e, z])


def _sez_to_ecef_rotation(lat_deg, lon_deg):
    """
    Rotation matrix R such that r_ecef_offset = R @ r_sez.

    Built as Rz(lon) @ Ry(90-lat) applied to the SEZ basis, expressed
    directly as the combined matrix (Vallado Eq. 4-1 / topocentric-horizon
    rotation).
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    sl, cl = np.sin(lat), np.cos(lat)
    sg, cg = np.sin(lon), np.cos(lon)

    R = np.array([
        [sl * cg, -sg, cl * cg],
        [sl * sg,  cg, cl * sg],
        [-cl,       0,  sl],
    ])
    return R


def station_ecef_to_geodetic(station_ecef_km):
    """Convenience wrapper: station ECEF vector (km) -> (lat_deg, lon_deg, alt_km)."""
    x, y, z = station_ecef_km
    return ecef_to_geodetic(x, y, z)


def aer_to_eci(az_deg, el_deg, rng_km, station_ecef_km, gmst_deg):
    """
    Convert a topocentric AER observation to a geocentric equatorial (ECI)
    position vector.

    Pipeline: AER -> SEZ (topocentric) -> ECEF offset -> ECEF position
    (add station ECEF) -> ECI (rotate by GMST about Z).

    Parameters
    ----------
    az_deg, el_deg : float
        Azimuth (from North, clockwise) and elevation (from horizon), deg.
    rng_km : float
        Slant range, km.
    station_ecef_km : array-like, shape (3,)
        Station position in ECEF, km.
    gmst_deg : float
        Greenwich Mean Sidereal Time at the observation epoch, degrees.
        Must be computed per-observation (see time_utils.gmst_degrees).

    Returns
    -------
    r_eci : np.ndarray, shape (3,)
        Satellite position vector in ECI, km.

    Raises
    ------
    ValueError
        If rng_km is not positive, or el_deg is outside [-90, 90]. This
        validation lives here (not only in the GUI layer) so the function
        is safe to call directly from any other caller -- a script, a
        test, a future extension -- without silently accepting physically
        invalid input.
    """
    if rng_km <= 0:
        raise ValueError(f"Range must be positive, got {rng_km!r} km.")
    if not (-90.0 <= el_deg <= 90.0):
        raise ValueError(f"Elevation must be within [-90, 90] deg, got {el_deg!r} deg.")

    station_ecef_km = np.asarray(station_ecef_km, dtype=float)
    lat_deg, lon_deg, _alt = station_ecef_to_geodetic(station_ecef_km)

    r_sez = _sez_from_aer(az_deg, el_deg, rng_km)
    R = _sez_to_ecef_rotation(lat_deg, lon_deg)
    r_ecef_offset = R @ r_sez

    r_ecef = station_ecef_km + r_ecef_offset

    theta = np.radians(gmst_deg)
    ct, st = np.cos(theta), np.sin(theta)
    Rz = np.array([
        [ct, -st, 0],
        [st,  ct, 0],
        [0,    0, 1],
    ])
    r_eci = Rz @ r_ecef

    return r_eci


if __name__ == "__main__":
    # Sanity check: convert the assignment's station ECEF to geodetic and
    # print it, so the lat/lon look physically plausible (station should
    # land somewhere reasonable on Earth, not NaN / absurd values).
    station_ecef = [1344.143, 6068.601, 1429.311]
    lat, lon, alt = station_ecef_to_geodetic(station_ecef)
    print(f"Station ECEF: {station_ecef} km")
    print(f"Station geodetic: lat={lat:.4f} deg, lon={lon:.4f} deg, alt={alt:.4f} km")
