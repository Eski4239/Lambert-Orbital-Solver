"""
Time conversion utilities: UTC datetime -> Julian Date -> GMST.

These are needed per-observation, since Greenwich Mean Sidereal Time (GMST)
changes continuously with time. Two observations taken 2.5 hours apart
require two separate GMST evaluations (~37.5 degrees of Earth rotation
apart), NOT a single GMST reused for both.
"""

import math
from datetime import datetime, timezone


def julian_date(dt: datetime) -> float:
    """
    Convert a UTC datetime to Julian Date (JD), using the standard
    Fliegel & Van Flandern algorithm extended with a fractional day.

    dt must be timezone-aware UTC, or naive and assumed to already be UTC.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + (dt.second + dt.microsecond / 1e6) / 3600.0

    # Standard algorithm (Meeus / Vallado), valid for Gregorian calendar dates.
    if month <= 2:
        year -= 1
        month += 12

    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)

    jd = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + B
        - 1524.5
        + hour / 24.0
    )
    return jd


def gmst_seconds_from_jd(jd: float) -> float:
    """
    Compute Greenwich Mean Sidereal Time (GMST) in degrees [0, 360) for a
    given Julian Date, using the IAU 1982 GMST polynomial (Vallado Eq. 3-45).

    T is Julian centuries of UT1 since J2000.0 (TT/UT1 distinction ignored,
    i.e. JD is treated as UT1 directly -- standard simplifying assumption
    for problems that don't specify UT1-UTC offset / do not require
    sub-arcsecond sidereal-time accuracy).
    """
    T = (jd - 2451545.0) / 36525.0

    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T**2
        - 6.2e-6 * T**3
    )

    # gmst_sec is in "time seconds" scaled so that 86400 sec = 360 deg,
    # but the polynomial above actually already yields seconds of time
    # for sidereal-to-solar conversion; normalize into degrees directly.
    gmst_deg = math.fmod(gmst_sec, 86400.0) / 240.0  # 240 sec of time = 1 deg
    if gmst_deg < 0:
        gmst_deg += 360.0
    return gmst_deg


def gmst_degrees(dt: datetime) -> float:
    """Convenience wrapper: UTC datetime -> GMST in degrees [0, 360)."""
    jd = julian_date(dt)
    return gmst_seconds_from_jd(jd)


if __name__ == "__main__":
    # Sanity check against the well-known J2000.0 epoch reference:
    # 2000-01-01 12:00:00 UTC -> JD = 2451545.0, GMST ~ 280.4606 deg
    # (Vallado, "Fundamentals of Astrodynamics and Applications", example values)
    epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd = julian_date(epoch)
    gmst = gmst_seconds_from_jd(jd)

    print(f"J2000.0 epoch: {epoch.isoformat()}")
    print(f"Computed JD:   {jd:.6f}  (expected 2451545.000000)")
    print(f"Computed GMST: {gmst:.4f} deg  (expected ~280.4606 deg)")

    jd_ok = abs(jd - 2451545.0) < 1e-6
    gmst_ok = abs(gmst - 280.4606) < 1e-2
    print("PASS" if (jd_ok and gmst_ok) else "FAIL")

    # Second sanity check: the assignment's actual observation epochs, just
    # to confirm GMST differs meaningfully across the 2.5-hour gap.
    obs1 = datetime(2023, 4, 2, 0, 30, 0, tzinfo=timezone.utc)
    obs2 = datetime(2023, 4, 2, 3, 0, 0, tzinfo=timezone.utc)
    print(f"\nObs1 GMST: {gmst_degrees(obs1):.4f} deg")
    print(f"Obs2 GMST: {gmst_degrees(obs2):.4f} deg")
    print(f"Delta:     {gmst_degrees(obs2) - gmst_degrees(obs1):.4f} deg "
          f"(expected ~ 2.5h * 15.0410686 deg/h = {2.5 * 15.0410686:.4f} deg)")
