"""
Physical constants shared by the Earth-orbit and heliocentric scopes.

Units throughout the project: km, s, km^3/s^2, degrees for angles at API
boundaries (radians internally).
"""

# Earth (WGS84 / EGM96 values, matching core/frames.py)
MU_EARTH = 398600.4418           # km^3/s^2
R_EARTH = 6378.137               # km, equatorial radius
OMEGA_EARTH = 7.292115e-5        # rad/s, sidereal rotation rate

# Sun (IAU 2015 nominal / DE430)
MU_SUN = 1.32712440018e11        # km^3/s^2
R_SUN = 695700.0                 # km

AU = 149597870.7                 # km (IAU 2012, exact)
DAY = 86400.0                    # s
JULIAN_YEAR = 365.25 * DAY       # s
JD_J2000 = 2451545.0             # 2000-01-01 12:00 TT

# Mean obliquity of the ecliptic at J2000 (IAU 1980), for ecliptic <->
# equatorial rotations in the heliocentric scope.
OBLIQUITY_J2000_DEG = 23.43928
