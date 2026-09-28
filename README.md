# Lambert's Problem Orbit Determination Tool

Determines an orbit from two topocentric (Azimuth, Elevation, Range)
observations taken from a fixed ground station at two different UTC times,
by converting them to ECI position vectors and solving Lambert's problem.

## Dependencies

- **Python 3.9+** (developed/tested on 3.14).
- **Tkinter** for the GUI -- part of the Python standard library on
  Windows/macOS installers; on Linux it may need a separate OS package
  (e.g. `sudo apt install python3-tk` on Debian/Ubuntu). Only needed to
  run `main.py`/`gui.py`; the core solver and `verify.py`/
  `assignment_answer.py` do not require it.
- Third-party packages, pinned in `requirements.txt`:

  | Package      | Used for                                              |
  |--------------|--------------------------------------------------------|
  | `numpy`      | All vector/matrix math throughout the pipeline.        |
  | `matplotlib` | The GUI's embedded 3D orbit/Earth plot (`gui.py`).      |
  | `scipy`      | Listed as an allowed dependency for this project, but not currently imported anywhere in the code -- everything so far only needed numpy. Kept in `requirements.txt` in case a future extension needs it. |

No other third-party packages, no MATLAB, no paid software.

## Install

```
pip install -r requirements.txt
```

## Run

```
python main.py
```

This opens a Tkinter GUI with two input modes (AER + time, or raw position
vectors + time), a results panel with classical orbital elements, and an
embedded 3D plot of the resulting orbit.

To solve the assignment's own observation data end-to-end and print/save
the resulting orbital parameters (no GUI needed):

```
python assignment_answer.py
```

To verify the core Lambert solver (LamSolbert) independently against a textbook example
(no GUI needed):

```
python verify.py
```

To run the stress-test suite (edge cases, input validation, regression
checks -- see `tests/run_stress_tests.py`):

```
python tests/run_stress_tests.py
```

## Method, stage by stage

**Time (`time_utils.py`).** Each observation's UTC datetime is converted to
a Julian Date using the standard Gregorian-calendar JD algorithm, then to
Greenwich Mean Sidereal Time (GMST) via the IAU 1982 GMST polynomial. GMST
is recomputed independently for each observation rather than reused,
because the two observations here are 2.5 hours apart and GMST changes by
roughly 15.04 deg/hour.

**Frames (`frames.py`).** The station's ECEF position is converted to WGS84
geodetic latitude/longitude/altitude via the standard iterative Bowring
algorithm. Each AER observation is converted to a Cartesian vector in the
topocentric SEZ (South-East-Zenith) frame, rotated into an ECEF offset
using the station's geodetic latitude and longitude, added to the station's
ECEF position to get the satellite's ECEF position, and finally rotated
about the Z-axis by that observation's GMST to obtain the ECI position
vector. See the docstrings in `frames.py` for the exact sign/rotation
conventions used, and a flagged note on the one interpretive assumption
made (that the given station ECEF coordinates, being time-invariant for a
fixed ground station, apply unchanged at both observation epochs).

**Lambert's problem (`lamsolbert.py`).** Given the two ECI position vectors
and the time of flight between them, the universal-variable formulation is
used to solve for the transfer orbit's velocity vectors at each end. This
formulation (built on the Stumpff functions C(z) and S(z), with the
universal anomaly z found via Newton iteration, falling back to bisection
if Newton fails to converge) works uniformly across elliptical, parabolic,
and hyperbolic transfer geometries, unlike the classical p-iteration
method. Only the short-way transfer is implemented, as permitted by the
assignment. The solver, named LamSolbert (`lamsolbert()` in `lamsolbert.py`),
is verified independently in `verify.py` against the worked Lambert example
in Vallado's *Fundamentals of Astrodynamics and Applications* (Example 5-5).

**Orbital elements (`elements.py`).** From the position and velocity vector
at one epoch, the classical orbital elements (semi-major axis, eccentricity,
inclination, RAAN, argument of perigee, true anomaly) are computed via the
standard angular-momentum-vector / eccentricity-vector / vis-viva
formulation. Degenerate cases -- near-circular orbits (undefined argument
of perigee) and near-equatorial orbits (undefined RAAN) -- are detected and
reported as "undefined" with an explanatory note rather than raising an
exception or returning a nonsensical angle.

**GUI (`gui.py` / `main.py`).** A Tkinter front end lets you pick between
AER-mode and vector-mode inputs, run the full pipeline, and view the
resulting classical elements alongside a 3D plot of Earth and the computed
transfer orbit. The GUI contains no orbital-mechanics logic of its own --
it calls the exact same functions (`lamsolbert`, `rv_to_elements`,
`aer_to_eci`, `gmst_degrees`) that `verify.py` and the console pipeline
tests use, so there is nothing to keep in sync between "GUI math" and
"verified math."

## Notes / assumptions flagged in code

- GMST is computed treating the input JD as UT1 directly (TT/UT1 offset
  ignored), a standard simplification adequate for this problem's accuracy
  requirements.
- The station's ECEF coordinates are assumed valid at both observation
  epochs unchanged, since ECEF is by definition Earth-fixed (see
  `frames.py` module docstring for the full discussion).
- Only short-way (direct) Lambert transfers are solved; long-way transfer
  is not implemented.
