"""
Standalone end-to-end solution of the assignment's own observation data.

Hardcodes the assignment's ground station and two AER observations, runs
the full pipeline (topocentric AER -> ECI position via frames.py -> Lambert
solve via lamsolbert.py -> classical orbital elements via elements.py), and
prints and saves the resulting r1, r2, v1, v2, and full element set. This
is the standalone answer to the assignment's requirement to "present the
orbital parameters of the resulting orbit" -- independent of the GUI.

Run directly: `python assignment_answer.py`
Output is printed to the console and saved to assignment_answer_results.txt.
"""

from datetime import datetime, timezone

from core.time_utils import gmst_degrees
from core.frames import aer_to_eci
from core.lamsolbert import lamsolbert
from core.elements import rv_to_elements

MU_EARTH = 398600.4418

STATION_ECEF_KM = (1344.143, 6068.601, 1429.311)

OBSERVATION_1 = dict(
    utc=datetime(2023, 4, 2, 0, 30, 0, 0, tzinfo=timezone.utc),
    az_deg=132.67, el_deg=32.44, range_km=16945.450,
)
OBSERVATION_2 = dict(
    utc=datetime(2023, 4, 2, 3, 0, 0, 0, tzinfo=timezone.utc),
    az_deg=123.08, el_deg=50.06, range_km=37350.340,
)

RESULTS_FILENAME = "assignment_answer_results.txt"


def _fmt_angle(value):
    return "undefined" if value is None else f"{value:.6f} deg"


def solve_assignment():
    """Run the full pipeline on the assignment's data and return a dict
    with all intermediate and final results, for both printing and
    programmatic use (e.g. from tests)."""
    gmst1 = gmst_degrees(OBSERVATION_1["utc"])
    gmst2 = gmst_degrees(OBSERVATION_2["utc"])

    r1 = aer_to_eci(OBSERVATION_1["az_deg"], OBSERVATION_1["el_deg"],
                     OBSERVATION_1["range_km"], STATION_ECEF_KM, gmst1)
    r2 = aer_to_eci(OBSERVATION_2["az_deg"], OBSERVATION_2["el_deg"],
                     OBSERVATION_2["range_km"], STATION_ECEF_KM, gmst2)

    dt_seconds = (OBSERVATION_2["utc"] - OBSERVATION_1["utc"]).total_seconds()

    v1, v2 = lamsolbert(r1, r2, dt_seconds, mu=MU_EARTH, prograde=True)
    elements = rv_to_elements(r1, v1, mu=MU_EARTH)

    return {
        "gmst1_deg": gmst1, "gmst2_deg": gmst2,
        "r1": r1, "r2": r2, "dt_seconds": dt_seconds,
        "v1": v1, "v2": v2, "elements": elements,
    }


def format_report(result):
    els = result["elements"]
    lines = []
    lines.append("=" * 70)
    lines.append("ASSIGNMENT DATA -- FULL PIPELINE SOLUTION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Inputs")
    lines.append("-" * 70)
    lines.append(f"Ground station ECEF (km): X={STATION_ECEF_KM[0]}, "
                  f"Y={STATION_ECEF_KM[1]}, Z={STATION_ECEF_KM[2]}")
    lines.append(f"Observation 1: {OBSERVATION_1['utc'].isoformat()}  "
                  f"Az={OBSERVATION_1['az_deg']} deg, El={OBSERVATION_1['el_deg']} deg, "
                  f"Range={OBSERVATION_1['range_km']} km")
    lines.append(f"Observation 2: {OBSERVATION_2['utc'].isoformat()}  "
                  f"Az={OBSERVATION_2['az_deg']} deg, El={OBSERVATION_2['el_deg']} deg, "
                  f"Range={OBSERVATION_2['range_km']} km")
    lines.append(f"GMST at obs 1: {result['gmst1_deg']:.6f} deg")
    lines.append(f"GMST at obs 2: {result['gmst2_deg']:.6f} deg")
    lines.append(f"Time of flight (obs1 -> obs2): {result['dt_seconds']:.3f} s")
    lines.append("")
    lines.append("Position vectors (ECI, km)")
    lines.append("-" * 70)
    lines.append(f"r1 = [{result['r1'][0]:.6f}, {result['r1'][1]:.6f}, {result['r1'][2]:.6f}]")
    lines.append(f"r2 = [{result['r2'][0]:.6f}, {result['r2'][1]:.6f}, {result['r2'][2]:.6f}]")
    lines.append("")
    lines.append("Velocity vectors from Lambert solve (ECI, km/s)")
    lines.append("-" * 70)
    lines.append(f"v1 = [{result['v1'][0]:.6f}, {result['v1'][1]:.6f}, {result['v1'][2]:.6f}]")
    lines.append(f"v2 = [{result['v2'][0]:.6f}, {result['v2'][1]:.6f}, {result['v2'][2]:.6f}]")
    lines.append("")
    lines.append("Classical orbital elements (at observation 1 epoch)")
    lines.append("-" * 70)
    lines.append(f"Semi-major axis, a       : {els['a']:.6f} km")
    lines.append(f"Eccentricity, e          : {els['e']:.6f}")
    lines.append(f"Inclination, i           : {els['i']:.6f} deg")
    lines.append(f"RAAN                     : {_fmt_angle(els['raan'])}")
    lines.append(f"Argument of perigee      : {_fmt_angle(els['argp'])}")
    lines.append(f"True anomaly, nu         : {els['nu']:.6f} deg")
    lines.append(f"Angular momentum, h      : {els['h']:.6f} km^2/s")
    if els["notes"]:
        lines.append("")
        lines.append("Notes:")
        for note in els["notes"]:
            lines.append(f"  - {note}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    result = solve_assignment()
    report = format_report(result)
    print(report)

    with open(RESULTS_FILENAME, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nSaved to: {RESULTS_FILENAME}")
