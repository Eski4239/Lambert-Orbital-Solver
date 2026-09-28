"""
Standalone verification of lamsolbert.py against a standard textbook example.

Reference: Vallado, "Fundamentals of Astrodynamics and Applications",
4th ed., Example 5-5 (universal variable Lambert solver worked example).

Given:
    r1 = [15945.34, 0.0, 0.0] km
    r2 = [12214.83899, 10249.46731, 0.0] km
    dt = 76 * 60 s (76 minutes)
    prograde (short-way) transfer

Expected:
    v1 ~ [2.058913, 2.915965, 0.0] km/s
    v2 ~ [-3.451565, 0.910315, 0.0] km/s

Run directly: `python verify.py`
"""

import numpy as np
from core.lamsolbert import lamsolbert

MU_EARTH = 398600.4418


def run_vallado_example():
    r1 = np.array([15945.34, 0.0, 0.0])
    r2 = np.array([12214.83899, 10249.46731, 0.0])
    dt = 76 * 60.0

    v1_expected = np.array([2.058913, 2.915965, 0.0])
    v2_expected = np.array([-3.451565, 0.910315, 0.0])

    v1, v2 = lamsolbert(r1, r2, dt, mu=MU_EARTH, prograde=True)

    print("Vallado Example 5-5 Lambert verification")
    print("-" * 50)
    print(f"r1 = {r1} km")
    print(f"r2 = {r2} km")
    print(f"dt = {dt:.1f} s")
    print()
    print(f"v1 computed = {v1}")
    print(f"v1 expected = {v1_expected}")
    print(f"v2 computed = {v2}")
    print(f"v2 expected = {v2_expected}")
    print()

    tol = 1e-3  # km/s, a few decimal places
    v1_ok = np.allclose(v1, v1_expected, atol=tol)
    v2_ok = np.allclose(v2, v2_expected, atol=tol)

    if v1_ok and v2_ok:
        print("PASS: computed velocities agree with Vallado's textbook example.")
    else:
        print("FAIL: computed velocities do not agree with expected values.")
        if not v1_ok:
            print(f"  v1 diff = {v1 - v1_expected}")
        if not v2_ok:
            print(f"  v2 diff = {v2 - v2_expected}")

    assert v1_ok, "v1 does not match Vallado's expected value"
    assert v2_ok, "v2 does not match Vallado's expected value"


if __name__ == "__main__":
    run_vallado_example()
