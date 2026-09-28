"""
Tests for the Near-Earth Objects page helpers, including the browser-side
propagator (app/assets/solar.js), which is run under Node.js when available
and compared with core.kepler.position_at.

Run: python -m pytest tests
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.constants import AU, MU_SUN                                    # noqa: E402
from core.kepler import position_at                                     # noqa: E402
from helio.neo_catalog import elements_of, load_catalog                 # noqa: E402
from helio.planets import planet_state                                  # noqa: E402
from app.neo_figures import catalog_pack, pack_positions, planets_pack  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    df = load_catalog()
    picks = df[df["pdes"].isin(["433", "99942", "101955", "3200"])]
    return picks, catalog_pack(picks)


def test_pack_positions_match_core_kepler(sample):
    rows, pack = sample
    jd = 2462240.5  # 2029-04-13, Apophis flyby
    xyz = pack_positions(pack, jd).T
    for k, (_, row) in enumerate(rows.iterrows()):
        r, _ = position_at(*elements_of(row), jd, MU_SUN)
        np.testing.assert_allclose(xyz[k], r / AU, atol=1e-7)


def test_planets_pack_matches_planet_state():
    jd = 2461500.5
    xyz = pack_positions(planets_pack(jd), jd).T
    for k, name in enumerate(("Mercury", "Venus", "Earth", "Mars")):
        r, _ = planet_state(name, jd)
        np.testing.assert_allclose(xyz[k], r / AU, atol=1e-7)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_browser_propagator_matches_python(sample):
    _, pack = sample
    jds = [2461000.5, 2462240.5, 2463500.25]
    script = (
        "global.window = {};"
        f"eval(require('fs').readFileSync({json.dumps(os.path.join(ROOT, 'app', 'assets', 'solar.js'))}, 'utf8'));"
        f"const pack = {json.dumps(pack)};"
        f"const out = {json.dumps(jds)}.map(jd => window.orbitLabSolar.positions(pack, jd));"
        "console.log(JSON.stringify(out));"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    browser = json.loads(result.stdout)
    for jd, xyz in zip(jds, browser):
        np.testing.assert_allclose(np.array(xyz), pack_positions(pack, jd), atol=1e-9)


def test_date_helpers_round_trip():
    from app.neo_page import date_to_jd, jd_to_date
    assert jd_to_date(2451545.0) == "2000-01-01"
    assert jd_to_date(date_to_jd("2029-04-13")) == "2029-04-13"
