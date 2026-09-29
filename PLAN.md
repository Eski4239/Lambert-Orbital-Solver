# v2 plan — teacher feedback

Feedback (Sept 2026):
1. Improve the UI: more dynamic trajectory input, better visualisation.
2. Add a new scope: Near-Earth Objects around the Sun, from a catalog.

Decisions: Plotly + Dash local web app · bundled JPL SBDB snapshot + online
refresh · NEO scope = catalog browser + solar-system view + porkchop plot.

## Target layout

```
core/                 pure numpy, no UI
  constants.py        MU_EARTH, MU_SUN, AU, R_EARTH, ...
  time_utils.py       (moved) JD, GMST
  frames.py           (moved) ECEF/geodetic/AER -> ECI
  lamsolbert.py       (moved) Lambert solver, faster long-transfer path
  elements.py         (moved) rv -> classical elements
  kepler.py           NEW Kepler eq., elements -> r/v, propagation
helio/
  planets.py          NEW JPL approx. planet elements (Standish, 1800-2050)
  neo_catalog.py      NEW load / filter / refresh SBDB NEO catalog
  porkchop.py         NEW Earth -> NEO transfer grid via lamsolbert(MU_SUN)
app/                  Dash UI
  earth_page.py       orbit determination (dynamic inputs, 3D, animation)
  neo_page.py         NEO browser, solar system view, porkchop
data/                 bundled catalog (neo_asteroids.csv.gz), coastlines, land mask
legacy/gui.py         original Tkinter GUI, kept for reference
tests/                pytest suite + existing stress runner
main.py               launches the Dash app
```

## Phases

- [x] **0. Safety net** — git repo, v1 baseline on `main`, work on `v2`.
- [x] **1. Core** — package layout; constants; Kepler propagation;
      speed up Lambert long transfers (results unchanged); planet ephemeris;
      pytest suite (Vallado checks, round-trips, Lambert-vs-propagation).
- [x] **2. Earth scope UI** — Dash app shell with two tabs.
      Inputs: editable observation table (add/remove rows, paste CSV,
      choose pair), AER or vector mode, presets, live solve with inline
      validation, TOF slider, prograde/retrograde toggle.
      Visualisation: WebGL 3D Earth, orbit, r/v vectors, periapsis/apoapsis,
      node line, station + line-of-sight, animated satellite on a time
      slider, 2D ground track, ECI/ECEF toggle, dark theme.
- [x] **3. NEO scope** — catalog table with filters (class, PHA, H,
      MOID, a/e/i ranges) and search; heliocentric 3D view of planets +
      selected NEOs with date slider/animation; object detail card;
      porkchop plot (C3 / total Δv over departure x TOF) with click-to-show
      transfer trajectory.
- [x] **Pre-4 polish** — ephemeris-range validation, non-viable pick
      warning, refined best transfer (SciPy), closest approach, PNG/CSV
      export, "How it works" notes, jump-to-date, click-to-select in 3D,
      refreshed catalog kept separate from the bundled snapshot.
- [x] **4. Polish** — README rewritten for v2, stress tests extended
      (sections 8–10: v2 propagation, web-UI input validation, NEO edge
      cases), final review.
