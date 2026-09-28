"""
Plotly figures for the Earth-orbit page: 3D view (ECI or ECEF) and ground
track. Pure functions of an OrbitSolution plus display options.

Animation: every figure carries a trace with uid "sat" (the satellite) and,
in 3D, "earth" and "coast" (globe surface and coastlines). The browser moves
those traces directly (app/assets/animate.js) from the samples in
`sample_track`, so scrubbing the time slider never round-trips to Python.

The 3D scene always uses equal, explicit axis ranges sized to everything
drawn (aspectmode "cube"): the Earth stays round and the whole trajectory
fits, whatever orbit was shown before.
"""

import json
import os

import numpy as np
import plotly.graph_objects as go

from core.constants import MU_EARTH, R_EARTH
from core.frames import ecef_to_latlon, eci_to_ecef
from core.kepler import orbit_curve

N_SAMPLES = 300

ACCENT = "#1d5fd1"
ARC = "#e8590c"
TEXT = "#0f1b2d"
MUTED = "#5b6878"
GRID = "#e3e8ef"
STATION = "#12915a"
VELOCITY = "#7c3aed"

# Globe colours, indexed by the surfacecolor value built in _earth_grid:
# ocean 0-0.3 (deep at the equator, lighter poleward), land 0.4-0.7
# (green to olive poleward), ice 0.95.
EARTH_COLORSCALE = [
    [0.00, "#15509c"], [0.30, "#4b8bd0"],
    [0.40, "#3e8a3f"], [0.55, "#78994a"], [0.70, "#a9a070"],
    [0.90, "#e8eef5"], [1.00, "#f8fafc"],
]
# Flat, lighter tints for the 2D map so the track stays the focus.
MAP_OCEAN, MAP_LAND, MAP_ICE = "#d6e6f7", "#d7e7c9", "#f4f7fb"

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load_coastlines():
    with open(os.path.join(_DATA, "coastlines_110m.json"), encoding="utf-8") as f:
        lines = json.load(f)["lines"]
    lon, lat = [], []
    for line in lines:
        for x, y in line:
            lon.append(x)
            lat.append(y)
        lon.append(None)
        lat.append(None)
    return lon, lat


def _load_land_mask():
    """Boolean grid [lat 90..-90, lon -180..180] at 1 degree."""
    with open(os.path.join(_DATA, "land_mask_1deg.json"), encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    return np.array([[c == "1" for c in row] for row in rows])


COAST_LON, COAST_LAT = _load_coastlines()
LAND = _load_land_mask()


def _coast_ecef(radius=R_EARTH * 1.003):
    lon = np.array([np.nan if v is None else v for v in COAST_LON])
    lat = np.array([np.nan if v is None else v for v in COAST_LAT])
    lo, la = np.radians(lon), np.radians(lat)
    return (radius * np.cos(la) * np.cos(lo), radius * np.cos(la) * np.sin(lo),
            radius * np.sin(la))


def _earth_grid(step=3):
    """Globe mesh (ECEF, km) on a lat/lon grid, plus its surface colours."""
    lats = np.arange(90, -90 - step, -step)
    lons = np.arange(-180, 180 + step, step)
    LON, LAT = np.meshgrid(lons, lats)
    la, lo = np.radians(LAT), np.radians(LON)
    x = R_EARTH * np.cos(la) * np.cos(lo)
    y = R_EARTH * np.cos(la) * np.sin(lo)
    z = R_EARTH * np.sin(la)
    land = LAND[(90 - LAT).astype(int), (LON + 180).astype(int)]
    frac = np.abs(LAT) / 90.0
    color = np.where(land, 0.40 + 0.30 * frac, 0.30 * frac)
    ice = (land & (np.abs(LAT) >= 66)) | (np.abs(LAT) >= 78)
    return x, y, z, np.where(ice, 0.95, color)


def _grid_faces(rows, cols):
    """Two triangles per grid cell, as Mesh3d i/j/k vertex indices."""
    r, c = np.meshgrid(np.arange(rows - 1), np.arange(cols - 1), indexing="ij")
    a = (r * cols + c).ravel()
    b, d = a + 1, a + cols
    return np.concatenate([a, b]), np.concatenate([b, d + 1]), np.concatenate([d, d])


COAST_ECEF = _coast_ecef()
# The globe is a Mesh3d rather than a Surface: plotly re-processes every
# Surface on each restyle (~75 ms for this one, even when only the satellite
# moves), which made the animation stutter; a Mesh3d costs a few ms.
_EX, _EY, _EZ, _EC = _earth_grid()
EARTH_X, EARTH_Y, EARTH_Z, EARTH_C = _EX.ravel(), _EY.ravel(), _EZ.ravel(), _EC.ravel()
EARTH_I, EARTH_J, EARTH_K = _grid_faces(*_EX.shape)


def globe_payload():
    """Static ECEF geometry the browser rotates during ECI animation."""
    def clean(a):
        return [None if np.isnan(v) else round(float(v), 1) for v in a]
    return {"x": clean(COAST_ECEF[0]), "y": clean(COAST_ECEF[1]),
            "earth_x": np.round(EARTH_X, 1).tolist(), "earth_y": np.round(EARTH_Y, 1).tolist()}


def _rotz(x, y, deg):
    t = np.radians(deg)
    return np.cos(t) * x - np.sin(t) * y, np.sin(t) * x + np.cos(t) * y


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def sample_track(sol, n=N_SAMPLES):
    """
    Satellite samples over the animation window: one orbital period for
    closed orbits, twice the time of flight for open ones.
    """
    span = sol.period_s if sol.is_closed else 2 * sol.tof_s
    t = np.linspace(0.0, span, n)
    eci = np.array([sol.state_at(ti)[0] for ti in t])
    gmst = sol.gmst_at(t)
    ecef = eci_to_ecef(eci, gmst)
    lat, lon, alt = ecef_to_latlon(ecef)
    return {"t": t, "eci": eci, "ecef": ecef, "gmst": gmst, "lat": lat, "lon": lon, "alt": alt}


def track_payload(track):
    """JSON-friendly version of sample_track for the browser."""
    def r(a, d=1):
        return np.round(a, d).tolist()
    return {
        "t": r(track["t"], 1), "gmst": r(track["gmst"], 4), "alt": r(track["alt"], 1),
        "lat": r(track["lat"], 3), "lon": r(track["lon"], 3),
        "eci": {"x": r(track["eci"][:, 0]), "y": r(track["eci"][:, 1]), "z": r(track["eci"][:, 2])},
        "ecef": {"x": r(track["ecef"][:, 0]), "y": r(track["ecef"][:, 1]), "z": r(track["ecef"][:, 2])},
    }


# ----------------------------------------------------------------------
# 3D view
# ----------------------------------------------------------------------
def _earth_traces(frame, gmst_deg):
    ex, ey = EARTH_X, EARTH_Y
    cx, cy, cz = COAST_ECEF
    if frame == "eci":
        ex, ey = _rotz(ex, ey, gmst_deg)
        cx, cy = _rotz(cx, cy, gmst_deg)
    globe = go.Mesh3d(
        x=np.round(ex, 1), y=np.round(ey, 1), z=np.round(EARTH_Z, 1),
        i=EARTH_I, j=EARTH_J, k=EARTH_K, intensity=EARTH_C, intensitymode="vertex",
        colorscale=EARTH_COLORSCALE, cmin=0, cmax=1, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.6, diffuse=0.65, specular=0.12, roughness=0.75, fresnel=0.1),
        lightposition=dict(x=1e5, y=5e4, z=8e4), name="Earth", showlegend=False, uid="earth",
    )
    coast = go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", hoverinfo="skip", showlegend=False,
                         line=dict(color="rgba(18, 38, 58, 0.55)", width=1), uid="coast",
                         connectgaps=False)
    return [globe, coast]


def _to_frame(r_eci, gmst_deg, frame):
    return r_eci if frame == "eci" else eci_to_ecef(r_eci, gmst_deg)


def _arrow(tail, vec, color, name, legend):
    head = tail + vec
    line = go.Scatter3d(x=[tail[0], head[0]], y=[tail[1], head[1]], z=[tail[2], head[2]],
                        mode="lines", line=dict(color=color, width=5), name=name,
                        hoverinfo="skip", showlegend=legend, legendgroup=name)
    cone = go.Cone(x=[head[0]], y=[head[1]], z=[head[2]], u=[vec[0]], v=[vec[1]], w=[vec[2]],
                   anchor="tip", sizemode="absolute", sizeref=np.linalg.norm(vec) * 0.25,
                   colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip",
                   legendgroup=name)
    return [line, cone]


def _triad(length, frame):
    """Small labelled X/Y/Z direction marker at the origin, in place of axes."""
    names = ("X", "Y", "Z") if frame == "eci" else ("X (0° lon)", "Y", "Z")
    x, y, z, tx, ty, tz = [], [], [], [], [], []
    for i in range(3):
        tip = np.zeros(3)
        tip[i] = length
        x += [0, tip[0], None]
        y += [0, tip[1], None]
        z += [0, tip[2], None]
        tx.append(tip[0] * 1.1)
        ty.append(tip[1] * 1.1)
        tz.append(tip[2] * 1.1)
    return [
        go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip", showlegend=False,
                     line=dict(color="rgba(15, 27, 45, 0.35)", width=2)),
        go.Scatter3d(x=tx, y=ty, z=tz, mode="text", text=list(names), hoverinfo="skip",
                     showlegend=False, textfont=dict(size=11, color=MUTED)),
    ]


def figure_3d(sol, track, observations, frame="eci", layers=(), station_ecef=None, k=0):
    """
    observations: list of dicts {label, r_eci (3,), gmst_deg, is_pair}.
    layers: subset of {"velocity", "apsides", "station"}.
    k: current animation sample index (satellite + Earth rotation).
    """
    fig = go.Figure()
    for tr in _earth_traces(frame, track["gmst"][k]):
        fig.add_trace(tr)

    # Orbit path. In ECI it is the fixed conic; in ECEF it is the path over
    # the animation window, which winds around as the Earth turns beneath.
    # For closed ECI orbits only the part not covered by the transfer arc is
    # drawn, so the two lines don't z-fight where they overlap.
    if frame == "eci" and sol.is_closed:
        tr = np.linspace(sol.tof_s, sol.period_s, 400)
        pts = np.array([sol.state_at(t)[0] for t in tr]).T
    elif frame == "eci":
        pts = orbit_curve(sol.r1, sol.v1, MU_EARTH, n_points=400,
                          max_radius=1.2 * np.max(np.linalg.norm(track["eci"], axis=1)))
    else:
        pts = track["ecef"].T
    fig.add_trace(go.Scatter3d(x=pts[0], y=pts[1], z=pts[2], mode="lines", name="Orbit",
                               line=dict(color=ACCENT, width=4), hoverinfo="skip"))

    # The Lambert arc actually flown between the two solved observations.
    ta = np.linspace(0, sol.tof_s, 120)
    arc = np.array([sol.state_at(t)[0] for t in ta])
    arc = _to_frame(arc, sol.gmst_at(ta), frame)
    fig.add_trace(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines",
                               name="Transfer arc", line=dict(color=ARC, width=7), hoverinfo="skip"))

    # Observations.
    ob = np.array([_to_frame(o["r_eci"], o["gmst_deg"], frame) for o in observations])
    fig.add_trace(go.Scatter3d(
        x=ob[:, 0], y=ob[:, 1], z=ob[:, 2], mode="markers+text", name="Observations",
        text=[o["label"] for o in observations], textposition="top center",
        textfont=dict(size=12, color=TEXT),
        marker=dict(size=[6 if o["is_pair"] else 4 for o in observations],
                    color=[ARC if o["is_pair"] else "#ffffff" for o in observations],
                    line=dict(color=ARC, width=2)),
        hovertemplate="Obs %{text}<br>|r| = %{customdata:,.0f} km<extra></extra>",
        customdata=[np.linalg.norm(o["r_eci"]) for o in observations],
    ))

    layers = set(layers or ())
    extent = float(max(np.max(np.linalg.norm(pts.T, axis=1)), np.max(np.linalg.norm(arc, axis=1)),
                       np.max(np.linalg.norm(ob, axis=1)), R_EARTH))

    if "velocity" in layers:
        vmax = max(np.linalg.norm(sol.v1), np.linalg.norm(sol.v2))
        scale = 0.22 * extent / vmax
        for j, (r, v, t) in enumerate(((sol.r1, sol.v1, 0.0), (sol.r2, sol.v2, sol.tof_s))):
            g = sol.gmst_at(t)
            if frame == "eci":
                tail, vec = r, v * scale
            else:
                # Direction only: the ECEF velocity would add omega x r.
                tail = eci_to_ecef(r, g)
                vec = eci_to_ecef(v, g) * scale
            for tr in _arrow(tail, vec, VELOCITY, "Velocity", legend=j == 0):
                fig.add_trace(tr)

    if "apsides" in layers and frame == "eci":
        e_vec = sol.elements["e_vec"]
        e = sol.elements["e"]
        if e > 1e-6:
            p_hat = e_vec / e
            names, pos = ["Perigee"], [sol.rp_km * p_hat]
            if sol.is_closed:
                names.append("Apogee")
                pos.append(-sol.ra_km * p_hat)
            pos = np.array(pos)
            fig.add_trace(go.Scatter3d(
                x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text", text=names,
                textposition="bottom center", textfont=dict(size=11, color=MUTED),
                marker=dict(size=5, color=MUTED, symbol="diamond"), name="Apsides",
                hovertemplate="%{text}<extra></extra>"))
        h = sol.elements["h_vec"]
        n_vec = np.cross([0, 0, 1.0], h)
        if np.linalg.norm(n_vec) > 1e-8 * np.linalg.norm(h):
            n_hat = n_vec / np.linalg.norm(n_vec) * extent
            fig.add_trace(go.Scatter3d(x=[-n_hat[0], n_hat[0]], y=[-n_hat[1], n_hat[1]],
                                       z=[0, 0], mode="lines", name="Line of nodes",
                                       line=dict(color=MUTED, width=2, dash="dash"),
                                       hoverinfo="skip"))

    if "station" in layers and station_ecef is not None:
        st = np.asarray(station_ecef, dtype=float)
        sx, sy, sz, lx, ly, lz = [], [], [], [], [], []
        for o in observations:
            s = st if frame == "ecef" else np.array([*_rotz(st[0], st[1], o["gmst_deg"]), st[2]])
            r = _to_frame(o["r_eci"], o["gmst_deg"], frame)
            sx.append(s[0])
            sy.append(s[1])
            sz.append(s[2])
            lx += [s[0], r[0], None]
            ly += [s[1], r[1], None]
            lz += [s[2], r[2], None]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="Line of sight",
                                   line=dict(color=STATION, width=3, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", name="Ground station",
                                   marker=dict(size=5, color=STATION, symbol="square"),
                                   hovertemplate="Ground station<extra></extra>"))

    for tr in _triad(max(1.6 * R_EARTH, 0.28 * extent), frame):
        fig.add_trace(tr)

    src = track["eci"] if frame == "eci" else track["ecef"]
    fig.add_trace(go.Scatter3d(
        x=[src[k, 0]], y=[src[k, 1]], z=[src[k, 2]], mode="markers", name="Satellite",
        marker=dict(size=8, color=TEXT, line=dict(color="#ffffff", width=2)), uid="sat",
        hovertemplate="Satellite<extra></extra>"))

    # Equal explicit ranges on all three axes (see module docstring): a cube
    # centred on the bounding box of the Earth and everything drawn around it.
    everything = np.vstack([pts.T, arc, ob, [[-R_EARTH] * 3, [R_EARTH] * 3]])
    lo, hi = everything.min(axis=0), everything.max(axis=0)
    centre, half = (lo + hi) / 2, 0.55 * float(np.max(hi - lo))

    def hidden(i):
        return dict(visible=False, range=[centre[i] - half, centre[i] + half], showspikes=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT),
        legend=dict(orientation="h", x=0, y=0, yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)", font=dict(size=12)),
        scene=dict(xaxis=hidden(0), yaxis=hidden(1), zaxis=hidden(2), aspectmode="cube",
                   camera=dict(eye=dict(x=1.0, y=1.0, z=0.55))),
        uirevision=f"3d-{frame}",
    )
    return fig


# ----------------------------------------------------------------------
# Ground track
# ----------------------------------------------------------------------
def _break_at_dateline(lon, lat):
    xs, ys = [lon[0]], [lat[0]]
    for i in range(1, len(lon)):
        if abs(lon[i] - lon[i - 1]) > 180:
            xs.append(None)
            ys.append(None)
        xs.append(lon[i])
        ys.append(lat[i])
    return xs, ys


def figure_ground_track(track, observations, station_ecef=None, k=0):
    fig = go.Figure()
    lat = np.arange(-90, 91)[:, None]
    surface = np.where(LAND[::-1], np.where(np.abs(lat) >= 66, 2, 1), 0)  # ocean / land / ice
    fig.add_trace(go.Heatmap(
        z=surface, x0=-180, dx=1, y0=-90, dy=1, zmin=0, zmax=2, showscale=False, hoverinfo="skip",
        colorscale=[[0, MAP_OCEAN], [0.33, MAP_OCEAN], [0.34, MAP_LAND], [0.66, MAP_LAND],
                    [0.67, MAP_ICE], [1, MAP_ICE]]))
    fig.add_trace(go.Scatter(x=COAST_LON, y=COAST_LAT, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(40, 70, 90, 0.5)", width=0.8), showlegend=False))
    xs, ys = _break_at_dateline(track["lon"], track["lat"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Ground track",
                             line=dict(color=ACCENT, width=2.5), hoverinfo="skip"))

    if observations:
        ll = [ecef_to_latlon(eci_to_ecef(o["r_eci"], o["gmst_deg"]))[:2] for o in observations]
        fig.add_trace(go.Scatter(
            x=[p[1] for p in ll], y=[p[0] for p in ll], mode="markers+text", name="Observations",
            text=[o["label"] for o in observations], textposition="top center",
            textfont=dict(size=12, color=TEXT),
            marker=dict(size=10, color=[ARC if o["is_pair"] else "#ffffff" for o in observations],
                        line=dict(color=ARC, width=2)),
            hovertemplate="Obs %{text}<br>%{y:.2f}°, %{x:.2f}°<extra></extra>"))

    if station_ecef is not None:
        lat, lon, _ = ecef_to_latlon(np.asarray(station_ecef, dtype=float))
        fig.add_trace(go.Scatter(x=[lon], y=[lat], mode="markers", name="Ground station",
                                 marker=dict(size=12, color=STATION, symbol="triangle-up",
                                             line=dict(color="#ffffff", width=1.5)),
                                 hovertemplate="Ground station<br>%{y:.2f}°, %{x:.2f}°<extra></extra>"))

    fig.add_trace(go.Scatter(x=[track["lon"][k]], y=[track["lat"][k]], mode="markers",
                             name="Satellite", uid="sat",
                             marker=dict(size=12, color=TEXT, line=dict(color="#ffffff", width=2)),
                             hovertemplate="Satellite<extra></extra>"))

    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=30), paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT, size=11),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=12)),
        xaxis=dict(range=[-180, 180], dtick=30, gridcolor="rgba(255,255,255,0.6)", zeroline=False,
                   title=dict(text="Longitude (°)", font=dict(color=MUTED)), layer="above traces",
                   constrain="domain"),
        yaxis=dict(range=[-90, 90], dtick=30, gridcolor="rgba(255,255,255,0.6)", zeroline=False,
                   scaleanchor="x", scaleratio=1, layer="above traces", constrain="domain",
                   title=dict(text="Latitude (°)", font=dict(color=MUTED))),
        uirevision="ground",
    )
    return fig
