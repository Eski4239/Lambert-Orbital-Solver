"""
Plotly figures for the Earth-orbit page: 3D view (ECI or ECEF) and ground
track. Pure functions of an OrbitSolution plus display options.

Animation: every figure carries a trace with uid "sat" (the satellite) and,
in 3D, "coast" (coastlines). The browser moves those two traces directly
(app/assets/animate.js) from the samples in `sample_track`, so scrubbing the
time slider never round-trips to Python.
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
TEXT = "#17202b"
MUTED = "#667383"
GRID = "#e6eaef"
OCEAN = "#dfe9f5"
COAST = "#7d8b9c"
STATION = "#1f9d55"

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


COAST_LON, COAST_LAT = _load_coastlines()


def _coast_ecef(radius=R_EARTH * 1.002):
    lon = np.array([np.nan if v is None else v for v in COAST_LON])
    lat = np.array([np.nan if v is None else v for v in COAST_LAT])
    lo, la = np.radians(lon), np.radians(lat)
    return (radius * np.cos(la) * np.cos(lo), radius * np.cos(la) * np.sin(lo),
            radius * np.sin(la))


COAST_ECEF = _coast_ecef()


def coast_payload():
    """Coastline ECEF coordinates for the browser (NaN -> None breaks)."""
    def clean(a):
        return [None if np.isnan(v) else round(float(v), 1) for v in a]
    return {"x": clean(COAST_ECEF[0]), "y": clean(COAST_ECEF[1]), "z": clean(COAST_ECEF[2])}


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
    u, v = np.mgrid[0:2 * np.pi:61j, 0:np.pi:31j]
    sphere = go.Surface(
        x=R_EARTH * np.cos(u) * np.sin(v), y=R_EARTH * np.sin(u) * np.sin(v),
        z=R_EARTH * np.cos(v), surfacecolor=np.zeros_like(u),
        colorscale=[[0, OCEAN], [1, OCEAN]], showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.85, diffuse=0.25, specular=0.05), name="Earth",
        showlegend=False,
    )
    # Latitude circles are rotation-invariant, so they never need animating.
    gx, gy, gz = [], [], []
    for lat in (-60, -30, 0, 30, 60):
        th = np.linspace(0, 2 * np.pi, 91)
        rr = R_EARTH * 1.001 * np.cos(np.radians(lat))
        gx += list(rr * np.cos(th)) + [None]
        gy += list(rr * np.sin(th)) + [None]
        gz += [R_EARTH * 1.001 * np.sin(np.radians(lat))] * 91 + [None]
    grat = go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color="#c5d3e3", width=1))

    cx, cy, cz = COAST_ECEF
    if frame == "eci":
        cx, cy = _rotz(cx, cy, gmst_deg)
    coast = go.Scatter3d(x=cx, y=cy, z=cz, mode="lines", hoverinfo="skip", showlegend=False,
                         line=dict(color=COAST, width=1.5), uid="coast", connectgaps=False)
    return [sphere, grat, coast]


def _to_frame(r_eci, gmst_deg, frame):
    return r_eci if frame == "eci" else eci_to_ecef(r_eci, gmst_deg)


def _arrow(tail, vec, color, name):
    head = tail + vec
    line = go.Scatter3d(x=[tail[0], head[0]], y=[tail[1], head[1]], z=[tail[2], head[2]],
                        mode="lines", line=dict(color=color, width=5), name=name,
                        hoverinfo="skip", showlegend=False)
    cone = go.Cone(x=[head[0]], y=[head[1]], z=[head[2]], u=[vec[0]], v=[vec[1]], w=[vec[2]],
                   anchor="tip", sizemode="absolute", sizeref=np.linalg.norm(vec) * 0.25,
                   colorscale=[[0, color], [1, color]], showscale=False, hoverinfo="skip")
    return [line, cone]


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
                               line=dict(color=ACCENT, width=3), hoverinfo="skip"))

    # The Lambert arc actually flown between the two solved observations.
    ta = np.linspace(0, sol.tof_s, 120)
    arc = np.array([sol.state_at(t)[0] for t in ta])
    arc = _to_frame(arc, sol.gmst_at(ta), frame)
    fig.add_trace(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines",
                               name="Transfer arc", line=dict(color=ARC, width=6), hoverinfo="skip"))

    # Observations.
    ob = np.array([_to_frame(o["r_eci"], o["gmst_deg"], frame) for o in observations])
    fig.add_trace(go.Scatter3d(
        x=ob[:, 0], y=ob[:, 1], z=ob[:, 2], mode="markers+text", name="Observations",
        text=[o["label"] for o in observations], textposition="top center",
        textfont=dict(size=11, color=TEXT),
        marker=dict(size=[6 if o["is_pair"] else 4 for o in observations],
                    color=[ARC if o["is_pair"] else "#ffffff" for o in observations],
                    line=dict(color=ARC, width=2)),
        hovertemplate="%{text}<br>|r| = %{customdata:,.0f} km<extra></extra>",
        customdata=[np.linalg.norm(o["r_eci"]) for o in observations],
    ))

    layers = set(layers or ())
    extent = float(np.max(np.linalg.norm(pts.T, axis=1)))

    if "velocity" in layers:
        vmax = max(np.linalg.norm(sol.v1), np.linalg.norm(sol.v2))
        scale = 0.22 * extent / vmax
        for r, v, t in ((sol.r1, sol.v1, 0.0), (sol.r2, sol.v2, sol.tof_s)):
            g = sol.gmst_at(t)
            if frame == "eci":
                tail, vec = r, v * scale
            else:
                # Direction only: the ECEF velocity would add omega x r.
                tail = eci_to_ecef(r, g)
                vec = eci_to_ecef(v, g) * scale
            for tr in _arrow(tail, vec, "#6d4bd1", "Velocity"):
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
            sx.append(s[0]); sy.append(s[1]); sz.append(s[2])
            lx += [s[0], r[0], None]; ly += [s[1], r[1], None]; lz += [s[2], r[2], None]
        fig.add_trace(go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", name="Line of sight",
                                   line=dict(color=STATION, width=2, dash="dot"), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers", name="Ground station",
                                   marker=dict(size=4, color=STATION, symbol="square"),
                                   hovertemplate="Ground station<extra></extra>"))

    src = track["eci"] if frame == "eci" else track["ecef"]
    fig.add_trace(go.Scatter3d(
        x=[src[k, 0]], y=[src[k, 1]], z=[src[k, 2]], mode="markers", name="Satellite",
        marker=dict(size=7, color=TEXT, line=dict(color="#ffffff", width=2)), uid="sat",
        hovertemplate="Satellite<extra></extra>"))

    axis_names = ("X", "Y", "Z")
    suffix = " (ECI, km)" if frame == "eci" else " (ECEF, km)"

    def axis(i):
        return dict(title=dict(text=axis_names[i] + suffix, font=dict(size=11, color=MUTED)),
                    showbackground=False, gridcolor=GRID, zeroline=False,
                    tickfont=dict(size=10, color=MUTED), showspikes=False, nticks=5)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="#ffffff",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT),
        legend=dict(orientation="h", x=0, y=1.0, bgcolor="rgba(255,255,255,0.8)",
                    font=dict(size=12)),
        scene=dict(xaxis=axis(0), yaxis=axis(1), zaxis=axis(2), aspectmode="data",
                   camera=dict(eye=dict(x=1.35, y=1.35, z=0.75))),
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
    fig.add_trace(go.Scatter(x=COAST_LON, y=COAST_LAT, mode="lines", hoverinfo="skip",
                             line=dict(color=COAST, width=1), showlegend=False))
    xs, ys = _break_at_dateline(track["lon"], track["lat"])
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Ground track",
                             line=dict(color=ACCENT, width=2), hoverinfo="skip"))

    if observations:
        ll = [ecef_to_latlon(eci_to_ecef(o["r_eci"], o["gmst_deg"]))[:2] for o in observations]
        fig.add_trace(go.Scatter(
            x=[p[1] for p in ll], y=[p[0] for p in ll], mode="markers+text", name="Observations",
            text=[o["label"] for o in observations], textposition="top center",
            textfont=dict(size=11, color=TEXT),
            marker=dict(size=9, color=[ARC if o["is_pair"] else "#ffffff" for o in observations],
                        line=dict(color=ARC, width=2)),
            hovertemplate="%{text}<br>%{y:.2f}°, %{x:.2f}°<extra></extra>"))

    if station_ecef is not None:
        lat, lon, _ = ecef_to_latlon(np.asarray(station_ecef, dtype=float))
        fig.add_trace(go.Scatter(x=[lon], y=[lat], mode="markers", name="Ground station",
                                 marker=dict(size=10, color=STATION, symbol="triangle-up"),
                                 hovertemplate="Ground station<br>%{y:.2f}°, %{x:.2f}°<extra></extra>"))

    fig.add_trace(go.Scatter(x=[track["lon"][k]], y=[track["lat"][k]], mode="markers",
                             name="Satellite", uid="sat",
                             marker=dict(size=11, color=TEXT, line=dict(color="#ffffff", width=2)),
                             hovertemplate="Satellite<extra></extra>"))

    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=30), paper_bgcolor="#ffffff", plot_bgcolor="#f7f9fc",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT, size=11),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=12)),
        xaxis=dict(range=[-180, 180], dtick=30, gridcolor=GRID, zeroline=False,
                   title=dict(text="Longitude (°)", font=dict(color=MUTED))),
        yaxis=dict(range=[-90, 90], dtick=30, gridcolor=GRID, zeroline=False,
                   scaleanchor="x", scaleratio=1,
                   title=dict(text="Latitude (°)", font=dict(color=MUTED))),
        uirevision="ground",
    )
    return fig
