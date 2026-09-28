"""
Figures and browser data for the Near-Earth Objects page.

Solar-system view: a Sun-centred 3D scene in AU (ecliptic J2000) on a
space backdrop. Static parts (orbits, Sun, reference rings, transfer arc)
are drawn here; the moving bodies (uids "planets", "target", "cloud",
"craft") are positioned in the browser (app/assets/solar.js) for any date
on the slider, from compact orbit "packs":

    r(t) = a (cos E - e) P + a sqrt(1 - e^2) sin E Q,   M = M0 + n (t - epoch)

i.e. the same two-body model as core.kepler (P, Q from
core.kepler.perifocal_basis). tests/test_neo_page.py checks the browser
positions against core.kepler.position_at.
"""

import numpy as np
import plotly.graph_objects as go

from core.constants import AU, MU_SUN
from core.kepler import perifocal_basis, solve_kepler
from helio.planets import PLANETS, planet_elements

INNER_PLANETS = ("Mercury", "Venus", "Earth", "Mars")
TARGET = "#ff8a3d"
TRANSFER = "#4cc9f0"
SUN = "#ffd166"
TEXT = "#dfe7f3"
MUTED = "#8b9ab3"
PANEL = "#070b16"

MAX_CLOUD = 1500


# ----------------------------------------------------------------------
# Orbit packs (columnar, AU / radians / days) for the browser propagator
# ----------------------------------------------------------------------
def orbit_pack(a_km, e, i, raan, argp, M0_deg, epoch_jd):
    a_km, e = np.atleast_1d(a_km).astype(float), np.atleast_1d(e).astype(float)
    P, Q = perifocal_basis(np.atleast_1d(i), np.atleast_1d(raan), np.atleast_1d(argp))
    n = np.sqrt(MU_SUN / a_km**3) * 86400.0  # rad/day

    def col(x, d=9):
        return np.round(np.broadcast_to(x, a_km.shape), d).tolist()
    return {"a": col(a_km / AU), "e": col(e), "n": col(n, 12),
            "M0": col(np.radians(np.atleast_1d(M0_deg))), "ep": col(np.atleast_1d(epoch_jd), 5),
            "Px": col(P[:, 0]), "Py": col(P[:, 1]), "Pz": col(P[:, 2]),
            "Qx": col(Q[:, 0]), "Qy": col(Q[:, 1]), "Qz": col(Q[:, 2])}


def catalog_pack(df):
    pack = orbit_pack(df["a"].to_numpy() * AU, df["e"].to_numpy(), df["i"].to_numpy(),
                      df["om"].to_numpy(), df["w"].to_numpy(), df["ma"].to_numpy(),
                      df["epoch"].to_numpy())
    pack["pdes"] = df["pdes"].tolist()      # lets a click in 3D select the object
    pack["label"] = df["display"].tolist()
    return pack


def planets_pack(jd):
    """Planet elements frozen at `jd` (their secular drift over a few years is negligible here)."""
    els = [planet_elements(p, jd) for p in INNER_PLANETS]
    a, e, i, node, argp, M = (np.array(x) for x in zip(*els))
    pack = orbit_pack(a, e, i, node, argp, M, np.full(len(a), jd))
    pack["name"] = list(INNER_PLANETS)
    return pack


def ellipse_au(a_au, e, i, raan, argp, n=361):
    """Points (3, n) in AU along an elliptic orbit."""
    P, Q = perifocal_basis(i, raan, argp)
    E = np.linspace(0, 2 * np.pi, n)
    x, y = a_au * (np.cos(E) - e), a_au * np.sqrt(1 - e**2) * np.sin(E)
    return np.outer(P, x) + np.outer(Q, y)


# ----------------------------------------------------------------------
# Solar-system scene
# ----------------------------------------------------------------------
def _sun():
    halo = [(34, 0.10), (20, 0.25), (11, 1.0)]
    return [go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", hoverinfo="skip" if k else "text",
                         hovertext="Sun", showlegend=False,
                         marker=dict(size=s, color=SUN, opacity=o, line=dict(width=0)))
            for k, (s, o) in enumerate(halo)][::-1]


def _rings(radii):
    th = np.linspace(0, 2 * np.pi, 181)
    x, y, z = [], [], []
    for r in radii:
        x += list(r * np.cos(th)) + [None]
        y += list(r * np.sin(th)) + [None]
        z += [0.0] * 181 + [None]
    return [go.Scatter3d(x=x, y=y, z=z, mode="lines", hoverinfo="skip", showlegend=False,
                         line=dict(color="rgba(160, 180, 220, 0.10)", width=1)),
            go.Scatter3d(x=[r * np.cos(-0.35) for r in radii], y=[r * np.sin(-0.35) for r in radii],
                         z=[0] * len(radii), mode="text", text=[f"{r:g} AU" for r in radii],
                         hoverinfo="skip", showlegend=False,
                         textfont=dict(size=10, color="rgba(160, 180, 220, 0.45)"))]


def pack_positions(pack, jd):
    """Positions (3, n) in AU of every orbit in a pack at jd (server side, core.kepler)."""
    if not pack or not pack["a"]:
        return np.zeros((3, 0))
    a = np.array(pack["a"]) * AU
    P = np.array([pack["Px"], pack["Py"], pack["Pz"]])
    Q = np.array([pack["Qx"], pack["Qy"], pack["Qz"]])
    e = np.array(pack["e"])
    M = np.array(pack["M0"]) + np.array(pack["n"]) * (jd - np.array(pack["ep"]))
    E = solve_kepler(M, e)
    return (P * (a * (np.cos(E) - e)) + Q * (a * np.sqrt(1 - e**2) * np.sin(E))) / AU


def solar_figure(jd, planets=None, target=None, cloud=None, transfer=None):
    """
    Bodies are positioned for `jd` here; the browser moves them afterwards.
    planets / cloud: orbit packs. target: dict with name, short, a (AU), e,
    i, om, w (deg) and "pack", or None. transfer: dict with "path" (n, 3) AU,
    "dep_jd", "tof", or None.
    """
    planets_xyz = pack_positions(planets, jd)
    cloud_xyz = pack_positions(cloud, jd)
    cloud_count = cloud_xyz.shape[1]
    fig = go.Figure()
    for tr in _sun():
        fig.add_trace(tr)

    reach = 1.7
    if target is not None:
        reach = max(reach, target["a"] * (1 + target["e"]))
    rings = [r for r in (1, 2, 3, 4, 5) if r <= reach * 1.05]
    for tr in _rings(rings):
        fig.add_trace(tr)

    for name in INNER_PLANETS:
        a, e, i, node, argp, _ = planet_elements(name, jd)
        pts = ellipse_au(a / AU, e, i, node, argp)
        fig.add_trace(go.Scatter3d(x=pts[0], y=pts[1], z=pts[2], mode="lines", hoverinfo="skip",
                                   showlegend=False, line=dict(color=PLANETS[name].color, width=2),
                                   opacity=0.55))

    labels = (cloud or {}).get("label", [])
    fig.add_trace(go.Scatter3d(
        x=cloud_xyz[0], y=cloud_xyz[1], z=cloud_xyz[2], mode="markers", uid="cloud",
        name="Catalog asteroids", visible=cloud_count > 0,
        customdata=list(zip((cloud or {}).get("pdes", []), labels)),
        hovertemplate="%{customdata[1]}<br><i>click to select</i><extra></extra>",
        marker=dict(size=3, color="rgba(232, 196, 140, 0.6)", line=dict(width=0))))

    if target is not None:
        pts = ellipse_au(target["a"], target["e"], target["i"], target["om"], target["w"])
        fig.add_trace(go.Scatter3d(x=pts[0], y=pts[1], z=pts[2], mode="lines", name=target["name"],
                                   hoverinfo="skip", line=dict(color=TARGET, width=4)))

    if transfer is not None:
        p = np.asarray(transfer["path"])
        fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines", name="Transfer",
                                   hoverinfo="skip", line=dict(color=TRANSFER, width=4, dash="solid")))
        fig.add_trace(go.Scatter3d(
            x=[p[0, 0], p[-1, 0]], y=[p[0, 1], p[-1, 1]], z=[p[0, 2], p[-1, 2]], mode="markers+text",
            text=["Departure", "Arrival"], textposition="top center", showlegend=False,
            textfont=dict(size=11, color=TRANSFER), hoverinfo="skip",
            marker=dict(size=4, color=TRANSFER, symbol="diamond")))

    # Moving bodies: placeholders the browser positions for the slider date.
    fig.add_trace(go.Scatter3d(
        x=planets_xyz[0], y=planets_xyz[1], z=planets_xyz[2], mode="markers+text", uid="planets",
        text=list(INNER_PLANETS),
        textposition="top center", textfont=dict(size=11, color=TEXT), showlegend=False,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=[4, 5, 6, 5], color=[PLANETS[p].color for p in INNER_PLANETS],
                    line=dict(color="rgba(255,255,255,0.6)", width=1))))
    if target is not None:
        t_xyz = pack_positions(target["pack"], jd)
        fig.add_trace(go.Scatter3d(
            x=t_xyz[0], y=t_xyz[1], z=t_xyz[2], mode="markers+text", uid="target", text=[target["short"]],
            textposition="top center", textfont=dict(size=12, color=TARGET), showlegend=False,
            hovertemplate=target["name"] + "<extra></extra>",
            marker=dict(size=6, color=TARGET, line=dict(color="#ffffff", width=1.5))))
    if transfer is not None:
        p = np.asarray(transfer["path"])
        f = (jd - transfer["dep_jd"]) / transfer["tof"]
        c = p[int(round(f * (len(p) - 1)))] if 0 <= f <= 1 else [None] * 3
        fig.add_trace(go.Scatter3d(
            x=[c[0]], y=[c[1]], z=[c[2]], mode="markers", uid="craft", name="Spacecraft",
            hovertemplate="Spacecraft<extra></extra>",
            marker=dict(size=5, color="#ffffff", symbol="diamond", line=dict(color=TRANSFER, width=2))))

    half = 1.12 * reach
    axis = dict(visible=False, range=[-half, half], showspikes=False)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT),
        legend=dict(orientation="h", x=0, y=0, yanchor="bottom", bgcolor="rgba(7,11,22,0.6)",
                    font=dict(size=12, color=TEXT), itemsizing="constant"),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=dict(axis, range=[-half, half]), aspectmode="cube",
                   bgcolor="rgba(0,0,0,0)", camera=dict(eye=dict(x=0.35, y=-0.95, z=0.75))),
        uirevision="solar",
        showlegend=transfer is not None or target is not None,
    )
    return fig


# ----------------------------------------------------------------------
# Porkchop
# ----------------------------------------------------------------------
METRICS = {
    "dv_total": ("Total Δv (km/s)", "Rendezvous Δv"),
    "c3": ("Launch C3 (km²/s²)", "Launch energy C3"),
    "vinf_arr": ("Arrival v∞ (km/s)", "Arrival speed"),
}


def color_limits(z):
    """
    Colour range of a porkchop grid: from the minimum up to the 60th
    percentile or 3x the minimum (+1), whichever is lower. Cells above it are
    treated as outside the viable region (not coloured, flagged if picked).
    """
    finite = z[np.isfinite(z)]
    if not finite.size:
        return 0.0, 1.0
    zmin = float(finite.min())
    return zmin, float(min(np.percentile(finite, 60), zmin * 3 + 1))


def porkchop_figure(dep_dates, tof_days, z, metric, best=None, picked=None):
    """dep_dates: ISO date strings; z: (n_tof, n_dep); best/picked: (date, tof)."""
    zmin, zmax = color_limits(z)
    # Cells above the colour scale are left out, so only the useful
    # launch windows are coloured and the rest shows the space backdrop.
    z = np.where(z <= zmax, z, np.nan)
    title, _ = METRICS[metric]
    fig = go.Figure(go.Contour(
        x=dep_dates, y=tof_days, z=z, zmin=zmin, zmax=zmax, zauto=False, ncontours=18,
        colorscale="Viridis", reversescale=True, connectgaps=False,
        contours=dict(coloring="heatmap", showlines=True),
        line=dict(width=0.6, color="rgba(3, 6, 13, 0.45)"),
        colorbar=dict(title=dict(text=title, side="right", font=dict(color=TEXT)),
                      tickfont=dict(color=MUTED), outlinewidth=0, thickness=14),
        hovertemplate="Depart %{x}<br>Flight %{y:.0f} days<br>" + title + " %{z:.2f}<extra></extra>",
    ))
    if best is not None:
        fig.add_trace(go.Scatter(x=[best[0]], y=[best[1]], mode="markers+text", text=["Best"],
                                 textposition="top center", textfont=dict(color="#ffffff", size=12),
                                 marker=dict(symbol="star", size=16, color="#ffffff",
                                             line=dict(color=PANEL, width=1)),
                                 hoverinfo="skip", showlegend=False))
    if picked is not None:
        fig.add_trace(go.Scatter(x=[picked[0]], y=[picked[1]], mode="markers",
                                 marker=dict(symbol="circle-open", size=18, color=TRANSFER,
                                             line=dict(width=3)),
                                 hoverinfo="skip", showlegend=False))
    grid = dict(gridcolor="rgba(160, 180, 220, 0.12)", zeroline=False, color=MUTED)
    fig.update_layout(
        margin=dict(l=60, r=10, t=10, b=50), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color=TEXT),
        xaxis=dict(title="Departure date", **grid), yaxis=dict(title="Time of flight (days)", **grid),
        uirevision="porkchop",
    )
    return fig


def starfield_svg(n=260, seed=7, w=1200, h=700):
    """Static star backdrop as an SVG data URI (CSS background of the space panel)."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, w, n), rng.uniform(0, h, n)
    r = rng.choice([0.5, 0.7, 0.9, 1.3], n, p=[0.45, 0.3, 0.18, 0.07])
    o = rng.uniform(0.25, 0.9, n)
    stars = "".join(f"<circle cx='{a:.0f}' cy='{b:.0f}' r='{c}' fill='white' fill-opacity='{d:.2f}'/>"
                    for a, b, c, d in zip(x, y, r, o))
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}'>{stars}</svg>"
    return "url(\"data:image/svg+xml;utf8," + svg.replace("#", "%23") + "\")"
