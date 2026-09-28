"""
Near-Earth Objects page: a Sun-centred scope built on the same core.

Catalog (helio.neo_catalog) -> pick an asteroid -> see it among the inner
planets on any date (browser propagation, assets/solar.js) -> design an
Earth -> asteroid transfer on a porkchop plot (helio.porkchop, which calls
core.lamsolbert with the Sun's mu).
"""

from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from dash import (ClientsideFunction, Input, Output, State, callback, clientside_callback,
                  ctx, dash_table, dcc, html, no_update)

from app.neo_figures import (MAX_CLOUD, METRICS, catalog_pack, orbit_pack, planets_pack,
                             porkchop_figure, solar_figure, starfield_svg)
from core.constants import AU
from core.time_utils import julian_date
from helio.neo_catalog import (CLASS_NAMES, elements_of, estimated_diameter_km, fetch_catalog,
                               filter_catalog, load_catalog, snapshot_info)
from helio.porkchop import compute_porkchop, solve_transfer

WINDOW_DAYS = 4 * 365          # date slider span, from today
DEFAULT_OBJECT = "99942"       # Apophis: Earth flyby on 2029-04-13, inside the window
PAGE_SIZE = 10

_state = {"df": load_catalog(), "info": snapshot_info()}


def _today_jd():
    now = datetime.now(timezone.utc)
    return julian_date(datetime(now.year, now.month, now.day, tzinfo=timezone.utc))


def jd_to_date(jd):
    return (datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=jd - 2451545.0)).strftime("%Y-%m-%d")


def date_to_jd(text):
    return julian_date(datetime.strptime(text.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc))


def _row(pdes):
    df = _state["df"]
    hit = df[df["pdes"] == pdes]
    return None if hit.empty else hit.iloc[0]


def _short_name(row):
    return row["name"] if isinstance(row["name"], str) and row["name"] else row["pdes"]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def _readout(label, id_, unit=""):
    return html.Div(className="readout", children=[
        html.Div(label, className="readout-label"),
        html.Div(className="readout-value", children=[html.Span("—", id=id_),
                                                        html.Span(unit, className="readout-unit")]),
    ])


def layout():
    t0 = _today_jd()
    catalog_card = html.Div(className="card", children=[
        html.Div("Near-Earth asteroids", className="card-title"),
        dcc.Input(id="neo-search", type="text", debounce=False, className="inp search",
                  placeholder="Search name or designation, e.g. Apophis"),
        html.Div(className="filter-row", children=[
            dcc.Checklist(id="neo-classes", className="chips", inline=True,
                          options=[{"label": v, "value": k} for k, v in
                                   (("IEO", "Atira"), ("ATE", "Aten"), ("APO", "Apollo"), ("AMO", "Amor"))],
                          value=["IEO", "ATE", "APO", "AMO"]),
            dcc.Checklist(id="neo-pha", className="chips chips-warn", inline=True,
                          options=[{"label": "Hazardous only", "value": "pha"}], value=[]),
        ]),
        dash_table.DataTable(
            id="neo-table",
            columns=[{"id": "display", "name": "Object"}, {"id": "class_name", "name": "Class"},
                     {"id": "a", "name": "a AU", "type": "numeric", "format": {"specifier": ".3f"}},
                     {"id": "e", "name": "e", "type": "numeric", "format": {"specifier": ".3f"}},
                     {"id": "i", "name": "i °", "type": "numeric", "format": {"specifier": ".1f"}},
                     {"id": "H", "name": "H", "type": "numeric", "format": {"specifier": ".1f"}}],
            page_action="custom", page_current=0, page_size=PAGE_SIZE,
            sort_action="custom", sort_mode="single", sort_by=[{"column_id": "H", "direction": "asc"}],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px 7px", "border": "1px solid #e6eaef", "textAlign": "right",
                        "fontSize": "12px", "cursor": "pointer", "whiteSpace": "nowrap"},
            style_cell_conditional=[{"if": {"column_id": "display"}, "textAlign": "left",
                                     "maxWidth": "190px", "overflow": "hidden", "textOverflow": "ellipsis"},
                                    {"if": {"column_id": "class_name"}, "textAlign": "left"}],
            style_header={"backgroundColor": "#eef3fa", "fontWeight": 700, "borderBottom": "2px solid #c9d6ea"},
        ),
        html.Div(className="row between table-foot", children=[
            html.Span(id="neo-count", className="hint"),
            html.Span(className="legend-pha", children=[html.Span(className="dot-pha"), "potentially hazardous"]),
        ]),
        html.Div(className="row between table-foot", children=[
            html.Span(id="neo-snapshot", className="hint", children=_state["info"]),
            html.Button("Refresh from JPL", id="neo-refresh", className="btn btn-small"),
        ]),
    ])

    object_card = html.Div(className="card readouts-card", children=[
        html.Div(className="object-head", children=[
            html.Span(id="obj-name", className="object-name"),
            html.Span(id="obj-badges", className="row"),
        ]),
        html.Div(className="readouts", children=[
            _readout("Semi-major axis", "obj-a", "AU"),
            _readout("Eccentricity", "obj-e"),
            _readout("Inclination", "obj-i"),
            _readout("Perihelion", "obj-q", "AU"),
            _readout("Aphelion", "obj-Q", "AU"),
            _readout("Period", "obj-per", "yr"),
        ]),
        html.Div(id="obj-chips", className="subline"),
    ])

    tof_marks = {d: f"{d}" for d in (50, 200, 400, 600, 800)}
    space_card = html.Div(className="card space-card", style={"backgroundImage": starfield_svg()}, children=[
        html.Div(className="view-head", children=[
            dcc.RadioItems(id="neo-view", className="seg seg-dark", inline=True, value="solar",
                           options=[{"label": "Solar system", "value": "solar"},
                                    {"label": "Mission design", "value": "mission"}]),
            dcc.Checklist(id="neo-show-cloud", className="layers layers-dark", inline=True,
                          options=[{"label": "Show catalog asteroids", "value": "on"}], value=["on"]),
        ]),
        html.Div(id="solar-pane", children=[
            dcc.Graph(id="solar-graph", className="graph", config={"displaylogo": False}),
            html.Div(className="timebar timebar-dark", children=[
                html.Button("▶", id="neo-play", className="btn btn-icon btn-primary", title="Play / pause"),
                html.Div(className="slider-wrap", children=[
                    dcc.Slider(id="neo-day", min=0, max=WINDOW_DAYS, step=1, value=0, updatemode="drag",
                               allow_direct_input=False,
                               marks={d: jd_to_date(t0 + d)[:4] for d in range(0, WINDOW_DAYS + 1, 365)}),
                ]),
                html.Div(id="neo-date", className="time-readout"),
            ]),
            dcc.Interval(id="neo-ticker", interval=60, disabled=True),
        ]),
        html.Div(id="mission-pane", style={"display": "none"}, children=[
            html.Div(className="mission-controls", children=[
                html.Div(className="ctl", children=[
                    html.Span("Departure from", className="ctl-label"),
                    dcc.Input(id="pc-start", value=jd_to_date(t0), debounce=True, className="inp inp-dark",
                              type="text"),
                ]),
                html.Div(className="ctl", children=[
                    html.Span("Window (days)", className="ctl-label"),
                    dcc.Input(id="pc-span", value="730", debounce=True, className="inp inp-dark", type="text"),
                ]),
                html.Div(className="ctl ctl-wide", children=[
                    html.Span("Time of flight (days)", className="ctl-label"),
                    dcc.RangeSlider(id="pc-tof", min=20, max=900, step=10, value=[40, 500], marks=tof_marks,
                                    allow_direct_input=False, updatemode="mouseup"),
                ]),
                dcc.RadioItems(id="pc-metric", className="seg seg-dark", inline=True, value="dv_total",
                               options=[{"label": v[1], "value": k} for k, v in METRICS.items()]),
            ]),
            dcc.Loading(type="dot", color="#4cc9f0", children=dcc.Graph(
                id="porkchop-graph", className="graph graph-short", config={"displaylogo": False})),
            html.Div(className="transfer-bar", children=[
                html.Div(id="transfer-chips", className="subline subline-dark"),
                html.Button("Show in solar system →", id="show-transfer", className="btn btn-primary"),
            ]),
            html.P("Click anywhere on the plot to pick a transfer. Each cell is one Lambert solution "
                   "(Earth at departure → asteroid at arrival) using the Sun's gravity.",
                   className="hint hint-dark"),
        ]),
    ])

    return html.Div(className="page", children=[
        html.Div(className="col", children=[catalog_card]),
        html.Div(className="col", children=[object_card, space_card]),
        dcc.Store(id="neo-selected", data=DEFAULT_OBJECT),
        dcc.Store(id="neo-cloud"),
        dcc.Store(id="neo-transfer"),
        dcc.Store(id="solar-scene"),
        dcc.Store(id="neo-t0", data=t0),
    ])


# ----------------------------------------------------------------------
# Catalog
# ----------------------------------------------------------------------
@callback(
    Output("neo-table", "data"), Output("neo-table", "page_count"), Output("neo-count", "children"),
    Output("neo-cloud", "data"), Output("neo-table", "style_data_conditional"),
    Output("neo-table", "page_current"),
    Input("neo-search", "value"), Input("neo-classes", "value"), Input("neo-pha", "value"),
    Input("neo-table", "page_current"), Input("neo-table", "sort_by"), Input("neo-selected", "data"),
    Input("neo-snapshot", "children"),
)
def update_table(text, classes, pha, page, sort_by, selected, _snapshot):
    df = filter_catalog(_state["df"], text, classes or [], bool(pha))
    filters_changed = ctx.triggered_id in ("neo-search", "neo-classes", "neo-pha")
    page = 0 if filters_changed else (page or 0)
    if sort_by:
        df = df.sort_values(sort_by[0]["column_id"], ascending=sort_by[0]["direction"] == "asc",
                            na_position="last")
    n = len(df)
    pages = max(1, -(-n // PAGE_SIZE))
    page = min(page, pages - 1)
    rows = df.iloc[page * PAGE_SIZE:(page + 1) * PAGE_SIZE]
    data = rows[["pdes", "display", "class_name", "a", "e", "i", "H"]].assign(
        pha=np.where(rows["pha"], "Y", "N")).to_dict("records")
    count = (f"{page * PAGE_SIZE + 1:,}–{page * PAGE_SIZE + len(rows):,} of {n:,} objects" if n
             else "No objects match")
    # Background cloud: the brightest (largest) matches, capped for speed.
    cloud_df = df.nsmallest(MAX_CLOUD, "H") if n > MAX_CLOUD else df
    styles = [
        {"if": {"filter_query": '{pha} = "Y"', "column_id": "display"}, "color": "#c2410c", "fontWeight": 600},
        {"if": {"filter_query": f'{{pdes}} = "{selected}"'}, "backgroundColor": "#e6effd",
         "borderTop": "1px solid #1d5fd1", "borderBottom": "1px solid #1d5fd1"},
        {"if": {"state": "active"}, "backgroundColor": "#e6effd", "border": "1px solid #1d5fd1"},
    ]
    return data, pages, count, catalog_pack(cloud_df), styles, page


@callback(Output("neo-selected", "data"), Input("neo-table", "active_cell"),
          State("neo-table", "data"), prevent_initial_call=True)
def select_object(cell, rows):
    if not cell or not rows or cell["row"] >= len(rows):
        return no_update
    return rows[cell["row"]]["pdes"]


@callback(Output("neo-snapshot", "children"), Input("neo-refresh", "n_clicks"),
          prevent_initial_call=True, running=[(Output("neo-refresh", "disabled"), True, False)])
def refresh_catalog(_n):
    try:
        _state["df"] = fetch_catalog()
        _state["info"] = snapshot_info()
        return _state["info"]
    except Exception as exc:  # noqa: BLE001 - network errors are shown, not raised
        return f"Refresh failed ({exc.__class__.__name__}); using {_state['info']}"


# ----------------------------------------------------------------------
# Selected object
# ----------------------------------------------------------------------
def _chip(label, value, cls="chip"):
    return html.Span([label, html.B(value)], className=cls)


@callback(
    Output("obj-name", "children"), Output("obj-badges", "children"),
    Output("obj-a", "children"), Output("obj-e", "children"), Output("obj-i", "children"),
    Output("obj-q", "children"), Output("obj-Q", "children"), Output("obj-per", "children"),
    Output("obj-chips", "children"),
    Input("neo-selected", "data"), Input("neo-snapshot", "children"),
)
def show_object(pdes, _snapshot):
    row = _row(pdes)
    if row is None:
        return ("No object selected",) + (no_update,) * 8
    badges = [html.Span(CLASS_NAMES.get(row["class"], row["class"]), className="badge")]
    if row["pha"]:
        badges.append(html.Span("Potentially hazardous", className="badge badge-warn"))
    if np.isfinite(row["diameter"]):
        size = f"{row['diameter']:,.2f} km" if row["diameter"] >= 1 else f"{row['diameter'] * 1000:,.0f} m"
    else:
        d = estimated_diameter_km(row["H"])
        size = f"≈ {d:,.1f} km (est.)" if d >= 1 else f"≈ {d * 1000:,.0f} m (est.)"
    chips = [
        _chip("Diameter", size),
        _chip("Abs. magnitude H", f"{row['H']:.2f}"),
        _chip("Earth MOID", f"{row['moid']:.4g} AU · {row['moid_ld']:.2g} lunar distances"),
        _chip("Elements epoch", jd_to_date(row["epoch"])),
    ]
    period = row["per_y"] if np.isfinite(row["per_y"]) else row["a"] ** 1.5
    return (row["display"], badges, f"{row['a']:.4f}", f"{row['e']:.4f}", f"{row['i']:.2f}°",
            f"{row['q']:.4f}", f"{row['ad']:.4f}", f"{period:.2f}", chips)


# ----------------------------------------------------------------------
# Views
# ----------------------------------------------------------------------
@callback(Output("solar-pane", "style"), Output("mission-pane", "style"),
          Output("neo-show-cloud", "style"), Input("neo-view", "value"))
def toggle_view(view):
    if view == "solar":
        return {}, {"display": "none"}, {}
    return {"display": "none"}, {}, {"display": "none"}


@callback(
    Output("solar-graph", "figure"), Output("solar-scene", "data"),
    Input("neo-selected", "data"), Input("neo-cloud", "data"), Input("neo-show-cloud", "value"),
    Input("neo-transfer", "data"), State("neo-day", "value"), State("neo-t0", "data"),
)
def render_solar(pdes, cloud, show_cloud, transfer, day, t0):
    jd = t0 + (day or 0)
    row = _row(pdes)
    planets = planets_pack(jd)
    target = None
    if row is not None:
        target = {"name": row["display"], "short": _short_name(row), "a": row["a"], "e": row["e"],
                  "i": row["i"], "om": row["om"], "w": row["w"],
                  "pack": orbit_pack(*elements_of(row))}
    cloud = cloud if show_cloud else None
    fig = solar_figure(jd, planets=planets, target=target, cloud=cloud, transfer=transfer)
    scene = {"t0": t0, "planets": planets, "target": target["pack"] if target else None,
             "cloud": cloud, "craft": None}
    if transfer:
        path = np.asarray(transfer["path"])
        scene["craft"] = {"dep_jd": transfer["dep_jd"], "tof": transfer["tof"],
                          "x": path[:, 0].tolist(), "y": path[:, 1].tolist(), "z": path[:, 2].tolist()}
    return fig, scene


@lru_cache(maxsize=32)
def _porkchop(pdes, start_jd, span, tof_min, tof_max):
    row = _row(pdes)
    return compute_porkchop(elements_of(row), start_jd, span, tof_min, tof_max)


def _transfer_payload(pdes, dep_jd, tof):
    row = _row(pdes)
    tr = solve_transfer(elements_of(row), dep_jd, tof)
    return {"pdes": pdes, "dep_jd": dep_jd, "tof": tof, "path": (tr.path(160) / AU).round(6).tolist(),
            "c3": tr.c3, "vinf_dep": tr.vinf_dep, "vinf_arr": tr.vinf_arr, "dv_total": tr.dv_total}


@callback(
    Output("porkchop-graph", "figure"), Output("neo-transfer", "data"),
    Output("pc-start", "className"), Output("pc-span", "className"),
    Input("neo-selected", "data"), Input("pc-start", "value"), Input("pc-span", "value"),
    Input("pc-tof", "value"), Input("pc-metric", "value"), Input("porkchop-graph", "clickData"),
    Input("neo-view", "value"), State("neo-transfer", "data"),
)
def render_porkchop(pdes, start, span, tof, metric, click, view, transfer):
    if view != "mission" or _row(pdes) is None:
        # Only compute when visible, but drop a transfer that belongs to another object.
        if transfer and transfer.get("pdes") != pdes:
            return no_update, None, no_update, no_update
        return no_update, no_update, no_update, no_update
    ok, bad = "inp inp-dark", "inp inp-dark invalid"
    try:
        start_jd = date_to_jd(start)
        cls_start = ok
    except (ValueError, AttributeError):
        return no_update, no_update, bad, no_update
    try:
        span_d = float(span)
        if not 10 <= span_d <= 3650:
            raise ValueError
    except (TypeError, ValueError):
        return no_update, no_update, cls_start, bad

    pc = _porkchop(pdes, round(start_jd, 1), span_d, float(tof[0]), float(tof[1]))
    z = getattr(pc, metric)
    best = pc.best(metric)
    dates = [jd_to_date(j) for j in pc.dep_jd]

    if ctx.triggered_id == "porkchop-graph" and click:
        pt = click["points"][0]
        dep_jd = date_to_jd(str(pt["x"])[:10])
        transfer = _transfer_payload(pdes, dep_jd, float(pt["y"]))
    elif best is not None and (not transfer or transfer.get("pdes") != pdes
                               or ctx.triggered_id != "neo-view"):
        transfer = _transfer_payload(pdes, best[0], best[1])

    picked = (jd_to_date(transfer["dep_jd"]), transfer["tof"]) if transfer else None
    fig = porkchop_figure(dates, pc.tof_days, z, metric,
                          best=(jd_to_date(best[0]), best[1]) if best else None, picked=picked)
    return fig, transfer, ok, ok


@callback(Output("transfer-chips", "children"), Input("neo-transfer", "data"))
def show_transfer(tr):
    if not tr:
        return []
    arr = jd_to_date(tr["dep_jd"] + tr["tof"])
    return [
        _chip("Depart", jd_to_date(tr["dep_jd"]), "chip chip-dark"),
        _chip("Arrive", arr, "chip chip-dark"),
        _chip("Flight", f"{tr['tof']:.0f} days", "chip chip-dark"),
        _chip("C3", f"{tr['c3']:.2f} km²/s²", "chip chip-dark"),
        _chip("Arrival v∞", f"{tr['vinf_arr']:.2f} km/s", "chip chip-dark"),
        _chip("Total Δv", f"{tr['dv_total']:.2f} km/s", "chip chip-dark chip-strong"),
    ]


@callback(Output("neo-view", "value"), Output("neo-day", "value"),
          Input("show-transfer", "n_clicks"), State("neo-transfer", "data"), State("neo-t0", "data"),
          prevent_initial_call=True)
def show_transfer_in_3d(_n, tr, t0):
    if not tr:
        return no_update, no_update
    return "solar", int(np.clip(round(tr["dep_jd"] - t0), 0, WINDOW_DAYS))


clientside_callback(ClientsideFunction("solar", "render"), Output("neo-date", "children"),
                    Input("neo-day", "value"), Input("solar-scene", "data"))

clientside_callback(ClientsideFunction("solar", "tick"), Output("neo-day", "value", allow_duplicate=True),
                    Input("neo-ticker", "n_intervals"), State("neo-day", "value"), State("neo-day", "max"),
                    prevent_initial_call=True)

clientside_callback(ClientsideFunction("solar", "togglePlay"),
                    Output("neo-ticker", "disabled"), Output("neo-play", "children"),
                    Input("neo-play", "n_clicks"), State("neo-ticker", "disabled"),
                    prevent_initial_call=True)
