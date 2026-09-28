"""
Earth-orbit page: orbit determination from observations or position vectors.

Flow: inputs -> `solve` callback (validates, runs core.solve, reports fit
errors) -> "earth-solution" store -> `render` callback (figures, readouts)
-> browser-side animation (assets/animate.js). All orbital mechanics lives
in core/; this module only parses inputs and formats results.
"""

import time
from datetime import datetime, timedelta, timezone

import numpy as np
from dash import (ClientsideFunction, Input, Output, State, callback, clientside_callback,
                  ctx, dash_table, dcc, html, no_update)

from app.earth_figures import (N_SAMPLES, figure_3d, figure_ground_track, globe_payload,
                               sample_track, track_payload)
from app.presets import DEFAULT_PRESET, PRESETS, STATIONS, TIME_FMT, station_name
from core.frames import station_ecef_to_geodetic
from core.solve import Observation, observations_to_eci, solve_orbit

AXES = ("x", "y", "z")
OBS_COLUMNS = ("time", "az", "el", "range")


# ----------------------------------------------------------------------
# Parsing / validation (pure functions, unit-tested)
# ----------------------------------------------------------------------
def parse_time(text):
    s = str(text or "").strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise ValueError("use YYYY-MM-DD HH:MM:SS (UTC)")


def parse_number(text):
    try:
        v = float(str(text).strip().replace(",", ""))
    except ValueError:
        raise ValueError("not a number")
    if not np.isfinite(v):
        raise ValueError("not a finite number")
    return v


def parse_observations(rows):
    """
    Table rows -> (observations, errors). errors is a list of
    (row_index, column_id, message); observations is None if any error.
    """
    obs, errors = [], []
    for i, row in enumerate(rows):
        vals = {}
        for col in OBS_COLUMNS:
            try:
                vals[col] = parse_time(row.get(col)) if col == "time" else parse_number(row.get(col))
            except ValueError as exc:
                errors.append((i, col, str(exc)))
        if "az" in vals and not 0 <= vals["az"] < 360:
            errors.append((i, "az", "azimuth must be in [0, 360)"))
        if "el" in vals and not -90 <= vals["el"] <= 90:
            errors.append((i, "el", "elevation must be in [-90, 90]"))
        if "range" in vals and vals["range"] <= 0:
            errors.append((i, "range", "range must be positive"))
        if len(vals) == 4:
            obs.append(Observation(vals["time"], vals["az"], vals["el"], vals["range"]))
    return (None if errors else obs), errors


def station_label(ecef):
    """'Madrid, Spain · 40.42°N 3.70°W' (or 'Custom station · ...')."""
    lat, lon, _ = station_ecef_to_geodetic(ecef)
    name = station_name(ecef) or "Custom station"
    return (f"{name} · {abs(lat):.2f}°{'N' if lat >= 0 else 'S'} "
            f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}")


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
def _seg(id_, options, value):
    return dcc.RadioItems(id=id_, options=options, value=value, className="seg", inline=True)


def _vec_row(label, prefix, values):
    return html.Div(className="field-row", children=[
        html.Span(label, className="field-label"),
        *[dcc.Input(id=f"{prefix}-{a}", value=v, debounce=False, className="inp", type="text")
          for a, v in zip(AXES, values)],
    ])


def _readout(label, id_, unit=""):
    return html.Div(className="readout", children=[
        html.Div(label, className="readout-label"),
        html.Div(className="readout-value", children=[html.Span("—", id=id_),
                                                        html.Span(unit, className="readout-unit")]),
    ])


def layout():
    preset = PRESETS[DEFAULT_PRESET]
    vec = PRESETS["vallado"]["vectors"]
    station = STATIONS[preset["station"]]

    inputs_card = html.Div(className="card", children=[
        html.Div("Scenario", className="card-title"),
        dcc.Dropdown(id="preset", options=[{"label": p["label"], "value": k} for k, p in PRESETS.items()],
                     value=DEFAULT_PRESET, clearable=False, searchable=False),
        html.Div("Input", className="card-title section"),
        _seg("mode", [{"label": "Observations", "value": "obs"},
                      {"label": "Position vectors", "value": "vec"}], preset["mode"]),
        html.Div(style={"height": "12px"}),

        # --- Observation mode ---
        html.Div(id="obs-panel", children=[
            html.Details(className="station", children=[
                html.Summary(id="station-summary", children=station_label(station)),
                html.Div(className="station-body", children=[
                    html.Div(className="field-row", children=[
                        html.Span("ECEF", className="field-label"),
                        *[dcc.Input(id=f"st-{a}", value=str(v), debounce=False, className="inp", type="text")
                          for a, v in zip(AXES, station)],
                    ]),
                    html.P("Earth-fixed position, km.", className="hint"),
                ]),
            ]),
            dash_table.DataTable(
                id="obs-table",
                columns=[{"id": "time", "name": "UTC time"}, {"id": "az", "name": "Az °"},
                         {"id": "el", "name": "El °"}, {"id": "range", "name": "Range km"},
                         {"id": "fit", "name": "Fit Δ km", "editable": False}],
                data=preset["rows"], editable=True, row_deletable=True,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "5px 6px", "border": "1px solid #e6eaef", "textAlign": "right",
                            "minWidth": "52px", "whiteSpace": "nowrap"},
                style_cell_conditional=[{"if": {"column_id": "time"}, "textAlign": "left", "minWidth": "140px"},
                                        {"if": {"column_id": "fit"}, "color": "#667383"}],
                style_header={"backgroundColor": "#eef3fa", "fontWeight": 700,
                              "borderBottom": "2px solid #c9d6ea"},
                css=[{"selector": ".dash-cell-value", "rule": "font-family: var(--mono);"}],
            ),
            html.Div(className="row between", style={"marginTop": "8px"}, children=[
                html.Button("+ Add observation", id="add-row", className="btn btn-outline"),
                html.Div(className="pair-select", children=[
                    "Solve", dcc.Dropdown(id="pair-a", clearable=False, searchable=False),
                    "→", dcc.Dropdown(id="pair-b", clearable=False, searchable=False),
                ]),
            ]),
            html.P("Tip: paste rows straight from a spreadsheet.", className="hint"),
        ]),

        # --- Vector mode ---
        html.Div(id="vec-panel", style={"display": "none"}, children=[
            _vec_row("r₁", "r1", vec["r1"]),
            _vec_row("r₂", "r2", vec["r2"]),
            html.P("ECI position vectors, km.", className="hint", style={"marginBottom": "10px"}),
            html.Div(className="field-row two", children=[
                html.Span("Time of flight", className="field-label"),
                dcc.Input(id="tof", value=vec["tof"], debounce=False, className="inp", type="text"),
            ]),
            html.Div(className="field-row two", children=[
                html.Span("Epoch (optional)", className="field-label"),
                dcc.Input(id="epoch", value=vec["epoch"], debounce=False, className="inp", type="text",
                          placeholder="YYYY-MM-DD HH:MM:SS"),
            ]),
            html.Div("Scale time of flight", className="field-label", style={"marginTop": "8px"}),
            dcc.Slider(id="tof-scale", min=0.5, max=2.0, step=0.01, value=1.0,
                       marks={0.5: "×0.5", 1.0: "×1", 1.5: "×1.5", 2.0: "×2"},
                       updatemode="drag", allow_direct_input=False),
            html.P(id="tof-effective", className="hint"),
        ]),

        html.Div("Direction of motion", className="card-title section"),
        _seg("direction", [{"label": "Prograde", "value": "pro"},
                           {"label": "Retrograde", "value": "retro"}], "pro"),
    ])

    status = html.Div(id="status", className="status", children=[html.Span(className="dot"), "Ready"])

    readouts_card = html.Div(className="card readouts-card", children=[
        html.Div(className="readouts", children=[
            _readout("Semi-major axis", "ro-a", "km"),
            _readout("Eccentricity", "ro-e"),
            _readout("Inclination", "ro-i"),
            _readout("RAAN", "ro-raan"),
            _readout("Arg. of perigee", "ro-argp"),
            _readout("True anomaly", "ro-nu"),
        ]),
        html.Div(id="subline", className="subline"),
        html.Div(id="banner"),
    ])

    view_card = html.Div(className="card", children=[
        html.Div(className="view-head", children=[
            _seg("view", [{"label": "3D view", "value": "3d"},
                          {"label": "Ground track", "value": "ground"}], "3d"),
            html.Div(id="view-options", className="row", style={"gap": "16px"}, children=[
                _seg("frame", [{"label": "Inertial", "value": "eci"},
                               {"label": "Earth-fixed", "value": "ecef"}], "eci"),
                dcc.Checklist(id="layers", className="layers", value=[], inline=True,
                              options=[{"label": "Velocity", "value": "velocity"},
                                       {"label": "Apsides & nodes", "value": "apsides"},
                                       {"label": "Station", "value": "station"}]),
            ]),
        ]),
        dcc.Graph(id="orbit-graph", className="graph", config={"displaylogo": False,
                                                                "modeBarButtonsToRemove": ["toImage"]}),
        html.Div(className="timebar", children=[
            html.Button("▶", id="play", className="btn btn-icon btn-primary", title="Play / pause"),
            html.Div(className="slider-wrap", children=[
                dcc.Slider(id="time-k", min=0, max=N_SAMPLES - 1, step=1, value=0, marks=None,
                           updatemode="drag", allow_direct_input=False),
            ]),
            html.Div(id="time-readout", className="time-readout"),
        ]),
        dcc.Interval(id="ticker", interval=50, disabled=True),
    ])

    details = html.Details(className="card more", children=[
        html.Summary("State vectors and notes"),
        html.Div(id="details"),
    ])

    return html.Div(className="page", children=[
        html.Div(className="col", children=[inputs_card, status]),
        html.Div(className="col", children=[readouts_card, view_card, details]),
        dcc.Store(id="earth-solution"),
        dcc.Store(id="earth-track"),
        dcc.Store(id="coast-data", data=globe_payload()),
    ])


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
@callback(
    Output("obs-table", "data", allow_duplicate=True),
    Output("mode", "value"),
    Output("r1-x", "value"), Output("r1-y", "value"), Output("r1-z", "value"),
    Output("r2-x", "value"), Output("r2-y", "value"), Output("r2-z", "value"),
    Output("tof", "value"), Output("epoch", "value"), Output("tof-scale", "value"),
    Output("pair-a", "value", allow_duplicate=True), Output("pair-b", "value", allow_duplicate=True),
    *[Output(f"st-{a}", "value") for a in AXES],
    Input("preset", "value"),
    prevent_initial_call=True,
)
def load_preset(key):
    p = PRESETS[key]
    if p["mode"] == "obs":
        n = len(p["rows"])
        station = [str(c) for c in STATIONS[p["station"]]]
        return (p["rows"], "obs", *[no_update] * 8, 1.0, 0, n - 1, *station)
    v = p["vectors"]
    return (no_update, "vec", *v["r1"], *v["r2"], v["tof"], v["epoch"], 1.0, no_update, no_update,
            *[no_update] * 3)


@callback(Output("obs-table", "data", allow_duplicate=True), Input("add-row", "n_clicks"),
          State("obs-table", "data"), prevent_initial_call=True)
def add_row(_n, rows):
    rows = list(rows or [])
    new = {c: "" for c in OBS_COLUMNS}
    try:
        new["time"] = (parse_time(rows[-1]["time"]) + timedelta(minutes=10)).strftime(TIME_FMT)
    except (IndexError, KeyError, ValueError):
        pass
    return rows + [new]


@callback(Output("obs-panel", "style"), Output("vec-panel", "style"), Input("mode", "value"))
def toggle_mode(mode):
    return ({}, {"display": "none"}) if mode == "obs" else ({"display": "none"}, {})


@callback(Output("view-options", "style"), Input("view", "value"))
def toggle_view_options(view):
    return {"gap": "16px"} if view == "3d" else {"display": "none"}


def _status(ok, text, detail=""):
    kids = [html.Span(className="dot"), html.Span(text)]
    if detail:
        kids.append(html.Span(detail, className="muted"))
    return kids, "status" if ok else "status error"


@callback(
    Output("earth-solution", "data"),
    Output("status", "children"), Output("status", "className"),
    Output("obs-table", "style_data_conditional"),
    Output("obs-table", "data"),
    Output("pair-a", "options"), Output("pair-b", "options"),
    Output("pair-a", "value"), Output("pair-b", "value"),
    Output("station-summary", "children"),
    Output("tof-effective", "children"),
    *[Output(f"{p}-{a}", "className") for p in ("st", "r1", "r2") for a in AXES],
    Output("tof", "className"), Output("epoch", "className"),
    Input("mode", "value"), Input("obs-table", "data"),
    *[Input(f"st-{a}", "value") for a in AXES],
    Input("pair-a", "value"), Input("pair-b", "value"), Input("direction", "value"),
    *[Input(f"{p}-{a}", "value") for p in ("r1", "r2") for a in AXES],
    Input("tof", "value"), Input("epoch", "value"), Input("tof-scale", "value"),
)
def solve(mode, rows, sx, sy, sz, pa, pb, direction, r1x, r1y, r1z, r2x, r2y, r2z,
          tof_text, epoch_text, tof_scale):
    t_start = time.perf_counter()
    rows = [dict(r) for r in (rows or [])]
    n = len(rows)
    options = [{"label": f"Obs {i + 1}", "value": i} for i in range(n)]
    if pa is None or pa >= n:
        pa = 0
    if pb is None or pb >= n or pb == pa:
        pb = n - 1 if n - 1 != pa else 0

    bad = set()      # ids of invalid dcc.Inputs
    # DataTable's default active-cell colour is pink, which reads as an error.
    cell_styles = [{"if": {"state": "active"}, "backgroundColor": "#e8f0fc",
                    "border": "1px solid #1d5fd1"},
                   {"if": {"state": "selected"}, "backgroundColor": "#f3f7fd",
                    "border": "1px solid #c9d8f2"}]
    station_text = no_update
    tof_eff_text = ""
    prograde = direction != "retro"

    def field_classes():
        ids = [f"{p}-{a}" for p in ("st", "r1", "r2") for a in AXES] + ["tof", "epoch"]
        return ["inp invalid" if i in bad else "inp" for i in ids]

    def fail(message):
        for r in rows:
            r["fit"] = ""
        return (None, *_status(False, message), cell_styles, rows, options, options, pa, pb,
                station_text, tof_eff_text, *field_classes())

    try:
        if mode == "obs":
            station = []
            for a, v in zip(AXES, (sx, sy, sz)):
                try:
                    station.append(parse_number(v))
                except ValueError:
                    bad.add(f"st-{a}")
            if bad:
                return fail("Ground station position must be three numbers (km).")
            _lat, _lon, alt = station_ecef_to_geodetic(station)
            station_text = station_label(station)
            if not -50 < alt < 10:
                return fail(f"Station is {alt:,.0f} km from the Earth's surface: check the ECEF values.")

            observations, errors = parse_observations(rows)
            for i, col, _msg in errors:
                cell_styles.append({"if": {"row_index": i, "column_id": col},
                                    "backgroundColor": "#fdeeee", "color": "#b42323"})
            if errors:
                i, col, msg = errors[0]
                name = {"time": "time", "az": "azimuth", "el": "elevation", "range": "range"}[col]
                return fail(f"Obs {i + 1} {name}: {msg}.")
            if n < 2:
                return fail("Add at least two observations.")

            positions = observations_to_eci(observations, station)
            first, second = sorted((pa, pb), key=lambda k: observations[k].time)
            t0 = observations[first].time
            tof = (observations[second].time - t0).total_seconds()
            if tof <= 0:
                return fail(f"Obs {pa + 1} and {pb + 1} have the same time.")
            sol = solve_orbit(positions[first], positions[second], tof, prograde=prograde, epoch=t0)
            obs_payload = []
            for i, (o, r) in enumerate(zip(observations, positions)):
                dt = (o.time - t0).total_seconds()
                r_pred, _ = sol.state_at(dt)
                rows[i]["fit"] = "—" if i in (pa, pb) else f"{np.linalg.norm(r_pred - r):,.2f}"
                obs_payload.append({"label": str(i + 1), "r_eci": r.tolist(), "t": dt,
                                    "is_pair": i in (pa, pb)})
            station_payload = station
        else:
            vals = {}
            for p in ("r1", "r2"):
                for a, v in zip(AXES, {"r1": (r1x, r1y, r1z), "r2": (r2x, r2y, r2z)}[p]):
                    try:
                        vals[f"{p}-{a}"] = parse_number(v)
                    except ValueError:
                        bad.add(f"{p}-{a}")
            try:
                tof0 = parse_number(tof_text)
                if tof0 <= 0:
                    raise ValueError
            except ValueError:
                bad.add("tof")
            epoch = None
            if str(epoch_text or "").strip():
                try:
                    epoch = parse_time(epoch_text)
                except ValueError:
                    bad.add("epoch")
            if bad:
                return fail("Check the highlighted fields: positions in km, time of flight in "
                            "seconds (> 0), epoch as YYYY-MM-DD HH:MM:SS.")
            r1 = np.array([vals[f"r1-{a}"] for a in AXES])
            r2 = np.array([vals[f"r2-{a}"] for a in AXES])
            tof = tof0 * float(tof_scale or 1.0)
            tof_eff_text = f"Effective time of flight: {tof:,.0f} s ({tof / 60:,.1f} min)"
            sol = solve_orbit(r1, r2, tof, prograde=prograde, epoch=epoch)
            obs_payload = [{"label": "r₁", "r_eci": r1.tolist(), "t": 0.0, "is_pair": True},
                           {"label": "r₂", "r_eci": r2.tolist(), "t": tof, "is_pair": True}]
            station_payload = None
    except (ValueError, RuntimeError) as exc:
        return fail(str(exc))

    kind = "elliptic" if sol.is_closed else "hyperbolic" if sol.elements["e"] > 1 else "parabolic"
    ms = (time.perf_counter() - t_start) * 1000
    solution = {
        "r1": sol.r1.tolist(), "r2": sol.r2.tolist(), "tof": sol.tof_s, "prograde": prograde,
        "epoch": sol.epoch.strftime(TIME_FMT) if sol.epoch else None,
        "observations": obs_payload, "station": station_payload,
    }
    return (solution, *_status(True, f"Solved · {kind} orbit", f"{ms:.0f} ms"),
            cell_styles, rows, options, options, pa, pb, station_text, tof_eff_text,
            *field_classes())


def _rebuild(solution):
    epoch = parse_time(solution["epoch"]) if solution["epoch"] else None
    sol = solve_orbit(solution["r1"], solution["r2"], solution["tof"],
                      prograde=solution["prograde"], epoch=epoch)
    observations = [dict(o, r_eci=np.array(o["r_eci"]), gmst_deg=float(sol.gmst_at(o["t"])))
                    for o in solution["observations"]]
    return sol, observations


def _fmt(v, digits, unit=""):
    return "undefined" if v is None else f"{v:,.{digits}f}{unit}"


def _hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}h {s % 3600 // 60:02d}m {s % 60:02d}s"


@callback(
    Output("orbit-graph", "figure"), Output("earth-track", "data"),
    Output("ro-a", "children"), Output("ro-e", "children"), Output("ro-i", "children"),
    Output("ro-raan", "children"), Output("ro-argp", "children"), Output("ro-nu", "children"),
    Output("subline", "children"), Output("banner", "children"), Output("details", "children"),
    Input("earth-solution", "data"), Input("view", "value"), Input("frame", "value"),
    Input("layers", "value"), State("time-k", "value"),
)
def render(solution, view, frame, layers, k):
    if not solution:
        return (no_update,) * 11
    sol, observations = _rebuild(solution)
    track = sample_track(sol)
    k = int(k or 0)
    if view == "ground":
        fig = figure_ground_track(track, observations, solution["station"], k=k)
    else:
        fig = figure_3d(sol, track, observations, frame=frame, layers=layers,
                        station_ecef=solution["station"], k=k)

    el = sol.elements
    sub = [html.Span(["Period", html.B(_hms(sol.period_s) if sol.is_closed else "open orbit")],
                     className="chip"),
           html.Span(["Perigee alt.", html.B(f"{sol.perigee_alt_km:,.0f} km")], className="chip"),
           html.Span(["Apogee alt.", html.B(f"{sol.apogee_alt_km:,.0f} km" if sol.is_closed else "—")],
                     className="chip"),
           html.Span(["Time of flight", html.B(_hms(sol.tof_s))], className="chip")]
    banner = [html.Div(w, className="banner") for w in sol.warnings]

    def vec(name, v, unit, d):
        return f"{name} = [{v[0]:>13,.{d}f} {v[1]:>13,.{d}f} {v[2]:>13,.{d}f} ] {unit}"
    details = [html.Div(className="vectors", children="\n".join([
        vec("r₁", sol.r1, "km", 3), vec("v₁", sol.v1, "km/s", 6),
        vec("r₂", sol.r2, "km", 3), vec("v₂", sol.v2, "km/s", 6),
        f"Epoch (r₁): {solution['epoch'] or 'not given (GMST taken as 0)'} UTC",
    ]))]
    if el["notes"]:
        details.append(html.Ul([html.Li(n) for n in el["notes"]], className="notes"))

    return (fig, track_payload(track),
            _fmt(el["a"], 1), f"{el['e']:.5f}", _fmt(el["i"], 3, "°"),
            _fmt(el["raan"], 3, "°"), _fmt(el["argp"], 3, "°"), _fmt(el["nu"], 3, "°"),
            sub, banner, details)


clientside_callback(ClientsideFunction("orbit", "tick"), Output("time-k", "value"),
                    Input("ticker", "n_intervals"), State("time-k", "value"),
                    State("earth-track", "data"), prevent_initial_call=True)

clientside_callback(ClientsideFunction("orbit", "togglePlay"),
                    Output("ticker", "disabled"), Output("play", "children"),
                    Input("play", "n_clicks"), State("ticker", "disabled"),
                    prevent_initial_call=True)

clientside_callback(ClientsideFunction("orbit", "render"), Output("time-readout", "children"),
                    Input("time-k", "value"), Input("earth-track", "data"),
                    State("view", "value"), State("frame", "value"), State("coast-data", "data"))
