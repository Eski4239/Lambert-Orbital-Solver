"""Dash application shell: header, scope tabs, page modules."""

from dash import Dash, Input, Output, dcc, html

from app import earth_page

FONTS = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
         "&family=IBM+Plex+Sans:wght@400;500;600&display=swap")

BRAND_MARK = html.Img(className="brand-mark", src=(
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<circle cx='12' cy='12' r='4.2' fill='%231d5fd1'/>"
    "<ellipse cx='12' cy='12' rx='10.5' ry='5' fill='none' stroke='%2317202b' stroke-width='1.3' "
    "transform='rotate(-25 12 12)'/><circle cx='20.6' cy='8.3' r='1.6' fill='%23e8590c'/></svg>"))


def create_app():
    app = Dash(__name__, title="Orbit Lab", external_stylesheets=[FONTS],
               suppress_callback_exceptions=True)
    app.layout = html.Div([
        html.Div(className="header", children=[
            html.Div(className="brand", children=[
                BRAND_MARK,
                html.Span("ORBIT LAB", className="brand-name"),
                html.Span("Lambert orbit determination", className="brand-sub"),
            ]),
            dcc.Tabs(id="scope", value="earth", className="nav-tabs", parent_className="nav-tabs-parent",
                     children=[
                         dcc.Tab(label="Earth orbit", value="earth", className="nav-tab",
                                 selected_className="nav-tab--selected"),
                         dcc.Tab(label="Near-Earth objects", value="neo", className="nav-tab",
                                 selected_className="nav-tab--selected"),
                     ]),
        ]),
        html.Div(id="scope-earth", children=earth_page.layout()),
        html.Div(id="scope-neo", style={"display": "none"}, children=html.Div(
            "Near-Earth objects: coming in Phase 3.", className="placeholder")),
    ])

    @app.callback(Output("scope-earth", "style"), Output("scope-neo", "style"),
                  Input("scope", "value"))
    def switch_scope(scope):
        return ({}, {"display": "none"}) if scope == "earth" else ({"display": "none"}, {})

    return app

