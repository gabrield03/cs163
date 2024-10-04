import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html
import importlib

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Sidebar layout
sidebar_header = dbc.Row(
    [
        dbc.Col(html.H2("Navigation", className="display-4")),
        dbc.Col(
            [
                html.Button(
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    style={"color": "rgba(0,0,0,.5)", "border-color": "rgba(0,0,0,.1)"},
                    id="navbar-toggle",
                ),
                html.Button(
                    html.Span(className="navbar-toggler-icon"),
                    className="navbar-toggler",
                    style={"color": "rgba(0,0,0,.5)", "border-color": "rgba(0,0,0,.1)"},
                    id="sidebar-toggle",
                ),
            ],
            width="auto",
            align="center",
        ),
    ]
)

sidebar = html.Div(
    [
        sidebar_header,
        html.Div(
            [
                html.Hr(),
                html.P("Explore the project's pages!", className="lead"),
            ],
            id="blurb",
        ),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("Visualization", href="/visualizations", active="exact"),
                    dbc.NavLink("Analytics", href="/analytics", active="exact"),
                    dbc.NavLink("Real-Time Analysis", href="/real_time_analysis", active="exact"),
                    dbc.NavLink("Data", href="/data", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
            id="collapse",
        ),
    ],
    id="sidebar",
)

content = html.Div(id="page-content")

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Callback to render page content dynamically
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    # Dynamically import the correct page module based on URL
    if pathname == "/":
        page_module = importlib.import_module('pages.home')
    elif pathname == "/visualizations":
        page_module = importlib.import_module('pages.visualizations')
    elif pathname == "/analytics":
        page_module = importlib.import_module('pages.analytics')
    elif pathname == "/real_time_analysis":
        page_module = importlib.import_module('pages.real_time_analysis')
    elif pathname == "/data":
            page_module = importlib.import_module('pages.data')
    else:
        return html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ],
            className="p-3 bg-light rounded-3",
        )
    
    # Assuming each page module defines a 'layout' variable
    return page_module.layout

# Sidebar toggling logic
@app.callback(
    Output("sidebar", "className"),
    [Input("sidebar-toggle", "n_clicks")],
    [State("sidebar", "className")],
)
def toggle_classname(n, classname):
    if n and classname == "":
        return "collapsed"
    return ""

@app.callback(
    Output("collapse", "is_open"),
    [Input("navbar-toggle", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run_server(debug=True)