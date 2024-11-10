import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from pages import (
    home, home_old,
    project_objective, analytical_methods, 
    visualizations, real_time_analysis
)

# Set the App Properties
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[
        {'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}
    ],
    suppress_callback_exceptions=True
)

# Define the top navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            # Home link on the left
            dbc.NavbarBrand(
                "Home", href="/",
                className="ml-auto",
                style = {
                    'color': '#6d6b6b'
                }
            ),
            
            # Links on the right
            dbc.Nav(
                [
                    dbc.NavLink(
                        "Project Objective",
                        href="/project-objective",
                        active="exact",
                        style = {
                            'color': '#6d6b6b'
                        }
                    ),
                    dbc.NavLink(
                        "Analytical Methods",
                        href="/analytical-methods",
                        active="exact",
                        style = {
                            'color': '#6d6b6b'
                        }
                    ),
                    dbc.NavLink(
                        "Visualizations",
                        href="/visualizations",
                        active="exact",
                        style = {
                            'color': '#6d6b6b'
                        }
                    ),
                    # dbc.NavLink(
                    #     "Real-Time Analysis",
                    #     href="/real-time-analysis",
                    #     active="exact",
                    #     disabled = True,
                    #     style = {
                    #         'color': '#6d6b6b'
                    #     }
                    # ),
                    # dbc.NavLink(
                        # "Home (old)",
                        # href="/home",
                        # active="exact",
                        # style = {
                        # 'color': '#6d6b6b'
                        # }
                    # ),
                ],
                navbar=True,
                className="ml-auto",
            ),
        ],
        fluid=True,
    ),
    color="transparent",
    dark=True,
    className="mb-4 fixed-top",
)

# Main page layout
app.layout = html.Div(
    [
        dcc.Location(
            id = 'url'
        ),
        navbar,
        html.Div(
            id = 'page-content',
            # className = 'container mt-4',
            className = 'mt-4',
        ),
    ],
)

# Callback to render page content dynamically
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def render_page_content(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/visualizations':
        return visualizations.layout
    elif pathname == '/analytical-methods':
        return analytical_methods.layout
    elif pathname == '/real-time-analysis':
        return real_time_analysis.layout
    elif pathname == '/project-objective':
        return project_objective.layout
    # elif pathname == '/home':
    #     return home_old.layout
    else:
        return html.Div(
            [
                html.H1('404: Not found', className='text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognized...'),
            ],
            className='p-3 bg-light rounded-3',
        )

if __name__ == '__main__':
    app.run(debug=True)