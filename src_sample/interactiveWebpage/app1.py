import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from pages import (
    data_exploration,
    home,
    project_objective,
    analytical_methods, 
    real_time_analysis
)

# Set the App Properties
app = dash.Dash(
    __name__,
    external_stylesheets = [dbc.themes.LUX, '/assets/styles.css'],
    meta_tags = [
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
            'Home', href = '/',
                className = 'ml-auto',
                style = {
                    'color': '#ffffff',
                    'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                }
            ),
            
            # Links on the right
            dbc.Nav(
                [
                    dbc.NavLink(
                        'Project Objective',
                        href = '/project-objective',
                        active = 'exact',
                        style = {
                            'color': '#ffffff',
                            'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                        }
                    ),
                    dbc.NavLink(
                        'Analytical Methods',
                        href = '/analytical-methods',
                        active = 'exact',
                        style = {
                            'color': '#ffffff',
                            'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                        }
                    ),
                    dbc.NavLink(
                        'Data Exploration',
                        href = '/data-exploration',
                        active = 'exact',
                        style = {
                            'color': '#ffffff',
                            'textShadow': '-1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000',
                        }
                    ),
                ],
                navbar = True,
                className = 'ml-auto',
            ),
        ],
        fluid = True,
    ),
    color = 'transparent',
    dark = True,
    className = 'mb-4 fixed-top',
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
    elif pathname == '/data-exploration':
        return data_exploration.layout
    elif pathname == '/analytical-methods':
        return analytical_methods.layout
    elif pathname == '/real-time-analysis':
        return real_time_analysis.layout
    elif pathname == '/project-objective':
        return project_objective.layout
    else:
        return html.Div(
            [
                html.H1('404: Not found', className = 'text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognized...'),
            ],
            className='p-3 bg-light rounded-3',
        )

if __name__ == '__main__':
    app.run(debug=True)