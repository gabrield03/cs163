import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from pages import home, visualizations, analytics, real_time_analysis, data

#### Set the App Properties ####
# debating between LUX, MORPH, SLATE, SOLAR
app = dash.Dash(
    __name__,
    external_stylesheets = [dbc.themes.LUX],
    meta_tags = [
        {
            'name': 'viewport',
            'content': 'width=device-width, initial-scale=1'
        }
    ],
    suppress_callback_exceptions = True
)

#### Define the Container Partitions ####
# Sidebar header
sidebar_header = dbc.Row(
    [
        dbc.Col(
            html.H2(
                # 'Navigation',
                'Explore',
                style = {'color': '#ecf0f1'},
            )
        ),

        dbc.Col(
            [
                html.Button(
                    html.Span(className = 'navbar-toggler-icon'),
                    className = 'navbar-toggler',
                    style = {
                        'color': 'rgba(0,0,0,.5)',
                        'border-color': 'rgba(0,0,0,.1)',
                    },
                    id = 'navbar-toggle',
                ),
                
                html.Button(
                    html.Span(className = 'navbar-toggler-icon'),
                    className = 'navbar-toggler',
                    style = {
                        'color': 'rgba(0,0,0,.5)',
                        'border-color': 'rgba(0,0,0,.1)',
                    },
                    id = 'sidebar-toggle',
                ),
            ],
            width = 'auto',
            align = 'center',
        ),
    ],
)

# Side bar
sidebar = html.Div(
    [
        sidebar_header,
        # html.Div(
        #     [
        #         html.Hr(),
        #         html.P('Explore the project\'s pages!', className = 'lead'),
        #     ],
        #     id = 'blurb',
        # ),
        dbc.Collapse(
            dbc.Nav(
                [
                    dbc.NavLink(
                        'Home',
                        href = '/',
                        active = 'exact',
                        style = {'color': '#ecf0f1'},
                    ),

                    dbc.NavLink(
                        'Visualizations',
                        href = '/visualizations',
                        active = 'exact',
                        style = {'color': '#ecf0f1'},
                    ),

                    dbc.NavLink(
                        'Analytics',
                        href = '/analytics',
                        active = 'exact',
                        style = {'color': '#ecf0f1'},
                    ),

                    dbc.NavLink(
                        'Real-Time Analysis',
                        href = '/real_time_analysis',
                        active = 'exact',
                        style = {'color': '#ecf0f1'},
                    ),

                    dbc.NavLink(
                        'Data',
                        href = '/data',
                        active = 'exact',
                        style = {'color': '#ecf0f1'},
                    ),
                ],
                vertical = True,
                pills = True,
            ),
            id = 'collapse',
        ),
    ],
    id = 'sidebar',
)

content = html.Div(id = 'page-content')

# Add the contents to the app layout
app.layout = html.Div(
    [
        dcc.Location(id = 'url'),
        sidebar,
        content,
    ],
)

#### Plots ####
# Callback to render page content dynamically
@app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
)
# Function to dynamically render the page
def render_page_content(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/visualizations':
        return visualizations.layout
    elif pathname == '/analytics':
        return analytics.layout
    elif pathname == '/real_time_analysis':
        return real_time_analysis.layout
    elif pathname == '/data':
        return data.layout
    else:
        return html.Div(
            [
                html.H1('404: Not found', className = 'text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognized...'),
            ],
            className = 'p-3 bg-light rounded-3',
        )

# Callback for sidebar toggling
@app.callback(
    Output('sidebar', 'className'),
    Input('sidebar-toggle', 'n_clicks'),
    State('sidebar', 'className'),
)
# Function for sidebar toggling
def toggle_classname(n, classname):
    if n and classname == '':
        return 'collapsed'
    return ''

# Callback for collapsing sidebar
@app.callback(
    Output('collapse', 'is_open'),
    Input('navbar-toggle', 'n_clicks'),
    State('collapse', 'is_open'),
)
# Function for collapsing sidebar
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run(debug=True)