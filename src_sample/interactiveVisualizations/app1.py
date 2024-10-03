import dash
#import dash_bootstrap_components as dbc
#from dash_bootstrap_templates import 
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)
# app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Horizontal Navigation Bar
app.layout = html.Div([
    html.Div(
        className = 'nav-bar',
        children = [
            dcc.Link(
                f"{page['name'].title()}",
                href = page["relative_path"],
                style = {
                    'margin-right': '17px',
                    'textDecoration': 'none',
                    'color': 'blue',
                    'fontSize': '15px',
                    'padding': '0px 5px'
                },
            ) for page in dash.page_registry.values()
        ],
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
    ),
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)