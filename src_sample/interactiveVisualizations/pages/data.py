import dash
from dash import html, dcc, callback, Input, Output

dash.register_page(__name__)

layout = html.Div([
    html.H1('This is our Data page'),
    html.Div([
        "Select a city: ",
        dcc.RadioItems(
            options=['New York City', 'Montreal', 'San Francisco'],
            value='Montreal',
            id='data-input'
        )
    ]),
    html.Br(),
    html.Div(id='data-output'),
])


@callback(
    Output('data-output', 'children'),
    Input('data-input', 'value')
)
def update_city_selected(input_value):
    return f'You selected: {input_value}'