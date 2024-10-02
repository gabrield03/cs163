import dash
from dash import html

dash.register_page(__name__, path='/')

layout = html.Div(
    style={
        'backgroundColor': '#D3D3D3',
        'minHeight': '100vh',
        'padding': '20px',
    },
    children=[
        html.Br(), html.Br(),
        html.H1('Effects of Weather on Energy Consumption in the Bay Area', style={'text-align': 'center'}),

        # Image and text styles
        html.Div(
            style={'maxWidth': '800px', 'margin': '0 auto'},
            children=[
                html.Div(
                    style={'textAlign': 'center', 'margin': '20px'},
                    children=[
                        html.Img(
                            src='/assets/bayareascenic.jpg',
                            style={
                                'width': '100%',
                                'maxWidth': '800px',
                                'height': 'auto',
                                'borderRadius': '10px',
                            }
                        )
                    ]
                )
            ]
        ),

        # Project Intro
        html.Div(
            style={'text-align': 'center', 'font-size': '20px', 'margin': '20px 60px'},
            children=[
                html.H2("Introduction"),
                html.P([
                    "This project explores the impact of climate change on energy consumption in the California Bay Area, with a focus on San Jose and San Francisco.",
                    html.Br(),
                    "By analyzing historical weather data and energy usage trends, I aim to identify key weather factors, such as extreme temperatures, that influence",
                    html.Br(),
                    "electricity demand. Ultimately, this project hopes to shed light on how shifts in climate can affect local energy consumption, with the potential",
                    html.Br(),
                    "to apply these findings to other regions.",
                    html.Br(), html.Br(),
                    "Explore the data visualizations below to see how weather patterns correlate with energy usage in each region!"
                ])
            ]
        ),

        html.Br(), html.Br(),
    ]
)