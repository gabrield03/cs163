import dash
from dash import html

dash.register_page(__name__, path='/')

layout = html.Div([
    html.Br(), html.Br(),

    html.H1('Effects of Weather on Energy Consumption in the Bay Area', style = {'text-align': 'center'}),

    # Intro about the project
    html.Div([
        html.H2("Introduction"),
        html.P("This project explores the impact of climate change on energy consumption in the California Bay Area, with a focus on San Jose and San Francisco."),
        html.P("By analyzing historical weather data and energy usage trends, I aim to identify key weather factors, such as extreme temperatures, that influence electricity demand. Ultimately, this project hopes to shed light on how shifts in climate can affect local energy consumption, with the potential to apply these findings to other regions."),
        html.P("Explore the data visualizations below to see how weather patterns correlate with energy usage in each region!")
    ], style={'font-size': '20px', 'margin': '20px'}),

    html.Br(), html.Br(),
])