import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

layout = dbc.Container([
    dbc.Row([
        html.H1('Passage about the data analysis conducted:'),
        html.Br(), html.Br(),
    ]),
    dbc.Row([
        html.H3('This page will explain, describe, and illustrate the data analysis conducted for this project.'),
        html.Br(), html.Br(),
    ]),
])