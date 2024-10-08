import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import utils.data_pipeline
import os
import pickle

layout = dbc.Container([
    dbc.Row([
        html.H1('Passage about the data analysis conducted:',
                className = 'mb-5'),
    ]),
    dbc.Row([
        html.H3('This page will explain, describe, and illustrate the data analysis conducted for this project.'),
    ]),
])