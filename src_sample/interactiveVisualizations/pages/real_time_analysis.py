import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output

layout = html.Div([
    html.H1('Passage about real-time data analysis:'),

    html.Br(), html.Br(),

    html.H3('This page will gather real-time weather data and present energy consumption predictions.'),

    html.Br(), html.Br(),
])