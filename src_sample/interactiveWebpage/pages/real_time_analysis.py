import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import os
from joblib import dump, load

# sj_df = pd.DataFrame()
# df_df = pd.DataFrame()

# if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
#     sj_df = load('joblib_files/base_data/sj_combined.joblib')

# if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
#     sf_df = load('joblib_files/base_data/sf_combined.joblib')

layout = dbc.Container([
    dbc.Row([
        html.H1('Passage about real-time data analysis:',
                className = 'mb-5'),
    ]),
    dbc.Row([
        html.H3('This page will gather real-time weather data and present energy consumption predictions.'),
    ]),
])