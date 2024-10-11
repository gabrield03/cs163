import dash
from dash import html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
from utils.data_pipeline import format_columns

import os
import pickle

### Load Data ###
sj_df = ''
sf_df = ''
if os.path.exists('sj_combined.pkl'):
    with open('sj_combined.pkl', 'rb') as f:
        sj_df = pickle.load(f)

if os.path.exists('sf_combined.pkl'):
    with open('sf_combined.pkl', 'rb') as f:
        sf_df = pickle.load(f)

PAGE_SIZE = 10

data_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        'Passage about the data used:',
                    ),
                    className = 'text-center mb-5',
                    width = 12,
                    style = {'height': '100%'}
                ),
            ],
        ),
    ]
)

data_table1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        'San Jose Data'
                    ),
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            sj_df.to_dict('records'),
                            columns = format_columns(sj_df),
                            page_size = PAGE_SIZE,
                            style_table = {'overflowX': 'auto'},
                            style_data = {'backgroundColor': '#ecf0f1'},
                            style_header = {
                                'backgroundColor': '#bdc3c7',
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                            },
                        ),
                    ],
                    width = 12,
                ),
            ],
            style = {'marginBottom': '30px'}, 
        ),
    ],
)

data_table2 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        'San Francisco Data'
                    ),
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            sf_df.to_dict('records'),
                            columns = format_columns(sf_df),
                            page_size = PAGE_SIZE,
                            style_table = {'overflowX': 'auto'},
                            style_data = {'backgroundColor': '#ecf0f1'},
                            style_header = {
                                'backgroundColor': '#bdc3c7',
                                'fontWeight': 'bold',
                                'textAlign': 'center',
                            },
                        ),
                    ],
                    width = 12,
                ),
            ],
            style = {'marginBottom': '30px'}, 
        ),
    ],
)

layout = dbc.Container(
    [
        data_header,
        data_table1,
        data_table2,
    ],
    fluid = True,
)