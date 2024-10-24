import dash
from dash import html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
from utils.data_pipeline import (
    format_columns, 
    create_table_header, create_table_summary_statistics, create_table_rows
)

import os
import pickle
from joblib import dump, load

### Load Data ###
sj_df = pd.DataFrame()
sf_df = pd.DataFrame()

if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
    sj_df = load('joblib_files/base_data/sj_combined.joblib')

if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
    sf_df = load('joblib_files/base_data/sf_combined.joblib')

data_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        'Data',
                        className = 'mb-5 text-center',
                    ),
                    width = 12,
                    style = {'height': '100%'}
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

data_table1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H3('San Jose Data'),
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            dbc.Table(
                                [
                                    html.Thead(
                                        create_table_header(sj_df),
                                    ),
                                    html.Tbody(
                                        [
                                            create_table_summary_statistics(sj_df),
                                            *create_table_rows(sj_df),
                                        ],
                                    ),
                                ],
                                
                                bordered = True,
                                striped = True,
                                hover = True,
                                responsive = True,
                                className = 'table-light',
                            ),
                            style = {'maxHeight': '500px', 'overflow': 'auto'},
                        ),
                    ],
                    width=12,
                ),
            ],
            style = {'marginBottom': '30px'},
        ),
    ],
    className = 'mb-5',
)

data_table2 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H3('San Francisco Data'),
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            dbc.Table(
                                [
                                    html.Thead(
                                        create_table_header(sf_df),
                                    ),
                                    html.Tbody(
                                        [
                                            create_table_summary_statistics(sf_df),
                                            *create_table_rows(sf_df),
                                        ],
                                    ),
                                ],
                                
                                bordered = True,
                                striped = True,
                                hover = True,
                                responsive = True,
                                className = 'table-light',
                            ),
                            style = {'maxHeight': '500px', 'overflow': 'auto'},
                        ),
                    ],
                    width=12,
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