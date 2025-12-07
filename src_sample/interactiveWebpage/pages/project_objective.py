import dash
from dash import html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
from utils.data_pipeline import (
    format_columns, 
    create_table_header, create_table_summary_statistics, create_table_rows,
    load_joblib_from_github
)

import requests
from io import BytesIO
import os
import pickle
from joblib import dump, load

### Load Data ###
sj_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sj_combined.joblib"
sf_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sf_combined.joblib"

sj_df = load_joblib_from_github(sj_url)
sf_df = load_joblib_from_github(sf_url)


goals_and_sources_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Goals',
                        style = {
                            'text-align': 'center',
                            'font-size': '50px',
                            'font-variant': 'small-caps',
                        },
                    ),
                    width = 12,
                    className = 'mt-5 mb-2 text-center',
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'There are two major goals and one sub-goal for this project.',

                            html.Br(), html.Br(),
                            'First, I wanted to identify the weather features that affect ',
                            'energy consumption in San Jose and San Francisco. I was curious to ',
                            'to see if each region was disproportionately affected by certain weather ',
                            'variables. For instance, is San Jose severely impacted by high temperatures ',
                            'wind speed, precipitation, or some other factors or combination of factors?',

                            html.Br(), html.Br(),
                            'The second goal of this project was to explore how shifts in climate ',
                            '(e.g., increased frequency of extreme temperatures) affect energy usage. ',
                            'How do extreme weather events affect the energy usage of each region? ',

                            html.Br(), html.Br(),
                            'A sub-goal is to build a model that accurately predicts the ',
                            'energy usage of each region based on varying weather variables such as ',
                            'maximum temperature and minimum temperature.',
                        
                        ],
                        style = {
                            'text-align': 'left',
                        },
                    ),
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),
    ],
    className = 'mb-10',
)

data_table1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'San Jose Data',
                        style = {
                            'text-align': 'center',
                            'font-size': '40px',
                            'font-variant': 'small-caps',
                            #'text-shadow': '2px 2px 4px #000000',
                        },
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
)

data_table2 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'San Francisco Data',
                        style = {
                            'text-align': 'center',
                            'font-size': '40px',
                            'font-variant': 'small-caps',
                            #'text-shadow': '2px 2px 4px #000000'
                        },
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
    className = 'mb-10',
)

data_sources = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Data Sources',
                        style = {
                            'text-align': 'center',
                            'font-size': '40px',
                            'font-variant': 'small-caps',
                        },
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
                        html.P(
                            [
                                html.Span(
                                    'Electricity Data (PG&E)',
                                    style = {
                                        'font-size': '32px',
                                        'font-variant': 'small-caps',
                                    },
                                ),

                                # )
                                html.Br(), html.Br(),
                                'The PG&E data contains quarterly electricity consumption by ZIP code ',
                                'for northern and central California. Data consists of customer types ',
                                '(residential, commercial, etc.), average kilowatt-hours per customer ',
                                '(kWh) consumed, total kilowatt-hours (kWh) consumed, and the number of ',
                                'customers per ZIP code.',
                                html.Br(),
                                'Source: ',
                                dcc.Link(
                                    href = 'https://pge-energydatarequest.com/public_datasets',
                                    title = 'PG&E Link'
                                )
                            ],
                            style = {
                                'text-align': 'left',
                            },
                            className = 'mb-5',
                        ),
                        html.P(
                            [
                                html.Span(
                                    'Weather Data (NOAA)',
                                    style = {
                                        'font-size': '32px',
                                        'font-variant': 'small-caps',
                                    },
                                ),

                                # )
                                html.Br(), html.Br(),
                                'This National Oceanic and Atmospheric Administration (NOAA), consists of ',
                                'daily maximum and minimum temperatures, precipitation, and wind speed ',
                                'measurements from various Bay Area (San Jose and San Francisco) weather ',
                                'stations. The records are provided daily but were aggregated to monthly ',
                                'records to align with the eletricity data.',
                                html.Br(),
                                'Source: ',
                                dcc.Link(
                                    href = 'https://www.ncei.noaa.gov/cdo-web/search',
                                    title = 'NOAA Link'
                                )
                            ],
                            style = {
                                'text-align': 'left',
                            },
                        ),
                    ],
                    width = 12,
                    className = 'mb-2 text-center',
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

layout = dbc.Container(
    [
        html.Div(
            [
                goals_and_sources_section,
                data_table1,
                data_table2,
                data_sources,
            ],
            style = {
                'padding': '0px 100px',
                'backgroundColor': '#FAF9F6',
            },
        ),
    ],
    fluid = True,
)