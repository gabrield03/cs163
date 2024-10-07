import dash
from dash import html, dash_table, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd

from utils.data_preprocessing import load_and_preprocess_data, combine_regions

# Import data
sj_df, sf_df = load_and_preprocess_data()
combined_df = combine_regions(sj_df, sf_df)



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
                html.H3('SJ Data Table'),
            ],
            className = 'mb-2 text-center',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            sj_df.to_dict('records'),
                            [{"name": i, "id": i} for i in sj_df.columns],
                            page_size=PAGE_SIZE,
                            style_table={'overflowX': 'auto'},
                        ),
                    ],
                    width=10,
                ),
            ],
            style={'marginBottom': '30px'}, 
        ),
    ],
)

data_table2 = html.Div(
    [
        dbc.Row(
            [
                html.H3('SF Data Table'),
            ],
            className = 'mb-2 text-center',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            sf_df.to_dict('records'),
                            [{"name": i, "id": i} for i in sf_df.columns],
                            page_size=PAGE_SIZE,
                            style_table={'overflowX': 'auto'},
                        ),
                    ],
                    width=10,
                ),
            ],
            style={'marginBottom': '30px'}, 
        ),
    ],
)

data_table3 = html.Div(
    [
        dbc.Row(
            [
                html.H3('Combined Data Table on __something__'),
            ],
            className = 'mb-2 text-center',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dash_table.DataTable(
                            combined_df.to_dict('records'),
                            [{"name": i, "id": i} for i in combined_df.columns],
                            page_size=PAGE_SIZE,
                            style_table={'overflowX': 'auto'},
                        ),
                    ],
                    width=10,
                ),
            ],
            style={'marginBottom': '30px'}, 
        ),
    ],
)

layout = dbc.Container(
    [
        data_header,
        data_table1,
        data_table2,
        data_table3,
    ],
    fluid = True
)