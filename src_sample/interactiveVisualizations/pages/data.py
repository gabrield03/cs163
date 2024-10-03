import dash
from dash import html, dash_table, dcc, callback, Input, Output

import pandas as pd

# Import data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

dash.register_page(__name__)

layout = html.Div([
    html.H1('Passage about the data.'),

    html.Br(), html.Br(),

    dash_table.DataTable(sj_df.to_dict('records'), [{"name": i, "id": i} for i in sj_df.columns]),

    html.Br(), html.Br(),

        dash_table.DataTable(sf_df.to_dict('records'), [{"name": i, "id": i} for i in sf_df.columns])

])