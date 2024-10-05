import dash
from dash import html, dash_table, dcc, callback, Input, Output

import pandas as pd

# Import data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

# Combine sj and sf
# Combine sj and sf
sj_df['region'] = 'San Jose'
sf_df['region'] = 'San Francisco'

# Reshape sj_df to include temperature and energy columns
sj_melted_temp = pd.melt(
    sj_df,
    id_vars = [
        'zipcode', 'month', 'year', 'customerclass', 
        'combined', 'totalcustomers', 'totalkwh', 
        'averagekwh', 'year-month', 
        'awnd', 'prcp', 'wdf2', 'wdf5', 'wsf2', 'wsf5', 'region'
    ],
    value_vars = ['tmax', 'tmin'], 
    var_name = 'temp_type', 
    value_name = 'temp'
)

sj_melted_energy = pd.melt(
    sj_melted_temp,
    id_vars = [
        'zipcode', 'month', 'year', 'customerclass', 
        'combined', 'totalcustomers', 'year-month', 
        'awnd', 'prcp', 'wdf2', 'wdf5', 
        'wsf2', 'wsf5', 'region', 'temp_type', 'temp'
    ],
    value_vars = ['totalkwh', 'averagekwh'], 
    var_name = 'energy_type', 
    value_name = 'energy'
)

# Reshape sf_df to include temperature and energy columns
sf_melted_temp = pd.melt(
    sf_df,
    id_vars = [
        'zipcode', 'month', 'year', 'customerclass', 
        'combined', 'totalcustomers', 'totalkwh', 
        'averagekwh', 'year-month', 'prcp', 'region'
    ],
    value_vars = ['tmax', 'tmin'], 
    var_name = 'temp_type', 
    value_name = 'temp'
)

sf_melted_energy = pd.melt(
    sf_melted_temp,
    id_vars = [
        'zipcode', 'month', 'year', 'customerclass', 
        'combined', 'totalcustomers', 'year-month', 
        'prcp', 'region', 'temp_type', 'temp'
    ],
    value_vars = ['totalkwh', 'averagekwh'], 
    var_name = 'energy_type', 
    value_name = 'energy'
)

# Combine both dataframes
combined_df = pd.concat([sj_melted_energy, sf_melted_energy], ignore_index = True)



PAGE_SIZE = 10

layout = html.Div([
    html.H1('Passage about the data:'),

    html.Br(), html.Br(),

    dash_table.DataTable(
        sj_df.to_dict('records'),
        [{"name": i, "id": i} for i in sj_df.columns],
        page_size = PAGE_SIZE
    ),

    html.Br(), html.Br(),

    dash_table.DataTable(
        sf_df.to_dict('records'),
        [{"name": i, "id": i} for i in sf_df.columns],
        page_size = PAGE_SIZE
    ),

    html.Br(), html.Br(),

    dash_table.DataTable(
        combined_df.to_dict('records'),
        [{"name": i, "id": i} for i in combined_df.columns],
        page_size = PAGE_SIZE
    ),

    html.Br(), html.Br(),
])