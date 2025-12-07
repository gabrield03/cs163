import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from dash import dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import requests
from io import BytesIO
import os
from joblib import load

from utils.data_pipeline import load_joblib_from_github


### Load Data ###
sj_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sj_combined.joblib"
sf_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sf_combined.joblib"

sj_df = load_joblib_from_github(sj_url)
sf_df = load_joblib_from_github(sf_url)    

# Visualizations page header
visualizations_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Exploring the Data',
                    ),
                    className = 'text-center mb-5 mt-5',
                    width = 12,
                    style = {
                        'font-size': '50px',
                        'height': '100%',
                        'font-variant': 'small-caps',
                    },
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# Histogram plots - SJ
data_distribution_section = html.Div(
    [
        dbc.Row(
            [
                html.P(
                    'Regional Data Distributions',
                    style = {
                        'text-align': 'center',
                        'font-size': '40px',
                        'font-variant': 'small-caps',
                    },
                ),
            ],
            className = 'mb-2 text-center',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id = 'select_energy_option',
                            options = [
                                {'label': 'SJ Avg Energy', 'value': 'sj'},
                                {'label': 'SF Avg Energy', 'value': 'sf'},
                            ],
                            multi = False,
                            value = 'sj',
                            style = {
                                'backgroundColor': '#bdc3c7',
                                'color': '#2c3e50'
                            }, 
                        ),
                    ],
                    width = 3,
                    className = 'mb-2'
                ),
                dbc.Col([], width = 3),
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id = 'select_weather_option',
                            options = [
                                {'label': 'SJ Weather', 'value': 'sj'},
                                {'label': 'SF Weather', 'value': 'sf'},
                            ],
                            multi = False,
                            value = 'sj',
                            style = {
                                'backgroundColor': '#bdc3c7',
                                'color': '#2c3e50'
                            }, 
                        ),
                    ],
                    width = 3,
                    className = 'mb-2'
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'energy_data',
                            figure = {},
                        ),
                    ],
                    width = 6,
                ),
                #dbc.Col([], width = 2),
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'weather_data',
                            figure = {},
                        ),
                    ],
                    width = 6,
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col([], width = 1),
                dbc.Col(
                    [
                        html.Div(
                            id = 'energy_output_container',
                        ),
                    ],
                    width = 4,
                ),
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        html.Div(
                            id = 'weather_output_container',
                        ),
                    ],
                    width = 4,
                ),
            ],
            style = {
                'color': '#0f0f0f',
                'font-size': '12px',
            }
        ),
    ],
    className = 'mb-10',
)

# Heatmap plots - Both regions
comb_heatmaps = html.Div(
    [
        dbc.Row(
            [
                html.P(
                    'Heatmaps for San Jose and San Francisco Energy Usage and Weather History',
                    style = {
                        'text-align': 'center',
                        'font-size': '40px',
                        'font-variant': 'small-caps',
                    },
                ),
            ],
            className = 'mb-2 text-center',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            id = 'heatmap-tabs',
                            value = 'sj',
                            children = [
                                dcc.Tab(
                                    label = 'SJ Average (kWh) and Max Temp',
                                    value = 'sj',
                                    style = {
                                        'backgroundColor': '#bdc3c7',
                                        'color': '#2c3e50'
                                    }, 
                                    selected_style={
                                        'backgroundColor': '#1abc9c',
                                        'color': '#2c3e50'
                                    },
                                ),

                                dcc.Tab(
                                    label = 'SF Average (kWh) and Max Temp',
                                    value = 'sf',
                                    style = {
                                        'backgroundColor': '#bdc3c7',
                                        'color': '#2c3e50'
                                    }, 
                                    selected_style = {
                                        'backgroundColor': '#1abc9c',
                                        'color': '#2c3e50'
                                    },
                                ),
                            ],
                        ),
                    ],
                    className = 'mb-3',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'heatmap-energy',
                        ),
                    ],
                    width = 6,
                ),

                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'heatmap-temp',
                        ),
                    ],
                    width = 6,
                ), 
            ],
        ),
    ],
    className = 'mb-5',
)

# Add the contents to the app layout
layout = dbc.Container(
    [
        html.Div(
            [
                visualizations_header,
                data_distribution_section,
                comb_heatmaps,
            ],
            style = {
                'padding': '0px 100px',
                'backgroundColor': '#FAF9F6',
            },
        ),
    ],
    fluid = True
)

#### Plots ####
# Callback for energy usage histograms
@callback(
    [
        Output(component_id = 'energy_output_container', component_property = 'children'),
        Output(component_id = 'energy_data', component_property = 'figure')
    ],
    Input(component_id = 'select_energy_option', component_property = 'value')
)
# Function for SJ Histograms
def update_energy_usage(option_selected):
    container = None
    plot_title = None
    df = sj_df if option_selected == 'sj' else sf_df

    if option_selected == 'sj':
        plot_title = 'Avg Monthly Energy Usage - SJ'
        column = 'averagekwh'
        color = '#6600CC'
        kde_color = '#000000'
        x_range = [280, 575]

        container = html.P('Distribution of average monthly energy usage in San Jose (kWh)')
    
    else:
        plot_title = 'Avg Monthly Energy Usage - SF'
        column = 'averagekwh'
        color = '#CC00CC'
        kde_color = '#000000'
        x_range = [215, 410]

        container = html.P('Distribution of average monthly energy usage in San Francisco (kWh)')
    
    # Create histogram - using try except because of strange non-rendering error
    try:
        fig = px.histogram(
            df,
            x = column,
            nbins = 40,
            title = plot_title,
            color_discrete_sequence = [color],
        )
    except:
        fig = px.histogram(
            df,
            x = column,
            nbins = 40,
            title = plot_title,
            color_discrete_sequence = [color],
        )

    # Update the figure to set white outlines for the bars
    fig.update_traces(
        marker_line_color = 'white',
        marker_line_width = 1.5
    )

    # Calculate KDE
    kde = gaussian_kde(df[column].dropna())
    kde_range = np.linspace(df[column].min(), df[column].max(), 100)

    # Add KDE line
    fig.add_trace(
        go.Scatter(
            x = kde_range,
            y = kde(kde_range) * len(df[column]) * (df[column].max() - df[column].min()) / 40,
            mode = 'lines',
            name = f'{column} KDE',
            line = dict(
                color = kde_color,
                width = 2,
                dash = 'dash'
            )
        )
    )

    # Adjust layout
    fig.update_layout(
        title = plot_title,
        barmode = 'overlay',
        yaxis_title = 'Frequency',
        xaxis_title = 'Avg Energy Usage (kWh)',
        xaxis_range = x_range,
        legend = dict(
            x = 1,
            y = 1,
            xanchor = 'right',
            yanchor = 'top',
            bgcolor = 'rgba(255, 255, 255, 0.7)',
            bordercolor = 'black',
            borderwidth = 1,
        ),
        paper_bgcolor = '#ecf0f1',
        margin = dict(
            l = 0,
            r = 0,
        ),
    )

    return container, fig

# Callback for weather histograms
@callback(
    [
        Output(component_id = 'weather_output_container', component_property = 'children'),
        Output(component_id = 'weather_data', component_property = 'figure')
    ],
    Input(component_id = 'select_weather_option', component_property = 'value')
)
# Function for SJ Histograms
def update_weather(option_selected):
    df = sj_df if option_selected == 'sj' else sf_df

    container = None
    plot_title = None
    tmax_color = None
    tmin_color = None
    x_range = None

    if option_selected == 'sj':
        plot_title = 'Monthly Avg Max and Min Temp - SJ'
        container = html.P('Distribution of average monthly maximum and minimum temperatures in San Jose (°F).')
        tmax_color = '#8B0000'
        tmin_color = '#003FFF'
        x_range = [30, 90]
    
    else:
        plot_title = 'Monthly Avg Max and Min Temp - SF'
        container = html.P('Distribution of average monthly maximum and minimum temperatures in San Francisco (°F).')
        tmax_color = '#8B0000'
        tmin_color = '#003FFF'
        x_range = [39, 80]

    fig = go.Figure()

    # tmax histogram
    fig.add_trace(
        go.Histogram(
            x = df['tmax'],
            nbinsx = 40,
            name = 'Max Temp (tmax)',
            marker_color = tmax_color,
            opacity = 0.75
        )
    )

    # tmin histogram
    fig.add_trace(
        go.Histogram(
            x = df['tmin'],
            nbinsx = 40,
            name = 'Min Temp (tmin)',
            marker_color = tmin_color,
            opacity = 0.75
        )
    )

    # Calculate tmax KDE
    tmax_kde = gaussian_kde(df['tmax'].dropna())
    tmax_range = np.linspace(df['tmax'].min(), df['tmax'].max(), 100)

    # Add tmax KDE line
    fig.add_trace(
        go.Scatter(
            x = tmax_range,
            y = tmax_kde(tmax_range) * len(df['tmax']) * (df['tmax'].max() - df['tmax'].min()) / 40,
            mode = 'lines',
            name = 'Max Temp KDE',
            line = dict(
                color = '#FFCCCB',
                width = 2,
                dash = 'dash'
            )
        )
    )

    # Calculate tmin KDE
    tmin_kde = gaussian_kde(df['tmin'].dropna())
    tmin_range = np.linspace(df['tmin'].min(), df['tmin'].max(), 100)

    # Add tmin KDE line
    fig.add_trace(
        go.Scatter(
            x = tmin_range,
            y = tmin_kde(tmin_range) * len(df['tmin']) * (df['tmin'].max() - df['tmin'].min()) / 40,
            mode = 'lines',
            name = 'Min Temp KDE',
            line = dict(color = '#A5E5FF', width = 2, dash = 'dash')
        )
    )

    # Update layout for the plot
    fig.update_layout(
        title = plot_title,
        barmode = 'overlay',
        xaxis_title = 'Temperature (°F)',
        yaxis_title = 'Frequency',
        xaxis_range = x_range,
        legend = dict(
            x = 1,
            y = 1.3,
            xanchor = 'right',
            yanchor = 'top',
            bgcolor = 'rgba(255, 255, 255, 0.7)',
            bordercolor = 'black',
            borderwidth = 1,
        ),
        paper_bgcolor='#ecf0f1',
        margin = dict(
            l = 0,
            r = 0,
        ),
    )

    fig.update_traces(
        marker_line_color = 'white',
        marker_line_width = 1.5)

    return container, fig


# Callback for heatmaps
@callback(
    [
        Output('heatmap-energy', 'figure'),
        Output('heatmap-temp', 'figure')
    ],
    Input('heatmap-tabs', 'value')
)
# Function for heatmaps
def update_heatmap(region):
    # Initialize variables
    df = sj_df if region == 'sj' else sf_df

    value_col_energy = 'averagekwh'
    title_energy = f'Average Energy Usage ({region.upper()})'
    color_scale_energy = 'Bluyl'

    value_col_temp = 'tmax'
    title_temp = f'Average Max Temp ({region.upper()})'
    color_scale_temp = 'YlOrRd'

    # Pivot each df for the heatmap
    heatmap_data_energy = df.pivot_table(
        values = value_col_energy, 
        index = 'year', 
        columns = 'month',
        aggfunc = 'mean',
        observed = False
    )

    heatmap_data_temp = df.pivot_table(
        values = value_col_temp, 
        index = 'year', 
        columns = 'month',
        aggfunc = 'mean',
        observed = False
    )

    data = {
        'Energy': [heatmap_data_energy, color_scale_energy, title_energy],
        'Temp': [heatmap_data_temp, color_scale_temp, title_temp]
    }

    figs = []

    # Plot each data set
    for k, v in data.items():
        fig = px.imshow(
            v[0],
            labels = dict(
                x = "Month",
                y = "Year",
                color = k
            ),
            x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
            color_continuous_scale = v[1],
        )
        fig.update_layout(
            margin = {
                'r': 30,
                'l': 30,
                'b': 30
            },
            title = v[2],
            xaxis_title = None,
            yaxis_title = None,
            height = 600,
            paper_bgcolor = '#ecf0f1',
            xaxis = dict(
                tickangle = 45
            ),
        )

        figs.append(fig)
    
    return figs[0], figs[1]