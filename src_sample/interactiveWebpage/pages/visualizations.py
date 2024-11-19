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
import pickle
from joblib import dump, load

### Load Data ###
sj_df = pd.DataFrame()
sf_df = pd.DataFrame()

if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
    sj_df = load('joblib_files/base_data/sj_combined.joblib')

if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
    sf_df = load('joblib_files/base_data/sf_combined.joblib')

# Descriptions for each plot
sj_max_temp_heat = html.P(['Plot Description:', html.Br(), html.Br(), 'This heatmap presents the average monthly maximum and minimum temperatures in San Jose from 2013 to 2024 (°F).'])
sf_max_temp_heat = html.P(['Plot Description:', html.Br(), html.Br(), 'This heatmap presents the average monthly maximum and minimum temperatures in San Francisco from 2013 to 2024 (°F).'])
sj_averagekwh_heat = html.P(['Plot Description:', html.Br(), html.Br(), "This heatmap shows the distribution of average monthly energy usage in San Jose from 2013 to 2024 (kWh)."])
sf_averagekwh_heat = html.P(['Plot Description:', html.Br(), html.Br(), "This heatmap shows the distribution of average monthly energy usage in San Francisco from 2013 to 2024 (kWh)."])

#### Define the Container Partitions ####
# Visualizations page header
visualizations_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Visualizations',
                    ),
                    className = 'text-center mb-5 mt-5',
                    width = 12,
                    style = {
                        'font-size': '50px',
                        'height': '100%',
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
                        #'text-shadow': '2px 2px 4px #000000',
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
                            value = 'sj_avgkwh',
                            children = [
                                dcc.Tab(
                                    label = 'SJ Average kWh',
                                    value = 'sj_avgkwh',
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
                                    label = 'SF Average kWh',
                                    value = 'sf_avgkwh',
                                    style = {
                                        'backgroundColor': '#bdc3c7',
                                        'color': '#2c3e50'
                                    }, 
                                    selected_style = {
                                        'backgroundColor': '#1abc9c',
                                        'color': '#2c3e50'
                                    },
                                ),

                                dcc.Tab(
                                    label = 'SJ Max Temp',
                                    value = 'sj_tmax',
                                    style = {
                                        'backgroundColor': '#bdc3c7',
                                        'color': '#2c3e50'
                                    }, 
                                    selected_style = {
                                        'backgroundColor': '#1abc9c',
                                        'color': '#2c3e50'
                                    },
                                ),

                                dcc.Tab(
                                    label = 'SF Max Temp',
                                    value = 'sf_tmax',
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
                    className = 'mb-2',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'heatmap',
                            figure = {},
                        ),
                    ],
                    width = 7,
                ),

                dbc.Col(
                    [
                        html.Div(
                            id = 'heatmap_output_container',
                            children = [],
                            style = {
                                'color': '#0f0f0f',
                                # other stuff width, height, background-color, etc.
                            },
                        ),
                    ],
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
        Output(component_id = 'heatmap_output_container', component_property = 'children'),
        Output('heatmap', 'figure')
    ],
    Input('heatmap-tabs', 'value')
)
# Function for heatmaps
def update_heatmap(selected_tab):
    color_scale = 'Bluyl'
    container = ''
    df = ''

    # Choose dataset based on clicked tab
    if selected_tab == 'sj_avgkwh':
        df = sj_df
        value_column = 'averagekwh'
        title = 'Avg Energy Usage (SJ - 95110)'
        container = sj_averagekwh_heat

    elif selected_tab == 'sf_avgkwh':
        df = sf_df
        value_column = 'averagekwh'
        title = 'Avg Energy Usage (SF - 94102)'
        container = sf_averagekwh_heat

    elif selected_tab == 'sj_tmax':
        df = sj_df
        value_column = 'tmax'
        title = 'Avg Max Temp (SJ - 95110)'
        color_scale = 'YlOrRd'
        container = sj_max_temp_heat

    else:
        df = sf_df
        value_column = 'tmax'
        title = 'Avg Max Temp (SF - 94102)'
        color_scale = 'YlOrRd'
        container = sf_max_temp_heat

    # Create pivot table for heatmap: months (x-axis) and years (y-axis)
    heatmap_data = df.pivot_table(
        values = value_column, 
        index = 'year', 
        columns = 'month',
        aggfunc = 'mean',
        observed = False
    )

    # Generate the heatmap
    fig = px.imshow(
        heatmap_data,
        labels = dict(
            x = "Month",
            y = "Year",
            color = value_column
        ),
        x = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        y = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
        color_continuous_scale = color_scale
    )

    # Update the layout
    fig.update_layout(
        title = title,
        xaxis_title = "Month",
        yaxis_title = "Year",
        height = 600,
        paper_bgcolor='#ecf0f1',  # Outside the plot area background
    )
    
    return container, fig