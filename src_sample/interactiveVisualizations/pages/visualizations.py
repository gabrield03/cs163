import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from utils.data_pipeline import load_and_preprocess_data

#### Load Data ###
sj_df, sf_df = load_and_preprocess_data()

# Descriptions for each plot
sj_averagekwh = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram shows the distribution of average monthly energy usage in San Jose (kWh)."])
sf_averagekwh = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram shows the distribution of average monthly energy usage in San Francisco (kWh)."])
sj_averagekwh_heat = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This heatmap shows the distribution of average monthly energy usage in San Jose from 2013 to 2024 (kWh)."])
sf_averagekwh_heat = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This heatmap shows the distribution of average monthly energy usage in San Francisco from 2013 to 2024 (kWh)."])

sj_totalkwh = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram displays the distribution of total energy usage in San Jose over the months (kWh)."])
sf_totalkwh = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram displays the distribution of total energy usage in San Francisco over the months (kWh)."])

sj_max_min_temp = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram presents the average monthly maximum and minimum temperatures in San Jose (°F)."])
sf_max_min_temp = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This histogram presents the average monthly maximum and minimum temperatures in San Francisco (°F)."])
sj_max_temp_heat = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This heatmap presents the average monthly maximum and minimum temperatures in San Jose from 2013 to 2024 (°F)."])
sf_max_temp_heat = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This heatmap presents the average monthly maximum and minimum temperatures in San Francisco from 2013 to 2024 (°F)."])



### End preprocessing data ###

# # Layout of the Dash app
visualizations_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        'Passage about the visualizations created:',
                    ),
                    className = 'text-center mb-5',
                    width = 12,
                    style = {'height': '100%'}
                ),
            ],
        ),
    ]
)

# 1st row of plots - SJ Energy Usage and Temperature
sj_histplots_1 = html.Div(
    [
        dbc.Row(
            [
                html.H3('San Jose Energy Usage and Weather History'),
            ],
            className = 'mb-2 text-center',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id = 'select_sj_option',
                            options = [
                                {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                                {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                                {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}
                            ],
                            multi = False,
                            value = 'averagekwh',
                            style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                        ),
                    ],
                    width = {'size': 3}, className = 'mb-2'
                ),
            ]
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'sj_data',
                            figure = {},
                            style = {
                                # 'width': '95vh',
                                'width': '70vh',
                                'height': '50vh'
                            },
                        ),
                    ],
                    width = 7,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id = 'sj_output_container',
                            children = [],
                            style = {
                                'color': '#0f0f0f', # text color
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

# 2nd row of plots - SF Energy Usage and Temperature
sf_histplots_1 = html.Div(
    [
        dbc.Row(
            [
                html.H3('San Francisco Energy Usage and Weather History'),
            ],
            className = 'mb-2 text-center',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id = 'select_sf_option',
                            options = [
                                {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                                {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                                {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}
                            ],
                            multi = False,
                            value = 'averagekwh',
                            style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                        ),
                    ],
                    width = {'size': 3}, className = 'mb-2',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'sf_data',
                            figure = {},
                            style = {
                                # 'width': '95vh',
                                'width': '70vh',
                                'height': '50vh'
                            },
                        ),
                    ],
                    width = 7,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id = 'sf_output_container',
                            children = [],
                            style = {
                                'color': '#0f0f0f', # text color
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

# 3rd row of plots - Both Regions Average Energy Usage and AverageMax Temperature
comb_heatmaps = html.Div(
    [
        dbc.Row(
            [
                html.H3('Heatmaps for San Jose and San Francisco Energy Usage and Weather History'),
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
                                    style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                                    selected_style={'backgroundColor': '#1abc9c', 'color': '#2c3e50'},
                                ),
                                dcc.Tab(
                                    label = 'SF Average kWh',
                                    value = 'sf_avgkwh',
                                    style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                                    selected_style={'backgroundColor': '#1abc9c', 'color': '#2c3e50'},
                                ),
                                dcc.Tab(
                                    label = 'SJ Max Temp',
                                    value = 'sj_tmax',
                                    style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                                    selected_style={'backgroundColor': '#1abc9c', 'color': '#2c3e50'},
                                ),
                                dcc.Tab(
                                    label = 'SF Max Temp',
                                    value = 'sf_tmax',
                                    style={'backgroundColor': '#bdc3c7', 'color': '#2c3e50'}, 
                                    selected_style={'backgroundColor': '#1abc9c', 'color': '#2c3e50'},
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
                            style = {
                                # 'width': '95vh',
                                'width': '70vh',
                                'height': '50vh'
                            },
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
                                'color': '#0f0f0f', # text color
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

layout = dbc.Container(
    [
        visualizations_header,
        sj_histplots_1,
        sf_histplots_1,
        comb_heatmaps,
    ],
    fluid = True
)

# Connect the Plotly graphs with Dash Components
## Plots 1 - SJ
@callback(
    [Output(component_id = 'sj_output_container', component_property = 'children'),
     Output(component_id = 'sj_data', component_property = 'figure')],
    [Input(component_id = 'select_sj_option', component_property = 'value')]
)
def update_sj_graph(option_selected):
    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Avg Energy Usage by Month (SJ - 95110)'
            column = 'averagekwh'
            color = '#6600CC'
            kde_color = '#000000'
            x_range = [250, 625]

            container = sj_averagekwh

        else:
            plot_title = 'Total Energy Usage by Month (SJ - 95110)'
            column = 'totalkwh'
            color = '#4C9900'
            kde_color = '#000000'
            x_range = [1750000, 4000000]

            container = sj_totalkwh

        # Create histogram
        try:
            fig = px.histogram(
                sj_df,
                x = column,
                nbins = 40,
                title = plot_title,
                color_discrete_sequence = [color],
            )
        except:
            fig = px.histogram(
                sj_df,
                x = column,
                nbins = 40,
                title = plot_title,
                color_discrete_sequence = [color],
            )

        # Update the figure to set white outlines for the bars
        fig.update_traces(
            marker_line_color = 'white',
            marker_line_width = 1.5)

        # Calculate KDE
        kde = gaussian_kde(sj_df[column].dropna())
        kde_range = np.linspace(sj_df[column].min(), sj_df[column].max(), 100)

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x = kde_range,
                y = kde(kde_range) * len(sj_df[column]) * (sj_df[column].max() - sj_df[column].min()) / 40,
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
            paper_bgcolor='#ecf0f1',  # Outside the plot area background
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title = 'Avg Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title = 'Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Avg Max and Min Temps by Month (SJ - 95110)'
        container = sj_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(
            go.Histogram(
                x = sj_df['tmax'],
                nbinsx = 40,
                name = 'Max Temp (tmax)',
                marker_color = '#8B0000',
                opacity = 0.75
            )
        )

        # tmin histogram
        fig.add_trace(
            go.Histogram(
                x = sj_df['tmin'],
                nbinsx = 40,
                name = 'Min Temp (tmin)',
                marker_color = '#003FFF',
                opacity = 0.75
            )
        )

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sj_df['tmax'].dropna())
        tmax_range = np.linspace(sj_df['tmax'].min(), sj_df['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(
            go.Scatter(
                x = tmax_range,
                y = tmax_kde(tmax_range) * len(sj_df['tmax']) * (sj_df['tmax'].max() - sj_df['tmax'].min()) / 40,
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
        tmin_kde = gaussian_kde(sj_df['tmin'].dropna())
        tmin_range = np.linspace(sj_df['tmin'].min(), sj_df['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(
            go.Scatter(
                x = tmin_range,
                y = tmin_kde(tmin_range) * len(sj_df['tmin']) * (sj_df['tmin'].max() - sj_df['tmin'].min()) / 40,
                mode = 'lines',
                name = 'Min Temp KDE',
                line = dict(color = '#A5E5FF', width = 2, dash = 'dash')
            )
        )

        # Update layout for the plot
        fig.update_layout(
            title = plot_title,
            barmode = 'overlay',
            xaxis_title = 'Temperature (F)',
            yaxis_title = 'Frequency',
            xaxis_range = [30, 90],
            legend = dict(
                x = 1,
                y = 1,
                xanchor = 'right',
                yanchor = 'top',
                bgcolor = 'rgba(255, 255, 255, 0.7)',
                bordercolor = 'black',
                borderwidth = 1,
            ),
            paper_bgcolor='#ecf0f1',  # Outside the plot area background
        )

        fig.update_traces(
            marker_line_color = 'white',
            marker_line_width = 1.5)

        return container, fig


## Plots 2 - SF
@callback(
    [Output(component_id = 'sf_output_container', component_property = 'children'),
     Output(component_id = 'sf_data', component_property = 'figure')],
    [Input(component_id = 'select_sf_option', component_property = 'value')]
)
def update_sf_graph(option_selected):
    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Avg Energy Usage by Month (SF - 94102)'
            column = 'averagekwh'
            color = '#CC00CC'
            kde_color = '#000000'
            x_range = [200, 450]

            container = sf_averagekwh

        else:
            plot_title = 'Total Energy Usage by Month (SF - 94102)'
            column = 'totalkwh'
            color = '#999900'
            kde_color = '#000000'
            x_range = [2500000, 5000000]

            container = sf_totalkwh

        # Create histogram
        fig = px.histogram(
            sf_df,
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
        kde = gaussian_kde(sf_df[column].dropna())
        kde_range = np.linspace(sf_df[column].min(), sf_df[column].max(), 100)

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x = kde_range,
                y = kde(kde_range) * len(sf_df[column]) * (sf_df[column].max() - sf_df[column].min()) / 40,
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
            paper_bgcolor='#ecf0f1',  # Outside the plot area background
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title = 'Avg Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title = 'Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Avg Max and Min Temps by Month (SF - 94102)'
        container = sf_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(
            go.Histogram(
                x = sf_df['tmax'],
                nbinsx = 40,
                name = 'Max Temp (tmax)',
                marker_color = '#8B0000',
                opacity = 0.75
            )
        )

        # tmin histogram
        fig.add_trace(
            go.Histogram(
                x = sf_df['tmin'],
                nbinsx = 30,
                name = 'Min Temp (tmin)',
                marker_color = '#003FFF',
                opacity = 0.75
            )
        )

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(['tmax'].dropna())
        tmax_range = np.linspace(sf_df['tmax'].min(), sf_df['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(
            go.Scatter(
                x = tmax_range,
                y = tmax_kde(tmax_range) * len(sf_df['tmax']) * (sf_df['tmax'].max() - sf_df['tmax'].min()) / 40,
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
        tmin_kde = gaussian_kde(sf_df['tmin'].dropna())
        tmin_range = np.linspace(sf_df['tmin'].min(), sf_df['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(
            go.Scatter(
                x = tmin_range,
                y = tmin_kde(tmin_range) * len(sf_df['tmin']) * (sf_df['tmin'].max() - sf_df['tmin'].min()) / 30,
                mode = 'lines',
                name = 'Min Temp KDE',
                line = dict(color = '#A5E5FF', width = 2, dash = 'dash')
            )
        )

        # Update layout for the plot
        fig.update_layout(
            title = plot_title,
            barmode = 'overlay',
            xaxis_title = 'Temperature (F)',
            yaxis_title = 'Frequency',
            xaxis_range = [30, 90],
            legend = dict(
                x = 1,
                y = 1,
                xanchor = 'right',
                yanchor = 'top',
                bgcolor = 'rgba(255, 255, 255, 0.7)',
                bordercolor = 'black',
                borderwidth = 1,
            ),
            paper_bgcolor='#ecf0f1',  # Outside the plot area background
        )

        fig.update_traces(
            marker_line_color = 'white',
            marker_line_width = 1.5)

        return container, fig


# SJ Heatmap
# Callback to update heatmap
@callback(
    [Output(component_id = 'heatmap_output_container', component_property = 'children'),
     Output('heatmap', 'figure')],
     Input('heatmap-tabs', 'value')
)
def update_heatmap(selected_tab):
    color_scale = 'Aggrnyl'
    container = ''

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
        color_scale = 'Inferno'
        container = sj_max_temp_heat

    else:
        df = sf_df
        value_column = 'tmax'
        title = 'Avg Max Temp (SF - 94102)'
        color_scale = 'Inferno'
        container = sf_max_temp_heat

    # Create pivot table for heatmap: months (x-axis) and years (y-axis)
    heatmap_data = df.pivot_table(
        values = value_column, 
        index = 'year', 
        columns = 'month_numeric',
        aggfunc = 'mean'
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
        y = sorted(df['year'].unique(), reverse = True),
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