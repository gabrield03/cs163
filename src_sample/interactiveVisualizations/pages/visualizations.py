import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# Import data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

### Preprocessing + pipeline ###
sj_df['month_numeric'] = sj_df['month'].map({
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
    'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
    'Nov': 11, 'Dec': 12
})
sf_df['month_numeric'] = sf_df['month'].map({
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
    'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
    'Nov': 11, 'Dec': 12
})

# Combine sj and sf
sj_df['region'] = 'San Jose'
sf_df['region'] = 'San Francisco'




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
                                'width': '95vh',
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
                            style = {'width': '95vh', 'height': '50vh'},
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
                                dcc.Tab(label = 'SJ Average kWh', value = 'sj_avgkwh'),
                                dcc.Tab(label = 'SF Average kWh', value = 'sf_avgkwh'),
                                dcc.Tab(label = 'SJ Max Temp', value = 'sj_tmax'),
                                dcc.Tab(label = 'SF Max Temp', value = 'sf_tmax'),
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
                            style = {'width': '95vh', 'height': '50vh'},
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
    sj_dff = sj_df.copy()

    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SJ - 95110)'
            column = 'averagekwh'
            color = '#6600CC'
            kde_color = '#000000'
            x_range = [250, 625]

            container = sj_averagekwh

        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SJ - 95110)'
            column = 'totalkwh'
            color = '#4C9900'
            kde_color = '#000000'
            x_range = [1750000, 4000000]

            container = sj_totalkwh

        # Create histogram
        fig = px.histogram(
            sj_dff,
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
        kde = gaussian_kde(sj_dff[column].dropna())
        kde_range = np.linspace(sj_dff[column].min(), sj_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x = kde_range,
                y = kde(kde_range) * len(sj_dff[column]) * (sj_dff[column].max() - sj_dff[column].min()) / 40,
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
            )
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title = 'Average Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title = 'Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SJ - 95110)'
        container = sj_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(
            go.Histogram(
                x = sj_dff['tmax'],
                nbinsx = 40,
                name = 'Max Temp (tmax)',
                marker_color = '#8B0000',
                opacity = 0.75
            )
        )

        # tmin histogram
        fig.add_trace(
            go.Histogram(
                x = sj_dff['tmin'],
                nbinsx = 40,
                name = 'Min Temp (tmin)',
                marker_color = '#003FFF',
                opacity = 0.75
            )
        )

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sj_dff['tmax'].dropna())
        tmax_range = np.linspace(sj_dff['tmax'].min(), sj_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(
            go.Scatter(
                x = tmax_range,
                y = tmax_kde(tmax_range) * len(sj_dff['tmax']) * (sj_dff['tmax'].max() - sj_dff['tmax'].min()) / 40,
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
        tmin_kde = gaussian_kde(sj_dff['tmin'].dropna())
        tmin_range = np.linspace(sj_dff['tmin'].min(), sj_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(
            go.Scatter(
                x = tmin_range,
                y = tmin_kde(tmin_range) * len(sj_dff['tmin']) * (sj_dff['tmin'].max() - sj_dff['tmin'].min()) / 40,
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
    sf_dff = sf_df.copy()

    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SF - 94102)'
            column = 'averagekwh'
            color = '#CC00CC'
            kde_color = '#000000'
            x_range = [200, 450]

            container = sf_averagekwh

        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SF - 94102)'
            column = 'totalkwh'
            color = '#999900'
            kde_color = '#000000'
            x_range = [2500000, 5000000]

            container = sf_totalkwh

        # Create histogram
        fig = px.histogram(
            sf_dff,
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
        kde = gaussian_kde(sf_dff[column].dropna())
        kde_range = np.linspace(sf_dff[column].min(), sf_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x = kde_range,
                y = kde(kde_range) * len(sf_dff[column]) * (sf_dff[column].max() - sf_dff[column].min()) / 40,
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
            )
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title = 'Average Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title = 'Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SF - 94102)'
        container = sf_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(
            go.Histogram(
                x = sf_dff['tmax'],
                nbinsx = 40,
                name = 'Max Temp (tmax)',
                marker_color = '#8B0000',
                opacity = 0.75
            )
        )

        # tmin histogram
        fig.add_trace(
            go.Histogram(
                x = sf_dff['tmin'],
                nbinsx = 30,
                name = 'Min Temp (tmin)',
                marker_color = '#003FFF',
                opacity = 0.75
            )
        )

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sf_dff['tmax'].dropna())
        tmax_range = np.linspace(sf_dff['tmax'].min(), sf_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(
            go.Scatter(
                x = tmax_range,
                y = tmax_kde(tmax_range) * len(sf_dff['tmax']) * (sf_dff['tmax'].max() - sf_dff['tmax'].min()) / 40,
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
        tmin_kde = gaussian_kde(sf_dff['tmin'].dropna())
        tmin_range = np.linspace(sf_dff['tmin'].min(), sf_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(
            go.Scatter(
                x = tmin_range,
                y = tmin_kde(tmin_range) * len(sf_dff['tmin']) * (sf_dff['tmin'].max() - sf_dff['tmin'].min()) / 30,
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
            )
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
        title = 'San Jose Average kWh Heatmap'
        container = sj_averagekwh_heat

    elif selected_tab == 'sf_avgkwh':
        df = sf_df
        value_column = 'averagekwh'
        title = 'San Francisco Average kWh Heatmap'
        container = sf_averagekwh_heat

    elif selected_tab == 'sj_tmax':
        df = sj_df
        value_column = 'tmax'
        title = 'San Jose Avg Max Temperature Heatmap'
        color_scale = 'Inferno'
        container = sj_max_temp_heat

    else:
        df = sf_df
        value_column = 'tmax'
        title = 'San Francisco Avg Max Temperature Heatmap'
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
        height = 600
    )
    
    return container, fig