import dash
from dash import html, dcc, callback
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.data_pipeline import find_regional_diff
import os
import pickle
from joblib import dump, load

##### Load Data #####
sj_df = pd.DataFrame()
sf_df = pd.DataFrame()

if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
    sj_df = load('joblib_files/base_data/sj_combined.joblib')

if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
    sf_df = load('joblib_files/base_data/sf_combined.joblib')


# Animated plot descriptions
comb_energy_jan = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of January.'])
comb_energy_feb = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of February.'])
comb_energy_mar = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of March.'])
comb_energy_apr = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of April.'])
comb_energy_may = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of May.'])
comb_energy_jun = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of June.'])
comb_energy_jul = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of July.'])
comb_energy_aug = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of August.'])
comb_energy_sep = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of September.'])
comb_energy_oct = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of October.'])
comb_energy_nov = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of November.'])
comb_energy_dec = html.P(['Plot Description:', html.Br(), html.Br(), 'This animated plot shows the average energy usage in both SJ and SF regions for the month of December.'])


#### Define the Container Partitions ####
# Home page header
home_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Effects of Weather on Energy Consumption in the Bay Area',
                    ),
                    className = 'text-center mb-5 mt-5',
                    width = 12,
                    style = {
                        'font-size': '50px',
                        'height': '100%',
                        'text-shadow': '2px 2px 4px #000000',
                    },
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# Home page introduction title
intro_text = html.Div(
    [
        html.P(
            "Introduction",
            style = {
                'text-align': 'center',
                'font-size': '40px',
                'font-variant': 'small-caps',
                'text-shadow': '2px 2px 4px #000000'
            },
        ),
    ],
)

# Home page introduction text
intro_content = html.Div(
    [
        html.P(
            [
                'This project explores the impact of climate change on energy consumption in the California Bay Area, with a focus on San Jose and San Francisco. \
                By analyzing historical weather data and energy usage trends, I aim to identify key weather factors, such as extreme temperatures, that influence \
                electricity demand. Ultimately, this project hopes to shed light on how shifts in climate can affect local energy consumption, with the potential \
                to apply these findings to other regions.',

                html.Br(), html.Br(),
                'Explore the various pages to seehow weather patterns correlate with energy usage in each region!'
            ],
            style = {
                'text-align': 'justify',
                'font-size': '20px',
                'margin': '20px 7%',
                'word-break': 'keep-all'
            },
        ),
    ],
    className = 'mb-5',
)

# Home page animated line plot by month
animated_plot_1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            id = 'animated-plot-tabs',
                            value = 'Jan',
                            children = [
                                dcc.Tab(
                                    label = 'Jan',
                                    value = 'Jan',
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
                                    label = 'Feb',
                                    value = 'Feb',
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
                                    label = 'Mar',
                                    value = 'Mar',
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
                                    label = 'Apr',
                                    value = 'Apr',
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
                                    label = 'May',
                                    value = 'May',
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
                                    label = 'Jun',
                                    value = 'Jun',
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
                                    label = 'Jul',
                                    value = 'Jul',
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
                                    label = 'Aug',
                                    value = 'Aug',
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
                                    label = 'Sep',
                                    value = 'Sep',
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
                                    label = 'Oct',
                                    value = 'Oct',
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
                                    label = 'Nov',
                                    value = 'Nov',
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
                                    label = 'Dec',
                                    value = 'Dec',
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
                            id = 'combined_energy_line_by_mo',
                            figure = {},
                        ),
                    ],
                    width = 7,
                ),

                dbc.Col(
                    [
                        html.Div(
                            id = 'animated_output_container',
                            children = [],
                            style = {
                                'color': '#0f0f0f',
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
    className = 'mb-5',
)



table_content = html.Div(id = 'table-content')

# Add the contents to the layout
layout = dbc.Container(
    [
        home_header,
        intro_text,
        intro_content,
        animated_plot_1,
    ],
    fluid = True
)

#### Plots ####
# Animated plot callback
@callback(
    [
        Output('animated_output_container', 'children'),
        Output('combined_energy_line_by_mo', 'figure')
    ],
    [
        Input('animated-plot-tabs', 'value'),
    ]
)
# Animated plot function
def update_energy_line_plot(selected_month):
    container = ''

    # Dictionary mapping months to corresponding containers
    month_container = {
        'Jan': comb_energy_jan,
        'Feb': comb_energy_feb,
        'Mar': comb_energy_mar,
        'Apr': comb_energy_apr,
        'May': comb_energy_may,
        'Jun': comb_energy_jun,
        'Jul': comb_energy_jul,
        'Aug': comb_energy_aug,
        'Sep': comb_energy_sep,
        'Oct': comb_energy_oct,
        'Nov': comb_energy_nov,
        'Dec': comb_energy_dec
    }

    # Get the container for the selected month
    container = month_container.get(selected_month)
    
    # Filter data for the selected month
    sf_filtered = sf_df[sf_df['month'] == selected_month]
    sj_filtered = sj_df[sj_df['month'] == selected_month]

    # Remove Jan - Jul for sj_filtered if year is 2013
    sj_filtered = sj_filtered.loc[~((sj_filtered['year'] == 2013) & (sj_filtered['month-numeric'].between(1, 7)))]

    # Reset the index after dropping rows
    sj_filtered = sj_filtered.reset_index(drop=True)

    # Find the regional differences in averagekwh
    region_avgkwhdiff = find_regional_diff(sj_filtered, sf_filtered, 'averagekwh', 'averagekwhdiff')
    region_filtered = region_avgkwhdiff[region_avgkwhdiff['month'] == selected_month]

    # Create the figure to hold the line plot
    fig = go.Figure()

    # Plot the full lines for both regions
    fig.add_trace(
        go.Scatter(
            x = sf_filtered['year-month'],
            y = sf_filtered['averagekwh'],
            mode = 'lines',
            name = 'SF',
            line = dict(color = 'blue'),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x = sj_filtered['year-month'],
            y = sj_filtered['averagekwh'],
            mode = 'lines', 
            name = 'SJ',
            line = dict(color = 'green'),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x = region_filtered['year-month'],
            y = region_filtered['averagekwhdiff'],
            mode = 'lines',
            name = 'SJ-SF Diff',
            line = dict(color = 'red'),
        )
    )

    # Add the dots to the line plot - set starting position
    fig.add_trace(
        go.Scatter(
            x = [sf_filtered['year-month'].values[0]],
            y = [sf_filtered['averagekwh'].values[0]],
            mode = 'markers',
            marker = dict(
                color = 'blue',
                size = 10),
            name = 'SF moving point',
            showlegend = False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x = [sj_filtered['year-month'].values[0]],
            y = [sj_filtered['averagekwh'].values[0]],
            mode = 'markers',
            marker = dict(
                color = 'green',
                size = 10),
            name = 'SJ moving point',
            showlegend = False,
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x = [region_filtered['year-month'].values[0]],
            y = [region_filtered['averagekwhdiff'].values[0]],
            mode = 'markers',
            marker = dict(
                color = 'red',
                size = 10),
            name = 'Regional Difference moving point',
            showlegend = False,
        )
    )
    
    # Set layout properties for the plot
    fig.update_layout(
        title = f'Average Energy Usage for {selected_month}',
        xaxis_title = 'Year-Month',
        yaxis_title = 'Avg Energy Usage (kWh)',
        xaxis_range = ['2012', '2025'],
        updatemenus = [
            dict(
                type = 'buttons',
                buttons = [
                    dict(
                        label = 'Play the animation',
                        method = 'animate',
                        args = [
                            None,
                            {
                                'frame': {
                                    'duration': 500,
                                    'redraw': False,
                                },
                                'mode': 'immediate',
                                'fromcurrent': True,
                                'transition': {'duration': 500},
                            },
                        ],
                    ),
                    dict(
                        label = 'Pause the animation',
                        method = 'animate',
                        args = [
                            None,
                            {
                                'frame': {
                                    'duration': 0,
                                    'redraw': True
                                },
                                'mode': 'immediate',
                                'fromcurrent': True,
                                'transition': {'duration': 0},
                            },
                        ],
                    ),
                ],
                direction = 'left',
                pad = {'r': 10, 't': 10},
                showactive = False,
                x = 0.5,
                xanchor = 'center',
                y = -0.4,
                yanchor = 'bottom',
            ),
        ],
        paper_bgcolor='#ecf0f1',  # Outside the plot area background
        autosize = True,
    )

    # Create frames for the animation
    frames = []
    max_len = min(len(sf_filtered), len(sj_filtered))

    # Loop over the data points
    for i in range(0, max_len):
        frame_data = [
            # Draw the lines
            go.Scatter(
                x = sf_filtered['year-month'],
                y = sf_filtered['averagekwh'],
                mode = 'lines',
                line = dict(color = 'blue'),
            ),

            go.Scatter(
                x = sj_filtered['year-month'],
                y = sj_filtered['averagekwh'],
                mode = 'lines',
                line = dict(color = 'green'),
            ),

            go.Scatter(
                x = region_filtered['year-month'],
                y = region_filtered['averagekwhdiff'],
                mode = 'lines',
                line = dict(color = 'red'),
            ),

            # Update the position of the moving dots for the lines
            go.Scatter(
                x = [sf_filtered['year-month'].values[i]],
                y = [sf_filtered['averagekwh'].values[i]],
                mode = 'markers',
                marker = dict(
                    color = 'blue',
                    size = 10
                ),
            ),

            go.Scatter(
                x = [sj_filtered['year-month'].values[i]],
                y = [sj_filtered['averagekwh'].values[i]],
                mode = 'markers',
                marker = dict(
                    color = 'green',
                    size = 10
                ),
            ),

            go.Scatter(
                x = [region_filtered['year-month'].values[i]],
                y = [region_filtered['averagekwhdiff'].values[i]],
                mode = 'markers',
                marker = dict(
                    color = 'red',
                    size = 10
                ),
            ),
        ]
        frames.append(
            go.Frame(
                data = frame_data,
                name = str(i)
            )
        )

    fig.update(frames = frames)

    return container, fig