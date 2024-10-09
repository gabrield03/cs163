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

##### Load Data #####
sj_df = ''
sf_df = ''

if os.path.exists('sj_combined.pkl'):
    with open('sj_combined.pkl', 'rb') as f:
        sj_df = pickle.load(f)

if os.path.exists('sf_combined.pkl'):
    with open('sf_combined.pkl', 'rb') as f:
        sf_df = pickle.load(f)


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
        html.H1(
            'Effects of Weather on Energy Consumption in the Bay Area',
            className = 'text-center mt-5 mb-5',
        ),
    ],
)

# home_picture = html.Div(
#     [
#         html.Div(
#             style = {
#                 'maxWidth': '800px',
#                 'margin': '0 auto'
#             },
#             children = [
#                 html.Div(
#                     style = {
#                         'textAlign': 'center',
#                         'margin': '20px'
#                     },
#                     children = [
#                         html.Img(
#                             src = '/assets/bayareascenic.jpg',
#                             style = {
#                                 'width': '100%',
#                                 'maxWidth': '800px',
#                                 'height': 'auto',
#                                 'borderRadius': '10px',
#                             },
#                         ),
#                     ],
#                 ),
#             ],
#         ),
#     ],
# )

# Home page introduction title
intro_text = html.Div(
    [
        html.H3(
            "Introduction",
            style = {
                'text-align': 'center'
            },
            className = 'mt-5',
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
                                    label = 'January',
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
                                    label = 'February',
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
                                    label = 'March',
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
                                    label = 'April',
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
                                    label = 'June',
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
                                    label = 'July',
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
                                    label = 'August',
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
                                    label = 'September',
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
                                    label = 'October',
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
                                    label = 'November',
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
                                    label = 'December',
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
                            # style = {
                            #     'width': '70vh',
                            #     'height': '50vh'
                            # },
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
        # home_picture,
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
    Input('animated-plot-tabs', 'value')
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
        title = f'Average Energy Usage (kWh) for {selected_month} (Animated)',
        xaxis_title = 'Year-Month',
        yaxis_title = 'Avg Energy Usage (kWh)',
        xaxis_range = ['2012', '2025'],
        updatemenus = [
            dict(
                type = 'buttons',
                buttons = [
                    dict(
                        label = 'Play',
                        method = 'animate',
                        args = [
                            None,
                            {
                                'frame': {'duration': 500, 'redraw': False},
                                'fromcurrent': True,
                                'transition': {'duration': 750},
                                'loop': True,
                            },
                        ],
                    ),
                ],
            ),
        ],
        paper_bgcolor='#ecf0f1',  # Outside the plot area background
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