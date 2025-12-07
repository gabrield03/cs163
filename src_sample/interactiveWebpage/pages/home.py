from dash import html, dcc, Input, Output, callback, clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math
import numpy as np

from  utils.data_pipeline import processing_pipeline, load_joblib_from_github

import requests
from io import BytesIO
import os
from joblib import load
import json


### Load Data ###
sj_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sj_combined.joblib"
sf_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sf_combined.joblib"

sj_df = load_joblib_from_github(sj_url)
sf_df = load_joblib_from_github(sf_url)

sj_df.drop(columns = ['awnd', 'wdf2', 'wdf5', 'wsf2', 'wsf5'], inplace = True)
# Adding a unique identifier for each dataframe to keep track of the region
sj_df['region'] = 'sj'
sf_df['region'] = 'sf'

# Concatenating the dataframes
combined_df = pd.concat([sj_df, sf_df], axis=0).reset_index(drop=True)
# Calculate 90th quantile
q90_tmax = combined_df.groupby('region')['tmax'].quantile(0.90).to_dict()
# Set tmax threshold to 90th quantile
combined_df['is_hot_extreme'] = combined_df.apply(lambda row: row['tmax'] >= q90_tmax[row['region']], axis=1)
# Calculate 10th quantile
q10_tmin = combined_df.groupby('region')['tmin'].quantile(0.10).to_dict()
# Set tmin threshold to 10th quantile
combined_df['is_cold_extreme'] = combined_df.apply(lambda row: row['tmin'] <= q10_tmin[row['region']], axis=1)
combined_df['year'] = combined_df['year'].astype(str)


token = open('assets/.mapbox_token').read()


# Define the layout for the home page
home_front_section = html.Div(
    children = [
        # Looping sf video
        html.Video(
            id = 'home-front-video',
            src = '/assets/visuals/sf-city1.mp4',
            autoPlay = True,
            loop = True,
            muted = True,
            controls = False,
            style = {
                'height': '75vh',
                'width': '100vw',
                'maxWidth': '100vw',
                'objectFit': 'cover',
                'position': 'absolute',
                'top': '0',
                'left': '0',
                'zIndex': '-1',
                'margin': 0,
            },
        ),
        # Home page title
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.P(
                                    'Weather Impact On Bay Area Energy',
                                    style = {
                                        'color': 'white',
                                        
                                        'fontSize': '3vw',
                                        'fontweight': 'bold',
                                        'fontFamily': 'serif',
                                        'font-variant': 'small-caps',
                                        'height': '100%',
                                        'text-shadow': '2px 2px 4px #000000',
                                    },
                                ),
                            ],
                            width = 12,
                        ),
                    ],
                ),
            ],
            style = {
                'textAlign': 'center',
                'zIndex': '1',
            },
            className = 'mt-5',
        ),
        # Home page looping list of words
        html.Div(
            [
                html.Div('Energy Consumption', id = 'word-energy-consumption', className = 'fade-word'),
                html.Div('Max Temperature', id = 'word-max-temp', className = 'fade-word'),
                html.Div('Min Temperature', id = 'word-min-temp', className = 'fade-word'),
                html.Div('Precipitation', id = 'word-precipitation', className = 'fade-word'),
                html.Div('Wind Speed', id = 'word-wind-speed', className = 'fade-word'),
            ],
            style = {
                'color': 'white',
                'textAlign': 'center',
                'fontSize': '2vw',
                'fontWeight': 'bold',
                'fontFamily': 'sans-serif',
                'height': '100%',
                'display': 'flex',
                'flexDirection': 'column',
                'justifyContent': 'center',
                'alignItems': 'center',
                'position': 'absolute',
                'top': '20%',
                'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'zIndex': '1',
            },
        ),
        # Dark gradient
        html.Div(
            id = 'home-front-gradient',
            style = {
                'height': '35vh',
                'width': '100vw',
                'maxWidth': '100vw',
                'position': 'absolute',
                'background': 'linear-gradient(to bottom, rgba(0,0,0,0), rgba(0,0,0,.3), rgba(0,0,0,.6), rgba(0,0,0,0.9), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1))',
                'bottom': '0',
                'left': '0',
                'zIndex': '-1',
            },
        ),

        dcc.Interval(id = 'page-load-interval', interval = 3000, n_intervals = 0),
    ],
)

# Random forest for feature importances
feature_importances_extreme_weather_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                'What Influences Energy Usage?',
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '27px',
                                'word-break': 'keep-all',
                                'font-style': 'normal',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'San Jose is most sensitive to ',
                                html.Span(
                                    'seasonality ',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': 'red',
                                    },
                                ),

                                'whereas San Francisco is most sensitive to ',
                                html.Span(
                                    'temperature extremes',
                                    id = 'tmax_tooltip',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': 'red',
                                    },
                                ),
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '18px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 5,
                ),
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        html.P(
                            [
                                'Shifts in Climate',
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '27px',
                                'word-break': 'keep-all',
                                'font-style': 'normal',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'San Jose has experienced an ',
                                html.Span(
                                    'increase in extreme hot ',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': 'red',
                                    },
                                ),
                                'events and San Francisco has experienced an ',
                                html.Span(
                                    'increase in extreme cold ',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': 'lightblue',
                                    },
                                ),
                                'events.',
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '18px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 5,
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'feature_importances_section',
                        ),
                    ],
                    width = 5,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'extreme-weather',
                        )
                    ],
                    width = 6,
                ),
            ],
            justify = 'between',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Dropdown(
                            id = 'region_option',
                            options = [
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            multi = False,
                            value = 'sj',
                            style = {
                                'backgroundColor': '#bdc3c7',
                                'color': '#2c3e50',
                            }, 
                        ),
                    ],
                    width = 2,
                ),
                dbc.Col(
                    [
                        dcc.Tabs(
                            id = 'extreme-weather-tabs',
                            value = 'hot_events',
                            children = [
                                dcc.Tab(
                                    label = 'Hot Events',
                                    value = 'hot_events',
                                    style = {
                                        'backgroundColor': 'transparent',
                                        'color': 'white',
                                    }, 
                                    selected_style={
                                        'backgroundColor': 'transparent',
                                        'color': 'red',
                                    },
                                ),
                                dcc.Tab(
                                    label = 'Cold Events',
                                    value = 'cold_events',
                                    style = {
                                        'backgroundColor': 'transparent',
                                        'color': 'white',
                                    }, 
                                    selected_style={
                                        'backgroundColor': 'transparent',
                                        'color': 'lightblue',
                                    },
                                ),
                            ],
                            className = 'mb-2',
                        ),
                    ],
                    width = 6,
                ),
            ],
            justify = 'between',
        ),
    ],
    className = 'mb-20 mt-5',
)

# Slider input for chloropleth plot + predictions
hypothetical_input_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                'Why Should I Care?',
                            ],
                            className = 'text-center',
                            style = {
                                'color': 'white',
                                'font-size': '27px',
                                'word-break': 'keep-all',
                                'font-style': 'normal',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 12,
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        html.P(
                            [
                                'Similar temperature values ',
                                'impact energy usage in each region differently.',

                                html.Br(),

                                'As local temperatures increase, San Jose\'s energy usage ',
                                html.Span(
                                    'increases ',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': '#8B008B',
                                    },
                                ),

                                'and San Francisco\'s ',
                                html.Span(
                                    'decreases ',
                                    style = {
                                        'font-weight': 'bold',
                                        'color': '#FFD700',
                                    },
                                ),
                                '.',
                                
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '18px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 8,
                ),
                dbc.Col([], width = 2),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col([], width = 1),
                dbc.Col(
                    [
                        html.P(
                            'Max Temperature (°F)',
                            style = {
                                'textAlign': 'center',
                                'color': 'white'
                            }
                        ),
                        dcc.Slider(
                            60, 110,
                            step = None,
                            marks = {
                                i: '{}'.format(i + 60) for i in range(0, 60, 10)
                            },
                            id = 'tmax-slider',
                            value = 60,
                        ),
                    ],
                    width = 5,
                ),
                dbc.Col(
                    [
                        html.P(
                            'Min Temperature (°F)',
                            style = {
                                'textAlign': 'center',
                                'color': 'white'
                            }
                        ),
                        dcc.Slider(
                            0, 50,
                            step = None,
                            marks = {
                                i: '{}'.format(i) for i in range(0, 60, 10)
                            },
                            id = 'tmin-slider',
                            value = 0,
                        ),
                    ],
                    width = 5,
                ),
                dbc.Col([], width = 1),
            ],
        ),
    ],
    className = 'mb-5',
)

chloropleth_desc_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                'Regional color shows the difference of average energy (kWh) usage as temperatures change',
                                'from the base temperatures: Max Temperature (°F): 60 and Min Temperature (°F): 0',
                            ],
                            className = 'text-center',
                            style = {
                                'color': 'white',
                                #'font-size': '27px',
                                'word-break': 'keep-all',
                                'font-style': 'normal',
                                #'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 12,
                ),
            ],
        ),
    ],
),

# Chloropleth plot
chloropleth_map_section = html.Div(
    [
        dcc.Graph(
            id = 'chloropleth-output', 
        ),
    ],
    style = {
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center', 
    },
)

# Hypothetical input predictions
prediction_output_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.P(
                                    'SF Prediction: ',
                                    style = {
                                        'display': 'inline',
                                    }
                                ),
                                html.Span(
                                    id = 'sf-prediction-output',
                                    style = {
                                        'display': 'inline',
                                    },
                                ),
                            ],
                            style = {
                                'textAlign': 'center',
                                'color': 'white',
                                'fontSize': '24px',
                            },
                        ),
                    ],
                    width = 6,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.P(
                                    'SJ Prediction: ',
                                    style = {
                                        'display': 'inline',
                                    }
                                ),
                                html.Span(
                                    id = 'sj-prediction-output',
                                    style = {
                                        'display': 'inline',
                                    },
                                ),
                            ],
                            style = {
                                'textAlign': 'center',
                                'color': 'white',
                                'fontSize': '24px',
                            },
                        ),
                    ],
                    width = 6,
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.P(
                                    [
                                        '1 kWh is equivalent to running an incandescent ',
                                        'lightbulb (100-watts) for ',
                                        html.Span(
                                            '10 hours',
                                            style = {
                                                'color': 'yellow'
                                            }
                                        )
                                    ],
                                    style = {
                                        'textAlign': 'center',
                                        'color': 'white',
                                        'fontSize': '20px',
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
    className = 'mb-20 mt-5',
)

info_combined_section = html.Div(
    [
        feature_importances_extreme_weather_section,
        hypothetical_input_section,
        chloropleth_map_section,
        prediction_output_section,
    ],
    style = {
        'backgroundColor': 'black',
        'position': 'absolute',
        'top': '75vh',
        'left': 0,
        'zIndex': '1',
        'padding': '0px 100px',
        'width': '100vw',
        'margin': 0,
    },
)

layout = dbc.Container(
    [
        home_front_section,
        info_combined_section,
    ],
    fluid = True,
)

# Client-side callback for fading in and out words
clientside_callback(
    """
    function(n_intervals) {
        const words = [
            "word-max-temp", 
            "word-min-temp", 
            "word-precipitation", 
            "word-wind-speed", 
            "word-energy-consumption"
        ];

        // Hide all words
        words.forEach(function(id) {
            const element = document.getElementById(id);
            if (element) {
                element.classList.remove("visible");
            }
        });

        // Show the next word
        const index = n_intervals % words.length;
        const nextWord = document.getElementById(words[index]);
        if (nextWord) {
            nextWord.classList.add("visible");
        }

        // Return the n_intervals to continue counting for the next cycle
        return n_intervals;
    }
    """,
    Output('page-load-interval', 'n_intervals'),
    Input('page-load-interval', 'n_intervals')
)

# Callback for Feature Importances
@callback(
    [
        Output('feature_importances_section', 'figure')
    ],
    [
        Input('region_option', 'value'),
    ]
)
# Animated plot function
def update_feature_importances_section(loc):
    importances_df = None

    df = None
    df = sj_df if loc == 'sj' else sf_df

    # importances_fn = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    # if not os.path.exists(importances_fn):
    #     importances_df = processing_pipeline(df, loc)
    # else:
    #     importances_df = load(importances_fn)



    importances_df_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/processed_data/{loc}_importances_df.joblib"
    importances_df = load_joblib_from_github(importances_df_url)

    rename_cols = {
        'season': 'Season', 'month': 'Month', 'year': 'Year',
        'tmax': 'Max (°F)', 'tmin': 'Min (°F)', 
        'prcp': 'Precip', 'totalcustomers': 'Customers'
    }

    importances_df['feature'] = importances_df['feature'].replace(rename_cols)
    importances_df.sort_values(by = ['importances'], ascending = True, inplace = True)
    importances_df['color_group'] = ['Top' if i > 3 else 'Other' for i in range(len(importances_df))]

    # Separate data for top and other bars
    top_df = importances_df[importances_df['color_group'] == 'Top'].reset_index(drop = True)
    other_df = importances_df[importances_df['color_group'] == 'Other'].reset_index(drop = True)

    # Create bar traces for top and other groups
    fig = px.bar(
        other_df,
        y = 'feature',
        x = 'importances',
        color = 'color_group',
        color_discrete_map = {
            'Top': 'green',
            'Other': 'gray'
        },
        range_x = [0, 0.48],
        orientation = 'h',
    )
    fig.add_bar(
        y = top_df['feature'],
        x = top_df['importances'],
        text = top_df['importances'].round(3),
        textposition = 'outside',
        textfont = dict(
            color = 'white'
        ),
        marker_color = 'green',
        orientation = 'h',
    )

    fig.update_layout(
        xaxis_title = None,
        yaxis_title = None,
        xaxis_showticklabels = False,
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        showlegend = False,
        xaxis = dict(
            showgrid = True,
            gridcolor = 'rgba(0,0,0,0)'
        ),
        yaxis = dict(
            color = 'white'
        ),
        margin = dict(
            l = 0,
            r = 0
        ),
    )

    return [fig]

# Callback for extreme weather bar plots
@callback(
    [
        Output('extreme-weather', 'figure')
    ],
    Input('extreme-weather-tabs', 'value')
)
# Function for the extreme weather bar plots
def update_extreme_weather(selected_tab):
    yearly_extreme_counts = combined_df[combined_df['is_hot_extreme']].groupby(['region', 'year']).size()
    yearly_total_counts = combined_df.groupby(['region', 'year']).size()
    yearly_extreme_frequency = (yearly_extreme_counts / yearly_total_counts * 100).fillna(0).reset_index()
    
    yearly_extreme_frequency.columns = ['region', 'year', 'occurrence_percentage']

    if selected_tab == 'cold_events':
        yearly_extreme_counts = combined_df[combined_df['is_cold_extreme']].groupby(['region', 'year']).size()
        yearly_total_counts = combined_df.groupby(['region', 'year']).size()
        yearly_extreme_frequency = (yearly_extreme_counts / yearly_total_counts * 100).fillna(0).reset_index()
        
        yearly_extreme_frequency.columns = ['region', 'year', 'occurrence_percentage']

    yearly_extreme_frequency = yearly_extreme_frequency[yearly_extreme_frequency['year'] != '2024']

    # Add regional bars
    fig = px.bar(
        yearly_extreme_frequency,
        x = 'year',
        y = 'occurrence_percentage',
        color = 'region',
        barmode = 'group',
        labels = {
            'occurrence_percentage': 'Occurrence %',
            'year': 'Year'
        },
        title = f"{'Hot' if selected_tab == 'hot_events' else 'Cold'} Events Occurrence Percentage by Year",
        color_discrete_sequence = ['#710280', '#808000']
    )

    # Add regional regression lines
    for region in yearly_extreme_frequency['region'].unique():
        region_data = yearly_extreme_frequency[yearly_extreme_frequency['region'] == region]
        
        # Regression coefficients
        x = region_data['year'].values.astype(float)
        y = region_data['occurrence_percentage'].values
        coeffs = np.polyfit(x, y, 1)
        trendline = coeffs[0] * x + coeffs[1]
        
        fig.add_trace(
            go.Scatter(
                x = region_data['year'],
                y = trendline,
                mode = 'lines',
                name = f'{region} reg line',
                line = dict(
                    width = 2,
                    dash = 'dash',
                    color = '#F9F902' if region == 'sj' else '#E100FF'
                )
            )
        )

    fig.update_layout(
        title_font_color = 'white',
        xaxis_title = None,
        yaxis_title = "Occurrence %",
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        xaxis = dict(
            showgrid = True,
            gridcolor = 'rgba(0,0,0,0)',
            color = 'white',
            tickangle = 45,
        ),
        yaxis = dict(
            color = 'white'
        ),
        margin = dict(
            l = 0,
            r = 0,
        ),
        legend = dict(
            yanchor = "top",
            y = 1.2,
            xanchor = "right",
            x = 1.25,
            font = dict(
                family = "Courier",
                size = 12,
                color = "white",
            )
        )
    )

    fig.update_yaxes(
        showgrid = False,
    )

    return [fig]


# Callback and function for energy predictions
@callback(
    [
        Output('chloropleth-output', 'figure'),
        Output('sf-prediction-output', 'children'),
        Output('sj-prediction-output', 'children'),
        
    ],
    [
        Input('tmax-slider', 'value'),
        Input('tmin-slider', 'value')
    ]
)
def update_chloropleth(tmax, tmin):
    tmax = float(tmax)
    tmin = float(tmin)

    sj_temp_at_low = 431 # sj low 431, high: 449
    sf_temp_at_low = 305 # sf low 305, high: 275

    # Access predictions
    # sj_pred = load(f'joblib_files/lstm/lstm_hypothetical_inputs/sj_{tmax}_{tmin}.joblib')
    # sf_pred = load(f'joblib_files/lstm/lstm_hypothetical_inputs/sf_{tmax}_{tmin}.joblib')

    sj_pred_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/lstm/lstm_hypothetical_inputs/sj_{tmax}_{tmin}.joblib"
    sf_pred_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/lstm/lstm_hypothetical_inputs/sf_{tmax}_{tmin}.joblib"

    sj_pred = load_joblib_from_github(sj_pred_url)
    sf_pred = load_joblib_from_github(sf_pred_url)


    sj_value = sj_pred.item() if hasattr(sj_pred, 'item') else sj_pred
    sf_value = sf_pred.item() if hasattr(sf_pred, 'item') else sf_pred

    # Format output
    sj_output = f'{math.floor(sj_value)} kWh'
    sf_output = f'{math.floor(sf_value)} kWh'

    bay_area = None
    with open('assets/geojson_bay_area.json', 'r') as resp:
        bay_area = json.load(resp)

    county_ids = ['06085', '06075']
    vals = [int(sj_value), int(sf_value)]

    df = pd.DataFrame(data = (county_ids, vals))
    
    # Change sj
    df.iloc[1, 0] = df.iloc[1, 0] - sj_temp_at_low

    # Change sf
    df.iloc[1, 1] = df.iloc[1, 1] - sf_temp_at_low


    vals = [int(df.iloc[1, 0]), int(df.iloc[1, 1])]

    fig = px.choropleth_mapbox(
        df,
        geojson = bay_area,
        locations = county_ids,
        color = vals,
        featureidkey = 'id',
        color_continuous_scale = 'Sunsetdark',
        range_color = [-30, 30],
        zoom = 7.9,
        center = {
            'lat': 37.397574,
            'lon': -121.808050
        },
        opacity = 0.5,
        labels = {
            'color': 'Energy Difference'
        }
    )

    fig.add_annotation(
        dict(
            font = dict(
                color = 'white',
                size = 12
            ),
            x = 0,
            y = -0.05,
            showarrow = False,
            text = 'Regional color reflects the difference from energy usage at the baseline temperatures - Max: 60°F, Min: 0°F',
            textangle = 0,
            xanchor = 'left',
            xref = 'paper',
            yref = 'paper'
        ),
    )

    fig.update_layout(
        paper_bgcolor = 'black',
        legend = dict(
            font = dict(
                color = 'white',
                size = 12
            ),
        ),
        width = 1020,
        height = 600,
        margin = {
            'r': 0,
            't': 0,
            'l': 0,
            'b': 30
        },
        mapbox_accesstoken = token,
    )
    
    return fig, sf_output, sj_output