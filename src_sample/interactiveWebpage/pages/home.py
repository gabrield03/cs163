from dash import html, dcc, Input, Output, callback, clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd

from  utils.data_pipeline import processing_pipeline

import os
from joblib import load


### Load Data ###
sj_df = pd.DataFrame()
sf_df = pd.DataFrame()

if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
    sj_df = load('joblib_files/base_data/sj_combined.joblib')

if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
    sf_df = load('joblib_files/base_data/sf_combined.joblib')
    
# Define the layout for the home page
home_front = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Weather Impact On Energy Usage',
                        style = {
                            'color': 'white',
                            'textAlign': 'center',
                            'fontSize': '4vw',
                            'fontweight': 'bold',
                            'fontFamily': 'roboto',
                            'font-variant': 'small-caps',
                            'height': '100%',
                            'text-shadow': '2px 2px 4px #000000',
                        },
                    ),
                    width = 12,
                ),
            ],
            className = 'mt-5 p-5',
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div("Energy Consumption", id="word-energy-consumption", className="fade-word"),
                            html.Div("Max Temperature", id="word-max-temp", className="fade-word"),
                            html.Div("Min Temperature", id="word-min-temp", className="fade-word"),
                            html.Div("Precipitation", id="word-precipitation", className="fade-word"),
                            html.Div("Wind Speed", id="word-wind-speed", className="fade-word"),
                        ],
                        style = {
                            'color': 'white',
                            'textAlign': 'center',
                            'fontSize': '3vw',
                            'fontWeight': 'bold',
                            'fontFamily': 'monospace',
                            'height': '100%',
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                        },
                    ),
                    width=12,
                ),
            ],
        ),
                
        # Video background
        html.Div(
            children = [
                html.Video(
                    id = 'home-front-video',
                    src = '/assets/sky-video2.mp4',
                    autoPlay = True,
                    loop = False,
                    muted = True,
                    controls = False,
                    style = {
                        'width': '100vw',
                        'maxWidth': '100vw',
                        'height': '75vh',
                        'objectFit': 'cover',
                        'position': 'fixed',
                        'top': '0',
                        'left': '0',
                        'zIndex': '-1',
                        'margin': 0,
                    },
                ),
            ],
        ),
        # Fade to black bottom section
        html.Div(
            id = 'home-front-gradient',
            style = {
                'height': '35vh',
                'background': 'linear-gradient(to bottom, rgba(0,0,0,0), rgba(0,0,0,.3), rgba(0,0,0,.6), rgba(0,0,0,0.9), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1))',
                'width': '100vw',
                'position': 'fixed',
                'bottom': '0',
                'left': '0',
                'zIndex': '-1',
            },
        ),

        dcc.Interval(id="page-load-interval", interval=3000, n_intervals=0),
    ],
    style={
        'height': '100vh',
        #'position': 'relative',
    },
)



# Random forest for feature importances
feature_importances_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                'What Influences Energy Usage the Most?',
                            ],
                            style = {
                                'color': 'white',
                                'font-size': '22px',
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
                dbc.Col(
                    [

                    ],
                    width = 2
                ),
                dbc.Col(
                    [
                    ],
                    width = 5,
                ),
            ],
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
                    align = 'center',
                ),
                dbc.Col(
                    [
                    ],
                    width = 5
                ),
                dbc.Col(
                    [
                    ],
                    width = 2,
                ),
            ],
            className = 'mb-3',
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
                    align = 'center',
                ),
                dbc.Col(
                    [
                    ],
                    width = 2
                ),
                dbc.Col(
                    [
                    ],
                    width = 5,
                ),
            ],
        ),
    ],
    className = 'mb-5',
)


# SHAP Dot plot function - unused bc pythonanywhere cant make parallel plots
def shap_parallel_coord_plot(loc):
    base_df = load(f'joblib_files/base_data/{loc}_combined.joblib')

    plot_title = 'San Jose' if loc == 'sj' else 'San Francisco'

    # Which columns for sj_df? All or only ones matching sf_df?
    dim_cols = [
        'zipcode', 'year', 'totalcustomers', 'totalkwh', 'averagekwh',
        'month-numeric', 'prcp', 'tmax', 'tmin'
    ]
    
    fig = px.parallel_coordinates(
        base_df,
        color = 'averagekwh',
        # dimensions = base_df.columns,
        dimensions = dim_cols,
        color_continuous_scale=px.colors.diverging.Tealrose,
    )

    fig.update_layout(
        #title = f'Data Distributions for {plot_title}',
        xaxis_title = 'Features',
        yaxis_title = 'Values',
    )

    return fig

# SHAP Dot Plots
shap_parallel_coord_plot_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'San Jose Data Relationship',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
                                'color': '#ffffff',
                            }
                        )
                    ],
                    width = 5,
                ),
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        html.P(
                            'San Francisco Data Relationship',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
                                'color': '#ffffff',
                            }
                        )
                    ],
                    width = 5,
                ),
            ],
            className = 'mb-1',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Img(
                            src="assets/shap_plots/sj_parallel_coord_plot.png",
                            alt="SHAP Parallel Coord Plot for SJ",
                            style={"width": "100%", "height": "auto"}
                        )
                        # dcc.Graph(
                        #     id = 'shap_parallel_coord_plot',
                        #     figure = shap_parallel_coord_plot('sj'),
                        # ),
                    ],
                    width = 5,
                    #align = 'center',
                ),
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        html.Img(
                            src="assets/shap_plots/sf_parallel_coord_plot.png",
                            alt="SHAP Parallel Coord Plot for SF",
                            style={"width": "100%", "height": "auto"}
                        )
                        # dcc.Graph(
                        #     id = 'shap_parallel_coord_plot',
                        #     figure = shap_parallel_coord_plot('sf'),
                        # ),
                    ],
                    width = 5,
                    #align = 'center',
                ),
            ],
            className = 'mb-5',
        ),
    ],
)



layout = dbc.Container(
    [
        home_front,
        feature_importances_section,
        shap_parallel_coord_plot_section,
        html.Div(
            id = 'home-page-content',
        ),
    ],
    # fluid = True,
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

    importances_fn = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    if not os.path.exists(importances_fn):
        importances_df = processing_pipeline(df, loc)
    else:
        importances_df = load(importances_fn)

    rename_cols = {
        'season': 'Season', 'month': 'Month', 'year': 'Year',
        'tmax': 'Max (°F)', 'tmin': 'Min (°F)', 
        'prcp': 'Precip', 'totalcustomers': 'Customers'
    }

    importances_df['feature'] = importances_df['feature'].replace(rename_cols)
    importances_df.sort_values(by = ['importances'], ascending = True, inplace = True)
    importances_df['color_group'] = ['Top' if i > 3 else 'Other' for i in range(len(importances_df))]
    
    fig = px.bar(
        importances_df,
        y = 'feature',
        x = 'importances',
        color = 'color_group',
        color_discrete_map = {
            'Top': 'green',
            'Other': 'gray',
        },
        range_x = [0, 0.48],
    )

    fig.update_layout(
        xaxis_title = None, # Importances
        yaxis_title = None, # Features
        xaxis_showticklabels = False,
        plot_bgcolor = 'rgba(0, 0, 0, 0)',
        paper_bgcolor = 'rgba(0, 0, 0, 0)',
        showlegend = False,
        xaxis = dict(showgrid = True, gridcolor = 'rgba(0,0,0,0)'),
        yaxis = dict(color = 'white'),
        margin = dict(l = 0, r = 0),
    )

    # Display values to the right of the top feature bars
    fig.update_traces(
        text = importances_df['importances'].round(3),
        textposition = 'outside',
        insidetextanchor = 'start',
        textfont = dict(color = 'white')
    )

    return [fig]