import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from  utils.data_pipeline import processing_pipeline
import os
import pickle

### Load Data ###
sj_df = pd.DataFrame()
sf_df = pd.DataFrame()

if os.path.exists('pickle_files/sj_combined.pkl'):
    with open('pickle_files/sj_combined.pkl', 'rb') as f:
        sj_df = pickle.load(f)

if os.path.exists('pickle_files/sf_combined.pkl'):
    with open('pickle_files/sf_combined.pkl', 'rb') as f:
        sf_df = pickle.load(f)

analytics_header = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        'Analytics',
                    ),
                    className = 'text-center mb-5',
                    width = 12,
                    style = {'height': '100%'},
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# Description of the analysis conducted
analytics_info = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'The two core advanced analysis methods that this project will implement are ',
                            'Feature Analysis and Time-Series Analysis',
                        ],
                        style = {
                            'font-size': '20px',
                            'word-break': 'keep-all',
                        },
                    )
                    # html.P(
                    #     [
                    #         'The analytic techniques and algorithms conducted for this project include a '
                    #         'Random Forest regressor to identify critical weather variables influencing '
                    #         'energy demand and a Long Short-Term Memory (LSTM) model for forecasting future '
                    #         'energy consumption patterns based on historical data.',
                    #     ],
                    #     style = {
                    #         'font-size': '20px',
                    #         'word-break': 'keep-all',
                    #     },
                    # ),
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'Feature Analysis',
                        ],
                    ),
                    style = {
                        'font-size': '20px',
                        'word-break': 'keep-all',
                    },
                    width = 3,
                ),
                dbc.Col(
                    html.P(
                        [
                            'This analysis will focus on understanding the influence of historical weather data on ',
                            'energy consumption in San Francisco and San Jose. The goal is to identify which weather ',
                            'variables (e.g., temperature, precipitation, wind speed) have a significant impact on ',
                            'energy consumption and whether these impacts differ between the two regions. This will help ',
                            'reveal any disproportionate effects of weather conditions on energy demand in each area.',
                        ],
                        style = {
                        'font-size': '16px',
                        'word-break': 'keep-all',
                    },
                    ),
                    width = 9,
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'Time-Series Analysis',
                        ],
                        style = {
                            'font-size': '20px',
                            'word-break': 'keep-all',
                        },
                    ),
                    width = 3,
                ),
                dbc.Col(
                    html.P(
                        [
                            'To forecast future energy consumption, I will use time-series models such as Long-Short ',
                            'Term Memory (LSTM) networks and Autoregressive Integrated Moving Average (ARIMA). The aim ',
                            'is to create accurate predictions based on historical weather data, which will help energy ',
                            'providers and the public prepare for fluctuations in energy demand.',
                        ],
                        style = {
                            'font-size': '16px',
                            'word-break': 'keep-all',
                        },
                    ),
                    width = 9,
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# Random forest for feature importances
analytics_objective_1_1 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Feature Analysis with Random Forest to Identify Key Weather Variables'
                        ),
                        html.P(
                            [
                                'Random Forest regressors are ensemble methods that train multiple decision trees and ',
                                'aggregate their results. By combining several base estimators within a given learning ',
                                'algorithm, Random Forests improve the generalizability of a single estimator. ',
                                'This method is particularly suitable for assessing the impact of climate variables ',
                                '(such as maximum and minimum temperature, precipitation, wind speed, etc.) as it ',
                                'captures non-linear relationships, enabling it to model the complex interactions ',
                                'inherent in weather patterns.',
                                html.Br(), html.Br(),

                                'To identify significant weather variables affecting energy consumption, I trained ',
                                'separate models for each region (San Francisco and San Jose). Each model calculates and ranks ',
                                'the feature importances of weather variables. ',

                                # put the following at the bottom after the feature importance methods have been applied
                                #'After determining the feature importances ',
                                #'for each region, I compare them to assess whether specific weather variables are more ',
                                #'influential in one region than the other.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
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
                                {'label': 'San Jose Dataset', 'value': 'sj_df'},
                                {'label': 'San Francisco Dataset', 'value': 'sf_df'},
                            ],
                            multi = False,
                            value = 'sj_df',
                            style = {
                                'backgroundColor': '#bdc3c7',
                                'color': '#2c3e50'
                            }, 
                        ),
                    ],
                    width = 7,
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'sj_feature_importances',
                            figure = {},
                        ),
                    ],
                    width = 7,
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# SHAP - explaining features
analytics_objective_1_2 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Feature Interpretability with SHapley Additive exPlanations (SHAP)'
                        ),
                        html.P(
                            [
                                'To complement the Random Forest, I am using SHAP to provide further explanation',
                                'of each variable in the model. SHAP values assign each feature an importance score ',
                                'based on its contribution to the prediction, offering a clear way to compare which ',
                                'weather variables are driving energy consumption in each region.',
                                html.Br(), html.Br(),

                                '[explanation of plot that will be below]'
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Dropdown(
        #                     id = 'region_option',
        #                     options = [
        #                         {'label': 'San Jose Dataset', 'value': 'sj_df'},
        #                         {'label': 'San Francisco Dataset', 'value': 'sf_df'},
        #                     ],
        #                     multi = False,
        #                     value = 'sj_df',
        #                     style = {
        #                         'backgroundColor': '#bdc3c7',
        #                         'color': '#2c3e50'
        #                     }, 
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        #     className = 'mb-3',
        # ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(
        #                     id = 'sj_feature_importances',
        #                     figure = {},
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        # ),
    ],
    className = 'mb-5',
)

# PDP - additional feature interpretability -> looking to change this to permutation models
analytics_objective_1_3 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Partial Dependence Plots (PDP)'
                        ),
                        html.P(
                            [
                                'PDPs may also be incorporated to visualize how changes in individual weather ',
                                'variables impact energy consumption, helping to clarify relationships that SHAP ',
                                'scores identify.',
                                html.Br(), html.Br(),

                                '[explanation of plot that will be below]'
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Dropdown(
        #                     id = 'region_option',
        #                     options = [
        #                         {'label': 'San Jose Dataset', 'value': 'sj_df'},
        #                         {'label': 'San Francisco Dataset', 'value': 'sf_df'},
        #                     ],
        #                     multi = False,
        #                     value = 'sj_df',
        #                     style = {
        #                         'backgroundColor': '#bdc3c7',
        #                         'color': '#2c3e50'
        #                     }, 
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        #     className = 'mb-3',
        # ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(
        #                     id = 'sj_feature_importances',
        #                     figure = {},
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        # ),
    ],
    className = 'mb-5',
)

# Regional Comparisons of feature importances
analytics_objective_1_4 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Comparing Feature Importances Between Regions'
                        ),
                        html.P(
                            [
                                'Two models—one for San Francisco and one for San Jose—will be trained. ',
                                'By comparing feature importance between the models, I will analyze whether ',
                                'specific weather variables have a greater impact in one region than in the other. ',
                                'The preliminary focus will be on variables like maximum and minimum temperatures, ',
                                'precipitation, and seasonality.',
                                html.Br(), html.Br(),

                                '[explanation of plot that will be below]'
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Dropdown(
        #                     id = 'region_option',
        #                     options = [
        #                         {'label': 'San Jose Dataset', 'value': 'sj_df'},
        #                         {'label': 'San Francisco Dataset', 'value': 'sf_df'},
        #                     ],
        #                     multi = False,
        #                     value = 'sj_df',
        #                     style = {
        #                         'backgroundColor': '#bdc3c7',
        #                         'color': '#2c3e50'
        #                     }, 
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        #     className = 'mb-3',
        # ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(
        #                     id = 'sj_feature_importances',
        #                     figure = {},
        #                 ),
        #             ],
        #             width = 7,
        #         ),
        #     ],
        # ),
    ],
    className = 'mb-5',
)

# Time-series analysis with LSTM
analytics_objective_2_1 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Time-Series Analysis to Forecast Energy Demand with LSTM and SARIMA'
                        ),
                        html.P(
                            [
                                'LSTM networks are designed to handle time-series data by retaining information over time, ',
                                'making them effective for capturing long-term dependencies in weather and energy usage data. ',
                                'I will use LSTM to predict future energy consumption based on historical weather data. The ',
                                'model will initially produce monthly energy predictions, with the possibility of transitioning ',
                                'to daily forecasts if daily energy records become available or if the current data (daily ',
                                'weather records and monthly energy data) is proves to be sufficient in predicting daily energy consumption.',
                                html.Br(), html.Br(),

                                'I use historical energy consumption data (average monthly energy usage) along with weather ',
                                'data as inputs for this model. Currently, the output of the model are predictions of future ',
                                'energy usage on a monthly basis. I also aim to implement daily predictions; however, due to ',
                                'the current format of the energy data (monthly records), the model may not generalize well. ',
                                'If I can obtain daily records, the focus will shift toward daily predictions.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(

        #                 ),
        #             ],
        #         ),
        #     ],
        # ),
    ],
    className = 'mb-5',
)

# Time-Series Analysis with SARIMA
analytics_objective_2_2 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Autoregressive Integrated Moving Average (ARIMA) [ETS?]'
                        ),
                        html.P(
                            [
                                'Seasonal ARIMA will be used to test time-series forecasting when dealing ',
                                'with smaller datasets or where simpler modeling is appropriate. ARIMA works ',
                                'by creating a linear equation that describes and forecasts the time-series ',
                                'data. It can provide some baseline predictions for energy consumption.',
                                html.Br(), html.Br(),

                                '[explanation of plot below]',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(

        #                 ),
        #             ],
        #         ),
        #     ],
        # ),
    ],
    className = 'mb-5',
)

layout = dbc.Container(
    [
        analytics_header,
        analytics_info,
        analytics_objective_1_1,
        analytics_objective_1_2,
        analytics_objective_1_3,
        analytics_objective_1_4,
        analytics_objective_2_1,
        analytics_objective_2_2,
    ],
    fluid = True,
)

# Callback for Feature Importances
@callback(
    [
        Output('sj_feature_importances', 'figure')
    ],
    [
        Input('region_option', 'value'),
    ]
)
# Animated plot function
def update_sj_feature_importances(selected_region):

    df = pd.DataFrame()
    if selected_region == 'sj_df':
        df = sj_df
    elif selected_region == 'sf_df':
        df = sf_df

    feature_importances = processing_pipeline(df)
    
    # Need to add title and expand the actual axis
    fig = px.scatter(
        feature_importances,
        x = 'feature',
        y = 'importances',
        #title = 'Feature Importances',
    )

    fig.update_layout(
        xaxis_title = 'Features',
        yaxis_title = 'Importances',
        xaxis_tickangle = -45,
    )

    return [fig]