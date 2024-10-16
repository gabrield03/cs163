import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from  utils.data_pipeline import processing_pipeline
import os
import pickle

### Load Data ###
sj_df = ''
sf_df = ''
if os.path.exists('sj_combined.pkl'):
    with open('sj_combined.pkl', 'rb') as f:
        sj_df = pickle.load(f)

if os.path.exists('sf_combined.pkl'):
    with open('sf_combined.pkl', 'rb') as f:
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

analytics_info = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'The analytic techniques and algorithms conducted for this project include a '
                            'Random Forest regressor to identify critical weather variables influencing '
                            'energy demand and a Long Short-Term Memory (LSTM) model for forecasting future '
                            'energy consumption patterns based on historical data.',
                        ],
                        style = {
                            'font-size': '20px',
                            'word-break': 'keep-all',
                        },
                    ),
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

analytics_objective_1 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Random Forest for Identifying Weather Variables Impacting Energy Usage'
                        ),
                        html.P(
                            [
                                'Random Forest regressors are ensemble methods that train multiple decision trees ',
                                'and aggregate their results. By combining several base estimators within a given ',
                                'learning algorithm, Random Forests improve the generalizability of a single estimator. ',
                                'This method is particularly suitable for assessing the impact of climate variables ',
                                '(such as maximum and minimum temperature, precipitation, wind speed, etc.) as it ',
                                'captures non-linear relationships, enabling it to model the complex interactions ',
                                'inherent in weather patterns.',
                                html.Br(), html.Br(),

                                'To identify significant weather variables affecting energy consumption, I trained ',
                                'separate models for each region (San Francisco and San Jose). Each model calculates and ranks ',
                                'the feature importances of weather variables. After determining the feature importances ',
                                'for each region, I compare them to assess whether specific weather variables are more ',
                                'influential in one region than the other.',
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

analytics_objective_2 = html.Div(
    [

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(
                            'Long-Short Term Memory (LSTM) to Predict Future Energy Demand'
                        ),
                        html.P(
                            [
                                'LSTMs are a type of Recurrent Neural Network (RNN) designed to remember information over ',
                                'time and apply that information to future calculations. They excel at capturing long-term ',
                                'dependencies in time-series data, making them ideal for predicting future energy usage by ',
                                'recognizing patterns in historical weather and seasonal data.',
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

layout = dbc.Container(
    [
        analytics_header,
        analytics_info,
        analytics_objective_1,
        analytics_objective_2,
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