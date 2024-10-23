import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

from  utils.data_pipeline import (
    processing_pipeline,
    calc_shap, calc_lstm,
    lstm_predict
)
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
                    width = 10,
                    align = 'center',
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
                    width = 10,
                    align = 'center'
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4(
                            'San Jose Features',
                        ),
                    ],
                    width = 6,
                    align = 'center',
                ),
                dbc.Col(
                    [
                        html.H4(
                            'San Francisco Features',
                        ),
                    ],
                    width = 6,
                    align = 'center',
                ),
            ],
        ),
        # dcc.Interval(
        #     id="sj_shap_interval", 
        #     n_intervals=0, 
        #     max_intervals=0,
        #     interval=1
        # ),
        # dcc.Interval(
        #     id="sf_shap_interval", 
        #     n_intervals=0, 
        #     max_intervals=0,
        #     interval=1
        # ),

        # dbc.Row(
        #     [
        #         dbc.Col(
        #             [
        #                 dcc.Graph(
        #                         id = 'sj_shap',
        #                         figure = {},
        #                 ),
        #             ],
        #             width = 6,
        #         ),
        #         dbc.Col(
        #             [
        #                 dcc.Graph(
        #                         id = 'sf_shap',
        #                         figure = {},
        #                 ),
        #             ],
        #             width = 6,
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
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2('LSTM Results'),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Dropdown to select the region
                        dcc.Dropdown(
                            id='region-select',
                            options=[
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            value='sj',
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
                        # Div to display LSTM plot
                        dcc.Graph(id='lstm-plot'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Div to display LSTM scores
                        html.Div(id='lstm-scores', style={'marginTop': 20}),
                    ],
                ),
            ],
            className = 'mb-5',
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2('LSTM Predictions'),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Dropdown to select the region
                        dcc.Dropdown(
                            id='region-dropdown',
                            options=[
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            value='sj',
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
                        html.Button('Predict Future', id='predict-button'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(id='future-prediction-graph'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                       html.Div(id='prediction-output')
                    ],
                ),
            ],
        ),
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

# # SJ SHAP Callback
# @callback(
#     Output('sj_shap', 'figure'),
#     [Input('sj_shap_interval', 'n_intervals')],
# )
# def update_sj_shap(n_intervals):
#     shap_plot = calc_shap('sj')
#     return shap_plot

# # SF SHAP Callback
# @callback(
#     Output('sf_shap', 'figure'),
#     [Input('sf_shap_interval', 'n_intervals')],
# )
# def update_sj_shap(n_intervals):
#     shap_plot = calc_shap('sf')
    
#     return shap_plot

# LSTM Analysis callback
@callback(
    [Output('lstm-scores', 'children'),
     Output('lstm-plot', 'figure')],
    [Input('region-select', 'value')]
)
def update_lstm_analysis(region):
    request_new_pickle = False
    specifier = 1

    plot_title = region
    if region == 'sj':
        plot_title = 'San Jose'
    else:
        plot_title = 'San Francisco'

    lstm_scores = None
    actual_data = None
    lstm_predictions = None

    if request_new_pickle:
        # Call the LSTM calculation function
        lstm_scores, actual_data, lstm_predictions = calc_lstm(region, request_new_pickle, specifier)
    else:
        pickle_filename = f'pickle_files/{region}_lstm_results_{specifier}.pkl'
        if os.path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as f:
                res = pickle.load(f)
                lstm_scores = res['scores']
                actual_data = res['actual_data']
                lstm_predictions = res['predictions']

    # Display LSTM scores
    scores_report = [
        html.H4("LSTM Scores"),
        html.P(f"MAE: {lstm_scores['mae']}"),
        html.P(f"MSE: {lstm_scores['mse']}"),
    ]

    # Create the plot
    fig = go.Figure()
    
    # Plot actual data
    fig.add_trace(go.Scatter(
        x=actual_data['time'], y=actual_data['values'],
        mode='lines', name='Actual'
    ))
    
    # Plot LSTM predictions
    fig.add_trace(go.Scatter(
        x=lstm_predictions['time'], y=lstm_predictions['values'],
        mode='lines', name='Predicted'
    ))
    
    # Update layout of the plot
    fig.update_layout(
        title=f"LSTM Predictions for {plot_title}",
        xaxis_title="Index?",
        yaxis_title="Energy Consumption",
        template="plotly_white"
    )
    
    return scores_report, fig

# Work in progress
# LSTM Future Predictions callback
@callback(
    Output('future-prediction-graph', 'figure'),
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('region-dropdown', 'value')
)
def update_future_prediction(n_clicks, region):
    fig = go.Figure()
    return fig, []

    # skip for now
    if n_clicks is None:
        return dash.no_update

    # Load  model
    with open('pickle_files/lstm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load data
    X_test = None
    y_test = None

    pickle_filename_X_test = f'pickle_files/{region}_X_test_step.pkl'
    pickle_filename_y_test = f'pickle_files/{region}_y_test.pkl'

    with open(pickle_filename_X_test, 'rb') as f:
        X_test = pickle.load(f)
    with open(pickle_filename_y_test, 'rb') as f:
        y_test = pickle.load(f)

    # Prepare the last known input data point
    last_known_data = X_test[-1][-5:]  # Last 10 sequences from X_test
    
    last_known_data = last_known_data.reshape((1, last_known_data.shape[0], last_known_data.shape[1]))  # Now (1, 5, 1)

    # Make future predictions
    future_steps = 2  # Predicting 2 months ahead
    predictions = lstm_predict(model, last_known_data, future_steps)

    # Prepare the plot
    future_dates = pd.date_range(start='2024-07', periods=future_steps, freq='ME')
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name='Predicted Future'))

    # Return the updated figure and output text
    return fig, f"Predicted Values: {predictions}"