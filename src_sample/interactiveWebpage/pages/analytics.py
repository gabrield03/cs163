import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.express as px
import plotly.graph_objs as go

from  utils.data_pipeline import (
    processing_pipeline,
    calc_shap,
    lstm_predict,
    pred_lstm_single_step, pred_lstm_multi_step,
    pred_sarima
)
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

analytics_header_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Analytics',
                    ),
                    className = 'text-center mb-5',
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
    className = 'mb-1',
)

# Description of the analysis conducted
analytics_info_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        [
                            'This section focuses on two of the following advanced analysis methods:',
                        ],
                        style = {
                            'font-size': '25px',
                            'word-break': 'keep-all',
                        },
                    )
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col([], width = 2),
                dbc.Col(
                    [
                        dbc.Button(
                            html.Span(
                                'Feature Analysis',
                                id = 'feature_analysis_tooltip',
                            ),
                            color = 'secondary',
                            id = 'feature_analysis_button',
                            className = 'me-1 small-button',
                            n_clicks = 0,
                            style = {
                                'font-size': '16px',
                                'background-color': '#bdc3c7',
                            },
                        ),
                    ],
                    width = 3,
                ),
                dbc.Col([], width = 3),
                dbc.Col(
                    [
                        dbc.Button(
                            html.Span(
                                'Time-Series Analysis',
                                id = 'time_series_analysis_tooltip',
                            ),
                            color = 'secondary',
                            id = 'time_series_analysis_button',
                            className = 'me-1 small-button',
                            n_clicks = 0,
                            style = {
                                'font-size': '16px',
                                'background-color': '#bdc3c7',
                            },
                        ),
                    ],
                    width = 3,
                ),
                dbc.Tooltip(
                    'click me!',
                    target = 'feature_analysis_tooltip',
                    placement = 'top',
                    style = {
                        'font-size': '14px',
                        'color': '#333333',
                    },
                ),
                dbc.Tooltip(
                    'click me!',
                    target = 'time_series_analysis_tooltip',
                    placement = 'top',
                    style = {
                        'font-size': '14px',
                        'color': '#333333',
                    },
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Collapse(
                            dbc.Card(
                                html.P(
                                    [
                                        'This analysis will focus on understanding the influence of historical weather data on ',
                                        'energy consumption in San Francisco and San Jose. The goal is to identify which weather ',
                                        'variables (e.g., temperature, precipitation, wind speed) have a significant impact on ',
                                        'energy consumption and whether these impacts differ between the two regions. This will help ',
                                        'reveal any disproportionate effects of weather conditions on energy demand in each area.',
                                    ],
                                    style = {
                                        'font-size': '20px',
                                        'font-style': 'italic',
                                        'color': '#444444',
                                    },
                                ),
                                body = True,
                                style = {
                                    'background-color': '#d4d7da',
                                },
                            ),
                            id = 'feature_analysis_collapse',
                            is_open = False,
                        ),
                    ],
                ),
                dbc.Col(
                    [
                        dbc.Collapse(
                            dbc.Card(
                                html.P(
                                    [
                                        'To forecast future energy consumption, I will use time-series models such as Long-Short ',
                                        'Term Memory (LSTM) networks and Autoregressive Integrated Moving Average (ARIMA). The aim ',
                                        'is to create accurate predictions based on historical weather data, which will help energy ',
                                        'providers and the public prepare for fluctuations in energy demand.',
                                    ],
                                    style = {
                                        'font-size': '20px',
                                        'font-style': 'italic',
                                        'color': '#444444',
                                    },
                                ),
                                body = True,
                                style = {
                                    'background-color': '#d4d7da',
                                },
                            ),
                            id = 'time_series_analysis_collapse',
                            is_open = False,
                        ),
                    ],
                ),
            ],
            className = 'mt-3',
        ),
    ],
    className = 'mb-5',
)

# Random forest for feature importances
feature_importances_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Feature Analysis with Random Forest to Identify Key Weather Variables',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000'
                            },
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
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'To identify significant weather variables affecting energy consumption, I trained ',
                                'separate models for each region (San Francisco and San Jose). Each model calculates and ranks ',
                                'the feature importances of weather variables.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
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
                                {'label': 'San Jose Dataset', 'value': 'sj'},
                                {'label': 'San Francisco Dataset', 'value': 'sf'},
                            ],
                            multi = False,
                            value = 'sj',
                            style = {
                                'backgroundColor': '#bdc3c7',
                                'color': '#2c3e50',
                            }, 
                        ),
                    ],
                    width = 4,
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
                            id = 'sj_feature_importances_section',
                        ),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# SHAP - explaining features
shap_intro_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Feature Interpretability with SHapley Additive exPlanations (SHAP)',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
                        html.P(
                            [
                                'To complement the Random Forest, I am using SHAP to provide further explanation ',
                                'of each variable in the model. SHAP values assign each feature an importance score ',
                                'based on its contribution to the prediction, offering a clear way to compare which ',
                                'weather variables are driving energy consumption in each region.',
                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'Below displays the mean SHAP values for each region. The size of the bubbles are ',
                                'indicative of the SHAP value. Several features from the San Jose data have been ',
                                'omitted so that each region has the their SHAP values calculated based on the same features.',
                                
                                html.Br(), html.Br(),
                                'The most important feature for the ',
                                html.Span(
                                    'San Jose region is season',
                                    id = 'season_tooltip',
                                    style = {
                                        'font-weight': 'bold',
                                        'font-style': 'normal',
                                        'color': 'red',
                                    },
                                ),
                                ' with a SHAP value of 20.88. Season is twice as important to the overall model predictions ',
                                'than the second and third most important features (tmax and tmin). ',

                                'On the other hand, the most important feature for the  ',
                                html.Span(
                                    'San Francisco region is tmax',
                                    id = 'tmax_tooltip',
                                    style = {
                                        'font-weight': 'bold',
                                        'font-style': 'normal',
                                        'color': 'red',
                                    },
                                ),
                                ' with a SHAP value of 14.5. The next two most important features (tmin and totalcustomers). ',
                                'The top two features (tmax and tmin) have much closer scores than in the San Jose region. ',
                                'Here, totalcustomers  represents the number of residents serviced in the area (the region\'s zipcode). ',
                                
                                html.Br(), html.Br(),
                                'These results seem to indicate that the weather impacts each region differently - namely ',
                                'that energy consumption in the San Jose region is more affected by season-specific weather ',
                                'than San Francisco. We can see that the overall (maximum and minimum) temperatures ',
                                'in San Francisco play a more important role than the time of the year (the season).'
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                            },
                        ),
                        dbc.Tooltip(
                            'SJ season - 20.88 SHAP value',
                            target = 'season_tooltip',
                            placement = 'top',
                            style = {
                                'font-size': '14px',
                                'color': '#333333',
                            },
                        ),
                        dbc.Tooltip(
                            'SF tmax - 14.5 SHAP value',
                            target = 'tmax_tooltip',
                            placement = 'top',
                            style = {
                                'font-size': '14px',
                                'color': '#333333',
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# SHAP Dot plot function
def shap_dot_plot():
    sj_shap = load('joblib_files/shap/sj_shap_plot.joblib')
    sf_shap = load('joblib_files/shap/sf_shap_plot.joblib')

    # Combine into one DataFrame for scatter plot
    dot_data = pd.concat(
        [
            sj_shap.assign(region = 'San Jose'),
            sf_shap.assign(region = 'San Francisco')
        ]
    )

    fig = px.scatter(
        dot_data,
        x = 'Feature',
        y = 'region',
        size = 'Mean SHAP Value',
        color = 'Mean SHAP Value',
        color_continuous_scale = 'Agsunset',
        title = 'Mean SHAP Value for Both Regions',
        labels = {
            'Mean SHAP Value': 'Mean SHAP Value',
            'Feature': 'Feature'
        },
        size_max = 60,
    )

    fig.update_layout(yaxis_title = 'Region', xaxis_title = 'Feature')

    return fig

# SHAP Dot Plots
shap_dot_plot_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'shap_dot_plot',
                            figure = shap_dot_plot(),
                        ),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
            className = 'mb-5',
        ),
    ],
)

# SHAP Decision Plots
shap_decision_plot_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'SHAP Decision Plots',
                            style = {
                                'text-align': 'center',
                                'font-size': '30px',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 12,
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'San Jose',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
                            },
                        ),
                    ],
                    width = 6,
                ),
                dbc.Col(
                    [
                        html.P(
                            'San Francisco',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
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
                        html.Img(
                            src = 'assets/shap_plots/sj_shap.png',
                            style = {
                                'width': '100%',
                                'height': 'auto',
                            },
                        ),
                    ],
                ),
                dbc.Col(
                    [
                        html.Img(
                            src = 'assets/shap_plots/sf_shap.png',
                            style = {
                                'width': '100%',
                                'height': 'auto',
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
    className = 'mb-5',
)

# PDP Section
pdp_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Partial Dependence Plots (PDP)',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
                        html.P(
                            [
                                'PDPs are global model-agnostic methods that show the marginal effect ',
                                'one or two features have on the predicted outcome of a model. ',
                                'Agnostic refers to universiality - PDP can be applied to any model (interpretable model or black box model). ',
                                'Essentially, a PDP plot visualizes the average effect of a feature\'s values by ',
                                'marginalizing all the other features in a dataset.',

                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'The top three more impactful features from each region (calculated by SHAP) are ',
                                'displayed in the plots below.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                            },
                        ),
                    ],
                ),
            ],
        ),
        # Partial Dependency Plots
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'PDP Plots',
                            style = {
                                'text-align': 'center',
                                'font-size': '30px',
                                'font-variant': 'small-caps',
                            }
                        )
                    ],
                    width = 12,
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'San Jose',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
                            }
                        )
                    ],
                    width = 6,
                ),
                dbc.Col(
                    [
                        html.P(
                            'San Francisco',
                            style = {
                                'text-align': 'center',
                                'font-size': '25px',
                                'font-variant': 'small-caps',
                            }
                        )
                    ],
                    width = 6,
                ),
            ],
            className = 'mb-1',
        ),
        dbc.Row(
            [
                # SJ PDP
                dbc.Col(
                    [
                        dbc.Carousel(
                            items=[
                                {
                                    'key': '1',
                                    'src': '/assets/pdp_plots/sj_pdp_season.png',
                                },
                                {
                                    'key': '2',
                                    'src': '/assets/pdp_plots/sj_pdp_tmax.png',
                                },
                                {
                                    'key': '3',
                                    'src': '/assets/pdp_plots/sj_pdp_tmin.png',
                                },
                            ],
                            controls = True,
                            indicators = True,
                            className = 'carousel-fade',
                        ),
                    ],
                    width = 6,
                ),
                # SF PDP
                dbc.Col(
                    [
                        dbc.Carousel(
                            items=[
                                {
                                    "key": "1",
                                    "src": "/assets/pdp_plots/sf_pdp_tmax.png",
                                },
                                {
                                    "key": "2",
                                    "src": "/assets/pdp_plots/sf_pdp_tmin.png",
                                },
                                {
                                    "key": "3",
                                    "src": "/assets/pdp_plots/sf_pdp_totalcustomers.png",
                                },
                            ],
                            controls = True,
                            indicators = True,
                            className = 'carousel-fade',
                        ),
                    ],
                    width = 6,
                ),   
            ],
        ),
    ],
    className = 'mb-5',
)

# Time-series analysis with LSTM
lstm_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Time-Series Analysis to Forecast Energy Demand with LSTM and SARIMA',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
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
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'I use historical energy consumption data (average monthly energy usage) along with weather ',
                                'data as inputs for this model. Currently, the output of the model are predictions of future ',
                                'energy usage on a monthly basis. I also aim to implement daily predictions; however, due to ',
                                'the current format of the energy data (monthly records), the model may not generalize well. ',
                                'If I can obtain daily records, the focus will shift toward daily predictions.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
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
                        html.P(
                            'LSTM Single-Step Prediction',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
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
                        dcc.Dropdown(
                            id = 'region_select',
                            options = [
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            value = 'sj',
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
                        dcc.Graph(
                            id = 'lstm_plot',
                        ),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id = 'lstm_val_score',
                            style = {
                                'marginTop': 20,
                                'color': 'white',
                            },
                        ),
                        html.Div(
                            id = 'lstm_test_score',
                            style = {
                                'marginTop': 20,
                                'color': 'white',
                            },
                        ),
                    ],
                ),
            ],
            className = 'mb-5',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'LSTM Multi-Step Prediction',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
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
                        dcc.Dropdown(
                            id = 'region_dropdown',
                            options = [
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            value = 'sj',
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
                        dcc.Graph(id='lstm_plot_multi'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id = 'lstm_val_score_multi',
                            style = {
                                'marginTop': 20,
                                'color': 'white',
                            },
                        ),
                        html.Div(
                            id = 'lstm_test_score_multi',
                            style = {
                                'marginTop': 20,
                                'color': 'white',
                            },
                        ),
                    ],
                ),
            ],
            className = 'mb-5',
        ),
    ],
    className = 'mb-5',
)

# Time-Series Analysis with SARIMA
sarima_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Seasonal Autoregressive Integrated Moving Average (SARIMA)',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
                        html.P(
                            [
                                'Seasonal ARIMA will be used to test time-series forecasting when dealing ',
                                'with smaller datasets or where simpler modeling is appropriate. ARIMA works ',
                                'by creating a linear equation that describes and forecasts the time-series ',
                                'data. It can provide some baseline predictions for energy consumption.',
                                
                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'SARIMA models are statistical models that are seasonal extensions of ARIMA ',
                                'models. The model is comprised of traditional ARIMA components (p, d, q) - ',
                                'p: the order of the auto regression, d: the number of times the data was differenced ',
                                'to become stationary, q: the order of the moving average. Seasonal components are ',
                                'accounted for with (P, D, Q, m) which are the same definitions but for seasonality. ',
                                'Also, m: the number of time steps in a seasonal cycle. SARIMA models can work well ',
                                'with small datasets like the ones used for this project. The predictions plotted ',
                                'here (2023-07 to 2024-06) show SARIMA\'s predictions and the actual labeled data. ',
                                'The mean absolute error (mae) is 22.895, meaning the SARIMA model\'s average ',
                                'prediction error for the 12 data points was off by 22 kWh.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
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
                        html.P(
                            'SARIMA Predictions',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
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
                        dcc.Dropdown(
                            id = 'sarima_region_dropdown',
                            options = [
                                {'label': 'San Jose', 'value': 'sj'},
                                {'label': 'San Francisco', 'value': 'sf'},
                            ],
                            value = 'sj',
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
                        dcc.Graph(
                            id = 'sarima_plot'
                        ),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            id = 'sarima_mae',
                            style = {
                                'marginTop': 20,
                                'color': 'white',
                            },
                        ),
                    ],
                ),
            ],
            className = 'mb-5',
        ),
    ],
    className = 'mb-5',
)

#### Summary of Analysis Section ####
analysis_summary_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Summary of Analysis',
                            style = {
                                'text-align': 'center',
                                'font-size': '60px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
                        html.P(
                            [
                                'Our analysis demonstrates that energy consumption patterns in San Jose ',
                                'and San Francisco are influenced by distinct weather-related factors, ', 
                                'revealing how local climate characteristics can lead to differing energy ',
                                'demands between regions. Using a random forest model paired with SHAP ',
                                '(SHapley Additive exPlanations)—a statistical method that breaks down the ',
                                'impact of each feature on model predictions—we quantified the importance ',
                                'of various factors. SHAP is particularly valuable here because it assigns ',
                                '"importance" scores to features based on their average impact on model ',
                                'predictions, enabling a clear assessment of each feature\'s role.',

                                html.Br(), html.Br(),

                                'In San Jose, seasonality emerged as the most significant predictor of energy ',
                                'consumption, with a mean SHAP value of 20, indicating a strong correlation ',
                                'between energy use and the time of year. Temperature variables followed, with ',
                                'maximum (Tmax) and minimum (Tmin) temperatures ranking second and third, ',
                                'suggesting that while temperature plays a role, seasonality\'s impact is ',
                                'notably higher. By contrast, San Francisco\'s energy consumption is most ',
                                'sensitive to temperature extremes. Tmax had the highest mean SHAP value (12), ',
                                'followed by Tmin (10) and the total number of customers (5), with seasonality ',
                                'showing a relatively minor influence.',

                                html.Br(), html.Br(),

                                'To further explore these findings, SHAP decision plots were used to illustrate ',
                                'each feature\'s contribution to specific predictions. Partial Dependence Plots ',
                                '(PDP) then provided insight into how variations in the top three features impact ',
                                'energy consumption predictions for each region.',

                                html.Br(), html.Br(),

                                
                                'These insights suggest a fundamental difference in climate sensitivity between ',
                                'the two regions: San Francisco\'s energy demands are more heavily influenced by ',
                                'shifts in temperature, possibly indicating a higher sensitivity to climate ',
                                'variability. Conversely, San Jose\'s reliance on seasonality hints that while its ',
                                'energy consumption may be less responsive to incremental temperature changes, ',
                                'seasonal cycles play a dominant role in its demand pattern. However, because ',
                                'temperature and seasonality are interdependent [NEED REFERENCE TO BACK THIS CLAIM], it would be naive to conclude ',
                                'that San Jose is less vulnerable to climate changes; further analysis is ',
                                'warranted to determine the relationship between seasonality and global temperature ',
                                'shifts. This study suggests that regional energy planning could benefit from ',
                                'tailored approaches that account for these differing sensitivities.',
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
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
        analytics_header_section,
        analytics_info_section,
        feature_importances_section,
        shap_intro_section,
        shap_dot_plot_section,
        shap_decision_plot_section,
        pdp_section,
        lstm_section,
        sarima_section,
        analysis_summary_section,
    ],
    fluid = True,
)

# Callback to toggle visibility of each collapse
@callback(
    Output('feature_analysis_collapse', 'is_open'),
    [
        Input('feature_analysis_button', 'n_clicks'),
    ],
    [
        State('feature_analysis_collapse', 'is_open')
    ],
)
def toggle_left(n_left, is_open):
    if n_left:
        return not is_open
    return is_open


@callback(
    Output('time_series_analysis_collapse', 'is_open'),
    [
        Input('time_series_analysis_button', 'n_clicks'),
    ],
    [
        State('time_series_analysis_collapse', 'is_open')
    ],
)
def toggle_left(n_right, is_open):
    if n_right:
        return not is_open
    return is_open

# Callback for Feature Importances
@callback(
    [
        Output('sj_feature_importances_section', 'figure')
    ],
    [
        Input('region_option', 'value'),
    ]
)
# Animated plot function
def update_sj_feature_importances_section(loc):
    feature_importances_section = None

    df = None
    df = sj_df if loc == 'sj' else sf_df

    importances_fn = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    if not os.path.exists(importances_fn):
        feature_importances_section = processing_pipeline(df, loc)
    else:
        feature_importances_section = load(importances_fn)
    
    fig = px.bar(
        feature_importances_section,
        x = 'feature',
        y = 'importances',
    )

    fig.update_layout(
        xaxis_title = 'Features',
        yaxis_title = 'Importances',
        xaxis_tickangle = -45,
    )

    return [fig]

# LSTM - single step
@callback(
    [
        Output('lstm_val_score', 'children'),
        Output('lstm_test_score', 'children'),
        Output('lstm_plot', 'figure'),
    ],
    [
        Input('region_select', 'value')
    ]
)
def update_lstm_single_step(region):
    request_new_joblib = False  # Change to True for new lstm file
    file_specifier = 1
    shift = 1

    joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_single_step_{file_specifier}.joblib'

    plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    lstm_results = None

    # Load LSTM scores and predictions
    if request_new_joblib:
        lstm_results = pred_lstm_single_step(region, file_specifier, shift)
    else:
        if os.path.exists(joblib_filename_lstm_res):
            lstm_results = load(joblib_filename_lstm_res)
    
    val_score = lstm_results['val_score']
    test_score = lstm_results['test_score']
    # lstm_val_score = lstm_results['lstm_val_score']
    # lstm_test_score = lstm_results['lstm_test_score']

    inputs = lstm_results['inputs']
    labels = lstm_results['labels']
    predictions = lstm_results['predictions']

    inputs = inputs.numpy()
    labels = labels.numpy()

    # Set scores
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score[1]:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score[1]:.3f}'

    fig = go.Figure()

    # Want the 3rd pred for this
    input_col_index = 2
    label_col_index = 0

    # Input line
    fig.add_trace(
        go.Scatter(
            x = list(range(len(inputs[0, 1:, input_col_index]))),
            y = inputs[0, 1:, input_col_index],
            mode = 'lines',
            name = 'Inputs',
            line = dict(color = 'blue'),
        )
    )

    # Label point (o)
    fig.add_trace(
        go.Scatter(
            x = list(range(len(labels[0, :, label_col_index]))),
            y = labels[0, :, label_col_index].flatten(), 
            mode = 'markers',
            name = 'Labels',
            marker = dict(size = 10, color = 'green', symbol = 'circle'),
        )
    )

    # Prediction point (x)
    fig.add_trace(
        go.Scatter(
            x = list(range(len(predictions[0, :, label_col_index]))),
            y = predictions[0, :, label_col_index].flatten(),
            mode = 'markers',
            name = 'Predictions',
            marker = dict(size = 10, color = 'red', symbol = 'x'),
        )
    )

    # Update the layout
    fig.update_layout(
        title = f'Single Step LSTM Prediction for {plot_title}',
        xaxis_title = 'Time Steps (Months)',
        yaxis_title = 'Average Energy Usage (kWh)',
        legend = dict(x = 0, y = 1),
    )

    return lstm_val_score, lstm_test_score, fig

# LSTM - multi step
@callback(
    [
        Output('lstm_val_score_multi', 'children'),
        Output('lstm_test_score_multi', 'children'),
        Output('lstm_plot_multi', 'figure'),
    ],
    [
        Input('region_dropdown', 'value')
    ]
)
def update_lstm_mutli_step(region):
    request_new_joblib = False  # Change to True for new lstm file
    file_specifier = 1
    shift = 12

    joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_multi_step_{file_specifier}.joblib'

    plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    lstm_results = None

    # Load LSTM scores and predictions
    if request_new_joblib:
        lstm_results = pred_lstm_multi_step(region, file_specifier, shift)
    else:
        if os.path.exists(joblib_filename_lstm_res):
            lstm_results = load(joblib_filename_lstm_res)
    
    val_score = lstm_results['val_score']
    test_score = lstm_results['test_score']
    inputs = lstm_results['inputs']
    labels = lstm_results['labels']
    predictions = lstm_results['predictions']

    inputs = inputs.numpy()
    labels = labels.numpy()

    # Set scores
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score[1]:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score[1]:.3f}'

    fig = go.Figure()

    # Want the 3rd pred for this
    input_col_index = 2
    label_col_index = 0

    # Inputs line (up to the last point of actual data)
    fig.add_trace(
        go.Scatter(
            x = list(range(len(inputs[0, :, input_col_index]))),
            y = inputs[0, :, input_col_index],
            mode = 'lines',
            name = 'Inputs',
            line = dict(color = 'blue'),
        )
    )

    #### Need to figure out x range for the points to be plotted ####
    # Labels (actual future values, for comparison with predictions)
    fig.add_trace(
        go.Scatter(
            x = list(range(len(inputs[0, :, input_col_index]), len(inputs[0, :, input_col_index]) + len(labels[0, :, label_col_index]))),
            y = labels[0, :, label_col_index].flatten(),
            mode = 'markers',
            name = 'Labels',
            marker = dict(size = 10, color = 'green', symbol = 'circle'),
        )
    )

    #### Need to figure out x range for the points to be plotted ####
    # Predictions (multi-step future predictions)
    fig.add_trace(
        go.Scatter(
            x = list(range(len(inputs[0, :, input_col_index]), len(inputs[0, :, input_col_index]) + len(predictions[0, :, label_col_index]))),
            y = predictions[0, :, label_col_index].flatten(),
            mode = 'markers',
            name = 'Predictions',
            marker = dict(size = 10, color = 'red', symbol = 'x'),
        )
    )

    # Update layout
    fig.update_layout(
        title = f'Multi-Step LSTM Prediction for {plot_title}',
        xaxis_title = 'Time Steps (Months)',
        yaxis_title = 'Average Energy Usage (kWh)',
        legend = dict(x = 0, y = 1)
    )

    return lstm_val_score, lstm_test_score, fig


# Work in progress
# LSTM Future Predictions callback
@callback(
    [
        Output('future-prediction-graph', 'figure'),
        Output('prediction-output', 'children'),
    ],
    [
        Input('predict-button', 'n_clicks'),
    ],
    [
        State('region-dropdown', 'value'),
    ],
)
def update_future_prediction(n_clicks, region): # BASICALLY PASS FOR NOW
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

# SARIMA
@callback(
    [
        Output('sarima_mae', 'children'),
        Output('sarima_plot', 'figure'),
    ],
    [
        Input('sarima_region_dropdown', 'value')
    ]
)
def update_sarima(region):
    request_new_joblib = False
    file_specifier = 1

    joblib_filename_sarima_res = f'joblib_files/sarima/{region}_sarima_{file_specifier}.joblib'

    plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    sarima_results = None

    # Load LSTM scores and predictions
    if request_new_joblib:
        sarima_results = pred_sarima(region, request_new_joblib, file_specifier)
    else:
        if os.path.exists(joblib_filename_sarima_res):
            sarima_results = load(joblib_filename_sarima_res)

    test = sarima_results['test']
    df = sarima_results['df']
    mae_SARIMA = sarima_results['mae_SARIMA']

    mae_score = f'Mean Absolute Error (MAE): {mae_SARIMA:.3f}'

    fig = go.Figure()

    # Main time series line
    fig.add_trace(
        go.Scatter(
            x = df['year-month'],
            y = df['averagekwh'],
            mode = 'lines',
            name = 'Observed Data',
            line = dict(color = 'blue')
        )
    )

    # SARIMA predictions
    sarima_label = 'SARIMA(2, 1, 1)(0, 1, 1)12' if region == 'sj' else 'SARIMA(1, 1, 1)(3, 1, 3)12'
    fig.add_trace(
        go.Scatter(
            x = test['year-month'],
            y = test['SARIMA_pred'],
            mode = 'lines',
            name = sarima_label,
            line = dict(color = 'green', dash = 'dot')
        )
    )

    # Update layout for the plot
    fig.update_layout(
        title = f'SARIMA Forecast for {plot_title}',
        xaxis_title = 'Date',
        yaxis_title = 'Average Energy Usage (kWh)',
        legend = dict(x = 0.01, y = 0.99),

        xaxis = dict(
            range = ['2023-05', '2024-07'],
            tickformat = '%Y-%m',
            tickvals = pd.date_range(start = '2023-05', end = '2024-07', freq = 'MS'),
        )
    )

    return mae_score, fig