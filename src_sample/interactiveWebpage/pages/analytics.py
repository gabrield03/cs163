import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly
import plotly.express as px
import plotly.graph_objs as go
from pdpbox import pdp

from  utils.data_pipeline import (
    processing_pipeline,
    calc_shap,
    lstm_predict,
    pred_lstm, pred_lstm_multi,
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

analytics_header = html.Div(
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
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
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
                            title=html.Div(
                                [
                                    html.Img(
                                        src = '/assets/Feature_Analysis_Icon.png',
                                        style = {
                                            'height': '20px',
                                            'margin-right': '20px',
                                        },
                                    ),
                                    html.P(
                                        "Feature Analysis",
                                        style = {
                                            'font-size': '25px',
                                            'color': '#000000',
                                            'margin': '0px',
                                        },
                                    ),
                                ],
                                style = {
                                    'display': 'flex',
                                    'align-items': 'center',
                                },
                            ),
                        ),
                        dbc.AccordionItem(
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
                            title=html.Div(
                                [
                                    html.Img(
                                        src = '/assets/Time-Series_Icon2.png',
                                        style = {
                                            'height': '20px',
                                            'margin-right': '17px',
                                        },
                                    ),
                                    html.P(
                                        'Time-Series Analysis',
                                        style = {
                                            'font-size': '25px',
                                            'color': '#000000',
                                            'margin': '0px',
                                        },
                                    ),
                                ],
                                style = {
                                    'display': 'flex',
                                    'align-items': 'center',
                                },
                            ),
                        ),
                    ],
                    start_collapsed = True,
                    flush = True,
                    always_open = True,
                ),
            ],
            className = 'mb-3',
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
                                'the feature importances of weather variables. [explanation of the plot] ',
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
                                'Toggling between the two options show the mean SHAP values for each region. Several ',
                                'features from the San Jose data have been omitted so that each region has the their ',
                                'SHAP values calculated based on the same features.',
                                
                                html.Br(),
                                'The most important feature for the San Jose region is ',
                                html.Span('season ', style = {'font-weight': 'bold', 'font-style': 'normal', 'color': 'red',}),
                                'with a SHAP value of 20.88. Season is twice as important to the overall model predictions ',
                                'than the second and third most important features (tmax and tmin). ',

                                'On the other hand, the most important feature for the San Francisco region is ',
                                html.Span('tmax ', style = {'font-weight': 'bold', 'font-style': 'normal', 'color': 'red',}),
                                'with a SHAP value of 14.5. The next two most important features (tmin and totalcustomers). ',
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
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.RadioItems(
                            id='regional_shap_option',
                            options=[
                                {
                                    'label': html.Div(['San Jose'], style={'color': '#ffffff', 'font-size': 20}),
                                    'value': 'sj',
                                },
                                {
                                    'label': html.Div(['San Francisco'], style={'color': '#ffffff', 'font-size': 20}),
                                    'value': 'sf',
                                },
                            ],
                            value='sj',
                            labelStyle={'display': 'inline-block', 'margin': '0 20px'},
                        ),
                    ],
                    width={"size": 6, "offset": 3},
                    style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
                ),
            ],
            className='mb-3',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                                id = 'regional_shap',
                                figure = {},
                        ),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
            className = 'mb-5',
        ),
        # Decision Plots
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

# PDP - additional feature interpretability -> looking to change this to permutation models
analytics_objective_1_3 = html.Div(
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
                                'displayed in the plots below. ',
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

# Regional Comparisons of feature importances
analytics_objective_1_4 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'Comparing Feature Importances Between Regions',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000',
                            },
                        ),
                        html.P(
                            [
                                'Two models—one for San Francisco and one for San Jose—will be trained. ',
                                'By comparing feature importance between the models, I will analyze whether ',
                                'specific weather variables have a greater impact in one region than in the other. ',
                                'The preliminary focus will be on variables like maximum and minimum temperatures, ',
                                'precipitation, and seasonality.',

                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                '[explanation of plot that will be below]'
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
                        # Dropdown to select the region
                        dcc.Dropdown(
                            id='region_dropdown',
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
                        dcc.Graph(id='lstm_plot_multi'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Div to display LSTM scores
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
analytics_objective_2_2 = html.Div(
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
                                '[explanation of plot below]',
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
                        # Dropdown to select the region
                        dcc.Dropdown(
                            id='sarima_region_dropdown',
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
                        dcc.Graph(id='sarima_plot'),
                    ],
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        # Div to display LSTM scores
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
def update_sj_feature_importances(loc):
    feature_importances = None

    df = pd.DataFrame()
    if loc == 'sj':
        df = sj_df
    elif loc == 'sf':
        df = sf_df

    importances_fn = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    if not os.path.exists(importances_fn):
        feature_importances = processing_pipeline(df, loc)
    else:
        feature_importances = load(importances_fn)
    
    # Need to add title and expand the actual axis
    fig = px.bar(
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

# SHAP Callback and Functions
@callback(
    Output('regional_shap', 'figure'),
    [Input('regional_shap_option', 'value')],
)
def update_sj_shap(loc):
    shap_plot_df = None

    joblib_filename_shap = f'joblib_files/shap/{loc}_shap_plot.joblib'
    if not os.path.exists(joblib_filename_shap):
        shap_plot_df = calc_shap(loc)
    else:
        shap_plot_df = load(joblib_filename_shap)


    region = 'San Jose'
    if loc == 'sf':
        region = 'San Francisco'


    # Create the Plotly figure for visualization
    fig = {
        'data': [
            {
                'x': shap_plot_df['Feature'],
                'y': shap_plot_df['Mean SHAP Value'],
                'type': 'bar',
                'marker': {'color': 'blue'},
            }
        ],
        'layout': {
            'title': f'Mean SHAP Values for {region}',
            'xaxis': {'title': 'Features'},
            'yaxis': {'title': 'Mean SHAP Value'},
        }
    }

    return fig

# LSTM - single step
@callback(
    [Output('lstm_val_score', 'children'),
     Output('lstm_test_score', 'children'),
     Output('lstm-plot', 'figure')],
    [Input('region-select', 'value')]
)
def update_lstm_analysis(region):
    request_new_joblib = False  # Change this to True if you want to force a recalculation
    file_specifier = 1

    joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_results_{file_specifier}.joblib'

    # Map region values to plot titles
    plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    lstm_results = None

    # Load LSTM scores and predictions
    if request_new_joblib:
        lstm_results = pred_lstm(region, request_new_joblib, file_specifier)
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

    # Prepare the scores report
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score[1]:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score[1]:.3f}'

    fig = go.Figure()

    # Want the 3rd pred for this
    input_col_index = 2
    label_col_index = 0

    # Input line
    fig.add_trace(go.Scatter(
        x = list(range(len(inputs[0, 1:, input_col_index]))),
        y = inputs[0, 1:, input_col_index],
        mode = 'lines',
        name = 'Inputs',
        line = dict(color = 'blue'),
    ))

    # Label point (o)
    fig.add_trace(go.Scatter(
        x = list(range(len(labels[0, :, label_col_index]))),
        y = labels[0, :, label_col_index].flatten(), 
        mode = 'markers',
        name = 'Labels',
        marker = dict(size = 10, color = 'green', symbol = 'circle'),
    ))

    # Prediction point (x)
    fig.add_trace(go.Scatter(
        x = list(range(len(predictions[0, :, label_col_index]))),
        y = predictions[0, :, label_col_index].flatten(),
        mode = 'markers',
        name = 'Predictions',
        marker = dict(size = 10, color = 'red', symbol = 'x'),
    ))

    # Update the layout
    fig.update_layout(
        title = f'Single Step LSTM Prediction for {plot_title}',
        xaxis_title = 'Time Steps (Months)',
        yaxis_title = 'Average Energy Usage (kWh)',
        legend = dict(x = 0, y = 1)
    )

    return lstm_val_score, lstm_test_score, fig

# LSTM - multi step
@callback(
    [Output('lstm_val_score_multi', 'children'),
     Output('lstm_test_score_multi', 'children'),
     Output('lstm_plot_multi', 'figure')],
    [Input('region_dropdown', 'value')]
)
def update_lstm_analysis_multi(region):
    request_new_joblib = False
    file_specifier = 1
    shift = 12

    joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_results_multi_{file_specifier}.joblib'

    # Map region values to plot titles
    plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    lstm_results = None

    # Load LSTM scores and predictions
    if request_new_joblib:
        lstm_results = pred_lstm_multi(region, request_new_joblib, file_specifier, shift)
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

    # Prepare the scores report
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score[1]:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score[1]:.3f}'

    fig = go.Figure()

    # Want the 3rd pred for this
    input_col_index = 2
    label_col_index = 0

    # Inputs line (up to the last point of actual data)
    fig.add_trace(go.Scatter(
        x=list(range(len(inputs[0, :, input_col_index]))),
        y=inputs[0, :, input_col_index],
        mode='lines',
        name='Inputs',
        line=dict(color='blue'),
    ))

    # Labels (actual future values, for comparison with predictions)
    fig.add_trace(go.Scatter(
        x=list(range(len(inputs[0, :, input_col_index]), len(inputs[0, :, input_col_index]) + len(labels[0, :, label_col_index]))),
        y=labels[0, :, label_col_index].flatten(),
        mode='markers',
        name='Labels',
        marker=dict(size=10, color='green', symbol='circle'),
    ))

    # Predictions (multi-step future predictions)
    fig.add_trace(go.Scatter(
        x=list(range(len(inputs[0, :, input_col_index]), len(inputs[0, :, input_col_index]) + len(predictions[0, :, label_col_index]))),
        y=predictions[0, :, label_col_index].flatten(),
        mode='markers',
        name='Predictions',
        marker=dict(size=10, color='red', symbol='x'),
    ))

    # Update layout
    fig.update_layout(
        title=f'Multi-Step LSTM Prediction for {plot_title}',
        xaxis_title='Time Steps (Months)',
        yaxis_title='Average Energy Usage (kWh)',
        legend=dict(x=0, y=1)
    )

    return lstm_val_score, lstm_test_score, fig


# Work in progress
# LSTM Future Predictions callback
@callback(
    Output('future-prediction-graph', 'figure'),
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('region-dropdown', 'value')
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

    # Map region values to plot titles
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
    fig.add_trace(go.Scatter(
        x=df['year-month'],
        y=df['averagekwh'],
        mode='lines',
        name='Observed Data',
        line=dict(color='blue')
    ))

    # SARIMA predictions
    sarima_label = 'SARIMA(2, 1, 1)(0, 1, 1)12' if region == 'sj' else 'SARIMA(1, 1, 1)(3, 1, 3)12'
    fig.add_trace(go.Scatter(
        x=test['year-month'],
        y=test['SARIMA_pred'],
        mode='lines',
        name=sarima_label,
        line=dict(color='green', dash='dot')
    ))

    # Update layout for the plot
    fig.update_layout(
        title=f'SARIMA Forecast for {plot_title}',
        xaxis_title='Date',
        yaxis_title='Average Energy Usage (kWh)',
        legend=dict(x=0.01, y=0.99),

        xaxis = dict(
            range=['2023-05', '2024-07'],
            tickformat='%Y-%m',
            tickvals=pd.date_range(start='2023-05', end='2024-07', freq='MS')
        )
    )

    return mae_score, fig