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
                                            'height': '25px',
                                            'margin-right': '10px',
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
                                            'height': '25px',
                                            'margin-right': '10px',
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
            className = 'mb-3',
        ),
        dbc.Row(
            [
                # SJ PDP
                dbc.Col(
                    [
                        dbc.Carousel(
                            items=[
                                {
                                    "key": "1",
                                    "src": "/assets/pdp_plots/sj_pdp_season.png",
                                },
                                {
                                    "key": "2",
                                    "src": "/assets/pdp_plots/sj_pdp_tmax.png",
                                },
                                {
                                    "key": "3",
                                    "src": "/assets/pdp_plots/sj_pdp_tmin.png",
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
                            'LSTM Results',
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
                        html.P(
                            'LSTM Future Predictions',
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
                        html.P(
                            'Autoregressive Integrated Moving Average (ARIMA) [ETS?]',
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

# LSTM 
@callback(
    [Output('lstm-scores', 'children'),
     Output('lstm-plot', 'figure')],
    [Input('region-select', 'value')]
)
def update_lstm_analysis(region):
    request_new_joblib = False
    file_specifier = 1

    joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_results_{file_specifier}.joblib'


    plot_title = region
    if region == 'sj':
        plot_title = 'San Jose'
    else:
        plot_title = 'San Francisco'

    lstm_scores = None
    actual_data = None
    lstm_predictions = None

    if request_new_joblib:
        # Call the LSTM calculation function
        lstm_scores, actual_data, lstm_predictions = calc_lstm(region, request_new_joblib, file_specifier)
    else:
        if os.path.exists(joblib_filename_lstm_res):
            res = load(joblib_filename_lstm_res)
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