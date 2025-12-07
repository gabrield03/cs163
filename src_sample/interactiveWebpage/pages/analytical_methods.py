import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from scipy.stats import ttest_ind

from utils.data_pipeline import (
    processing_pipeline,
    pred_lstm_single_step, pred_lstm_multi_step,
    pred_sarima,
    load_joblib_from_github
)
import requests
from io import BytesIO
import os
from joblib import dump, load


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

analytics_header_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Analytical Methods',
                    ),
                    className = 'text-center mb-5 mt-5',
                    width = 12,
                    style = {
                        'font-size': '50px',
                        'height': '100%',
                        'font-variant': 'small-caps',
                        
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
                            'This section focuses on three of the following advanced analysis methods:',
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
                    width = 4,
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            html.Span(
                                'Extreme Events Analysis',
                                id='extreme_events_analysis_tooltip',
                            ),
                            color='secondary',
                            id='extreme_events_analysis_button',
                            className='me-1 small-button',
                            n_clicks=0,
                            style={
                                'font-size': '16px',
                                'background-color': '#bdc3c7',
                            },
                        ),
                    ],
                    width = 4,
                ),
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
                    width = 4,
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
                    target='extreme_events_analysis_tooltip',
                    placement='top',
                    style={
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
                                        'This analysis focuses on understanding the influence of historical weather data on ',
                                        'energy consumption in San Francisco and San Jose. The goal is to identify which weather ',
                                        'variables have a significant impact on energy consumption and whether these impacts differ ',
                                        'between the two regions. It will reveal disproportionate effects of weather conditions ',
                                        'on energy usage.',
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
                                        'This analysis will look at extreme (hot and cold) events in each region. The thresholds ',
                                        'set for extreme events are the hottest 90% months in a year and the coldest 10% months ',
                                        'in a year. It focuses on identifying whether the occurrences of each event is statistically ',
                                        'significant or if it could be due to random weather variability.'
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
                            id = 'extreme_events_analysis_collapse',
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
                            [
                                html.Br(),
                                'Feature Interpretability with SHapley Additive exPlanations (SHAP)',
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
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
    sj_shap_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/shap/sj_shap_plot.joblib"
    sf_shap_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/shap/sf_shap_plot.joblib"

    sj_shap = load_joblib_from_github(sj_shap_url)
    sf_shap = load_joblib_from_github(sf_shap_url)

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
                        dbc.Carousel(
                            items=[
                                {
                                    'key': '1',
                                    'src': 'assets/shap_plots/sj_shap_decision.png',
                                },
                                {
                                    'key': '2',
                                    'src': 'assets/shap_plots/sj_shap_decision_lowest.png',
                                },
                                {
                                    'key': '3',
                                    'src': 'assets/shap_plots/sj_shap_decision_highest.png',
                                },
                            ],
                            controls = True,
                            indicators = True,
                            className = 'carousel-fade',
                        ),
                    ],
                    width = 6,
                ),
                dbc.Col(
                    [
                        dbc.Carousel(
                            items=[
                                {
                                    'key': '1',
                                    'src': 'assets/shap_plots/sf_shap_decision.png',
                                },
                                {
                                    'key': '2',
                                    'src': 'assets/shap_plots/sf_shap_decision_lowest.png',
                                },
                                {
                                    'key': '3',
                                    'src': 'assets/shap_plots/sf_shap_decision_highest.png',
                                },
                            ],
                            controls = True,
                            indicators = True,
                            className = 'carousel-fade',
                        ),
                    ],
                    width = 6,
                ),
            ]
        )
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
                            [
                                html.Br(),
                                'Partial Dependence Plots (PDP)',
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
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
    className = 'mb-10',
)



# Extreme weather events
extreme_weather_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                'Extreme Weather Events and Their Significance',
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'One of the goals of this project is to perform a surface-level analysis assessing ',
                                'regional shifts in climate, their frequency, and their potential impacts on energy usage. ',
                                'This involves evaluating changes in the occurrence of extreme weather events ',
                                '(both hot and cold) over time in San Jose and San Francisco.',

                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '25px',
                                'word-break': 'keep-all',
                            },
                        ),
                        html.P(
                            [
                                'The visualizations below show two bar plots: one for extreme cold events and another for ',
                                'extreme hot events. Each plot displays the proportions of extreme events observed in two time ',
                                'periods: Past (2013 - 2018) and Recent (2019 -2023), separated by region (SJ and SF). ',
                                'A two-sample t-test was performed to assess whether the observed differences in proportions ',
                                'between the Past and Recent periods are statistically significant.',
                            ],
                            style={
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                            },
                        ),
                        html.P(
                            [
                                'The two-sample t-test was chosen because the sample sizes for some periods are small, ',
                                'and it allows for comparing the means of two independent samples without assuming equal variance. ',
                                html.Br(), html.Br(), html.Br(),

                                html.H3('Hot Extreme Events:'),

                                html.Span(
                                    'San Jose (SJ):',
                                    style = {
                                        'font-weight': 'bold',
                                    },
                                ),
                                'The proportion of events increased from 5.13% in the Past to 16.67% in the Recent period, ',
                                'with a statistically significant p-value of 0.0375, suggesting a meaningful increase in extreme hot events over time.',
                                html.Br(),
                                html.Span(
                                    'San Francisco (SF):',
                                    style = {
                                        'font-weight': 'bold',
                                    },
                                ),
                                'The proportion of events showed a modest increase from 9.86% to 11.67%, ',
                                'but with a p-value of 0.7426, indicating no statistically significant change.',
                                
                                html.Br(), html.Br(), html.Br(),

                                html.H3('Cold Extreme Events:'),

                                html.Span(
                                    'San Jose (SJ):',
                                    style = {
                                        'font-weight': 'bold',
                                    },
                                ),
                                'The proportion of events decreased from 11.54% to 8.33%, but with a p-value of 0.5323, ',
                                'this change is not statistically significant.',
                                html.Br(),
                                html.Span(
                                    'San Francisco (SF):',
                                    style = {
                                        'font-weight': 'bold',
                                    },
                                ),
                                'The proportion of events increased significantly from 5.63% to 18.33%, ',
                                'with a p-value of 0.0295, suggesting a notable rise in extreme cold events in this region.',
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
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
                        dcc.Dropdown(
                            id = 'temperature_option',
                            options = [
                                {'label': 'Hot Extreme Events', 'value': 'hot'},
                                {'label': 'Cold Extreme Events', 'value': 'cold'},
                            ],
                            value = 'hot',

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
                        dcc.Graph(
                            id = 'extreme-weather-corr'
                        )
                    ],
                    width=12
                ),
            ],
            className = 'mb-3',
        ),
        dbc.Row(
            [
                dbc.Col([], width = 1),
                
                dbc.Col(
                    [
                        html.P(
                            [
                                'p-value (SJ): ',
                                html.Span(
                                    id = 'extreme-weather-sj-p-value',
                                ),
                            ],
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                    ],
                    width = 4,
                ),
                dbc.Col(
                    [
                        html.P(
                            [
                                'p-value (SF): ',
                                html.Span(
                                    id = 'extreme-weather-sf-p-value',
                                ),
                            ],
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                    ],
                    width = 4,
                ),

                dbc.Col([], width = 1),
            ],
            justify = 'between',
        ),
    ],
    className = 'mb-10',
)

# Time-series analysis with LSTM
lstm_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            [
                                html.Br(),
                                'Time-Series Analysis to Forecast Energy Demand with LSTM and SARIMA',
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                #'text-shadow': '2px 2px 4px #000000',
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
                                #'text-shadow': '2px 2px 4px #000000',
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
                            id = 'lstm_train_score',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                        html.Div(
                            id = 'lstm_val_score',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                        html.Div(
                            id = 'lstm_test_score',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
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
                                #'text-shadow': '2px 2px 4px #000000',
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
                            id = 'lstm_train_score_multi',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                        html.Div(
                            id = 'lstm_val_score_multi',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
                            },
                        ),
                        html.Div(
                            id = 'lstm_test_score_multi',
                            style = {
                                'marginTop': 20,
                                'color': 'black',
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
                            [
                                html.Br(),
                                'Seasonal Autoregressive Integrated Moving Average (SARIMA)',
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                #'text-shadow': '2px 2px 4px #000000',
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
                                #'text-shadow': '2px 2px 4px #000000',
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
                            [
                                html.Br(),
                                'Summary of Analysis',
                                html.Br(),
                            ],
                            style = {
                                'text-align': 'center',
                                'font-size': '60px',
                                'font-variant': 'small-caps',
                            },
                        ),

                        # Feature Analysis Summary
                        html.P(
                            [
                                'Analysis of Features',
                                html.Br(),
                            ],
                            style = {
                                'text-align': 'left',
                                'font-size': '35px',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'The analysis demonstrates that energy consumption patterns in San Jose ',
                                'and San Francisco are influenced by distinct weather-related factors, ', 
                                'revealing how local climate characteristics can lead to differing energy ',
                                'demands between regions. Using a random forest model paired with SHAP ',
                                '(SHapley Additive exPlanations) - a statistical method that breaks down the ',
                                'impact of each feature on model predictions - I quantified the importance ',
                                'of various factors. SHAP is particularly valuable here because it assigns ',
                                '"importance" scores to features based on their average impact on model ',
                                'predictions. This provides a clear assessment of each feature\'s role.',

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
                                'seasonal cycles play a dominant role in its demand pattern. ',
                                
                                html.Br(), html.Br(),
                                'It is important to note that this project does not delve into the possible ',
                                'interdependency between temperature and seasonality. So, it would be inappropriate',
                                'to conclude that San Jose is less vulnerable to changes in climate. ',
                                'It would be beneficial for future analysis to analyze the relationship', 
                                'between seasonality and global temperature shifts.',
                                
                                html.Br(), html.Br(),
                            ],
                            style = {
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                            },
                        ),


                        # Extreme Events Summary
                        html.P(
                            [
                                html.Br(),
                                'Analysis of Extreme Events',
                                html.Br(),
                            ],
                            style={
                                'text-align': 'left',
                                'font-size': '35px',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'The occurrence and significance of extreme weather events were analyzed to identify potential ',
                                'regional shifts in climate patterns and their impact on energy usage. The analysis focused ',
                                'on two types of events: extreme hot and extreme cold across two distinct periods: Past (2013 - 2018) ',
                                'and Recent (2019 - 2023), for San Jose and San Francisco. The results reveal that the ',
                                'frequency of these events have statistically changed for some regions based on the type of event.',

                                html.Br(), html.Br(),

                                'To assess the statistical significance of differences in event proportions between the Past and Recent periods, ',
                                'a two-sample t-test was performed. This test was chosen for its ability to evaluate differences in means between ',
                                'two independent samples. The resulting p-values are critical in determining whether observed differences ',
                                'represent genuine shifts in weather patterns or are attributable to random variability.',

                                html.Br(), html.Br(),

                                'For hot extreme events, the results indicate contrasting trends between the two regions. In San Jose, the ',
                                'proportion of hot extreme events increased significantly, rising from 5.13% to 16.67%. This increase is ',
                                'statistically significant (p-value = 0.0375), and suggests that extreme heat events have become ',
                                'more frequent in this region. This shift could have implications for increased energy demand due to cooling requirements during these time periods.',
                                'In San Francisco, the proportion of hot extreme events also increased (9.86% to 11.67%), but ',
                                'the resulting p-value of 0.7426 indicates no statistical significance. This suggests that, ',
                                'the observed increase in extreme heat events in SF may not represent a meaningful or consistent trend.',

                                html.Br(), html.Br(),

                                'For cold extreme events, the results reveal an interesting divergence in weather trends between the two regions. ',
                                'In San Jose, the proportion of cold extreme events declined from 11.54% to 8.33%. ',
                                'The p-value of 0.5323 indicates that this decrease is not statistically significant. The cold extremes ',
                                'that were observed could have been caused by random variability in the weather.',
                                'In San Francisco, the proportion of cold extreme events increased markedly, rising from 5.63% ',
                                'to 18.33%. Its corresponding p-value 0.0295 is statistically significant. This increase points to a meaningful ',
                                'rise in the frequency of cold extremes in SF, potentially reflecting localized shifts in weather variability.',

                                html.Br(), html.Br(),

                                'The results can help us understand the shifting climates in each region. Often, the two regions experience',
                                'different weather patterns: one facing increased heat extremes while the other faces rising cold extremes. ',
                                'The p-values serve as a measure of confidence in the observed changes. The low p-values as seen in SJ hot extremes ',
                                '(p = 0.0375) and SF cold extremes (p = 0.0295) suggests real and consistent temperature shifts, while the higher ',
                                'p-values indicate observations due to random variation.',

                                html.Br(), html.Br(),
                            ],
                            style={
                                'font-size': '20px',
                                'word-break': 'keep-all',
                                'font-style': 'italic',
                            },
                        ),


                        # Time-Series Summary
                        html.P(
                            [
                                'Analysis of Time-Series Predictions',
                                html.Br(),
                            ],
                            style = {
                                'text-align': 'left',
                                'font-size': '35px',
                                'font-variant': 'small-caps',
                            },
                        ),
                        html.P(
                            [
                                'Overall, the LSTM models show significant overfitting, as evidenced by the lower ',
                                'training MAE scores compared to their test MAE scores. In both regions, the ',
                                'training MAEs for the single-step and multi-step LSTM models are much lower than ',
                                'the validation and test MAE. The validation and test MAE\'s, for both models and ',
                                'both regions, are in the 30 kWh range, except for San Francisco multi-step model, ',
                                'where the test MAE is in the low 20s.',

                                html.Br(), html.Br(),

                                'Both regions have a dedicated LSTM and SARIMA model. When model performance is ',
                                'compared, we are comparing the SARIMA models with the multi-step LSTM models. The ',
                                'results are that the SARIMA models outperformed the LSTM multi-step models. ',
                                'Specifically, the SARIMA MAE scores were 22.895 for San Jose and 13.082 for San ',
                                'Francisco, while the LSTM multi-step models had test MAE scores of 34.985 and ',
                                '22.524, respectively.',

                                html.Br(), html.Br(),

                                'These results align with expectations given the nature of each model. Neural ',
                                'network models, the LSTMs for this project, typically require large datasets to ',
                                'accurately learn and generalize complex, nonlinear relationships. The datasets ',
                                'in this project were fairly small, which may have limited the LSTM models\' ',
                                'performance. In contrast, SARIMA models have fewer parameters and are less complex ',
                                'which allow it to effectively learn underlying patterns with fewer data points.',

                                html.Br(), html.Br(),
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
        html.Div(
            [
                analytics_header_section,
                analytics_info_section,
                feature_importances_section,
                shap_intro_section,
                shap_dot_plot_section,
                shap_decision_plot_section,
                pdp_section,
                extreme_weather_section,
                lstm_section,
                sarima_section,
                analysis_summary_section,
            ],
            style = {
                'padding': '0px 100px',
                'backgroundColor': '#FAF9F6',
            },
        ),
    ],
    fluid = True,
)

# Callback to toggle visibility of each collapse
@callback(
    [
        Output('feature_analysis_collapse', 'is_open'),
        Output('extreme_events_analysis_collapse', 'is_open'),
        Output('time_series_analysis_collapse', 'is_open'),
    ],
    [
        Input('feature_analysis_button', 'n_clicks'),
        Input('extreme_events_analysis_button', 'n_clicks'),
        Input('time_series_analysis_button', 'n_clicks'),
    ],
    [
        State('feature_analysis_collapse', 'is_open'),
        State('extreme_events_analysis_collapse', 'is_open'),
        State('time_series_analysis_collapse', 'is_open'),
    ],
)
def toggle_sections(feature_click, extreme_click, time_click, feature_open, extreme_open, time_open):
    # Determine which button triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return feature_open, extreme_open, time_open

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Toggle the state of the clicked button while keeping others unchanged
    return (
        not feature_open if button_id == 'feature_analysis_button' else feature_open,
        not extreme_open if button_id == 'extreme_events_analysis_button' else extreme_open,
        not time_open if button_id == 'time_series_analysis_button' else time_open,
    )

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

    # importances_fn = f'joblib_files/processed_data/{loc}_importances_df.joblib'
    # if not os.path.exists(importances_fn):
    #     feature_importances_section = processing_pipeline(df, loc)
    # else:
    #     feature_importances_section = load(importances_fn)

    importances_fn_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/processed_data/{loc}_importances_df.joblib"
    feature_importances_section = load_joblib_from_github(importances_fn_url)

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



# Helper function for analyze correlation
def perform_ttest(df, event_col, recent_col):
    # Split data into recent and past samples
    recent_sample = df[df[recent_col]][event_col]
    past_sample = df[~df[recent_col]][event_col]

    # Perform two-sample t-test
    t_stat, p_value = ttest_ind(recent_sample, past_sample, equal_var = False)

    return round(t_stat, 2), round(p_value, 4)

# Helper function to calculate the proportion of extreme events
def calculate_proportions(df, event_col, recent_col):
    # Group by recent and past periods
    counts = df.groupby(recent_col)[event_col].agg(['sum', 'count'])
    counts['proportion'] = counts['sum'] / counts['count']

    return counts

# Extreme weather callback
@callback(
    [
        Output('extreme-weather-corr', 'figure'),
        Output('extreme-weather-sj-p-value', 'children'),
        Output('extreme-weather-sf-p-value', 'children'),
    ],
    Input('temperature_option', 'value')
)
def analyze_regional_correlation(weather_type):
    # Load data
    # sj_fn = 'joblib_files/base_data/sj_combined.joblib'
    # sf_fn = 'joblib_files/base_data/sf_combined.joblib'
    # sj_df = load(sj_fn)
    # sf_df = load(sf_fn)

    sj_fn_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sj_combined.joblib"
    sf_fn_url = "https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/base_data/sf_combined.joblib"
    sj_df = load_joblib_from_github(sj_fn_url)
    sf_df = load_joblib_from_github(sf_fn_url)



    # Define thresholds for extreme events
    hot_thresholds = {'sj': sj_df['tmax'].quantile(0.90), 'sf': sf_df['tmax'].quantile(0.90)}
    cold_thresholds = {'sj': sj_df['tmin'].quantile(0.10), 'sf': sf_df['tmin'].quantile(0.10)}

    # Find the extreme events
    sj_df['is_hot_extreme'] = sj_df['tmax'] >= hot_thresholds['sj']
    sj_df['is_cold_extreme'] = sj_df['tmin'] <= cold_thresholds['sj']
    sf_df['is_hot_extreme'] = sf_df['tmax'] >= hot_thresholds['sf']
    sf_df['is_cold_extreme'] = sf_df['tmin'] <= cold_thresholds['sf']

    # Time periods for 'recent' and 'past'
    RECENT_YEARS = [2019, 2020, 2021, 2022, 2023]
    sj_df['is_recent'] = sj_df['year'].astype(int).isin(RECENT_YEARS)
    sf_df['is_recent'] = sf_df['year'].astype(int).isin(RECENT_YEARS)

    # Calculate proportions for the specific temp extreme
    event_col = f'is_{weather_type}_extreme'
    sj_stats = calculate_proportions(sj_df, event_col, 'is_recent')
    sf_stats = calculate_proportions(sf_df, event_col, 'is_recent')

    # Perform t-tests
    sj_ttest = perform_ttest(sj_df, event_col, 'is_recent')
    sf_ttest = perform_ttest(sf_df, event_col, 'is_recent')

    # Combine SJ and SF stats for plotting
    combined_stats = pd.concat(
        [
            sj_stats.assign(Region='SJ').reset_index(),
            sf_stats.assign(Region='SF').reset_index()
        ]
    )

    bar_colors_cold = {
        'SJ': '#3399FF',
        'SF': '#99CCFF'
    }
    bar_colors_hot = {
        'SJ': '#FF3333',
        'SF': '#FF9999'
    }
    
    # Generate grouped bar plot
    fig = px.bar(
        combined_stats,
        x = 'is_recent',
        y = 'proportion',
        color = 'Region',
        color_discrete_map = bar_colors_cold if weather_type == 'cold' else bar_colors_hot,
        barmode = 'group',
        labels = {
            'is_recent': 'Time Period',
            'proportion': 'Occurrence Percentage'
        },
        title = f'{weather_type.capitalize()} Extreme Events',
        text = 'proportion'
    )

    fig.update_traces(
        texttemplate = '%{text:.2%}',
        textposition = 'outside'
    )

    fig.update_layout(
        margin = {
            'r': 30,
            't': 30,
            'l': 30,
            'b': 30
        },
        legend_title_text = 'Region',
        xaxis = dict(
            tickmode = 'array',
            tickvals = [False, True],
            ticktext = ['Past', 'Recent']
        )
    )

    return (
        fig,
        f"{sj_ttest[1]:.4f}",  # SJ p-value
        f"{sf_ttest[1]:.4f}",  # SF p-value
    )

# LSTM - single step
@callback(
    [
        Output('lstm_train_score', 'children'),
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

    # joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_single_step_{file_specifier}.joblib'

    # lstm_results = None

    # # Load LSTM scores and predictions
    # if request_new_joblib:
    #     lstm_results = pred_lstm_single_step(region, file_specifier, shift)
    # else:
    #     if os.path.exists(joblib_filename_lstm_res):
    #         lstm_results = load(joblib_filename_lstm_res)

    lstm_results_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/lstm/{region}_lstm_single_step_{file_specifier}.joblib"
    lstm_results = load_joblib_from_github(lstm_results_url)

    train_score = lstm_results['train_score']
    val_score = lstm_results['val_score']
    test_score = lstm_results['test_score']
    scaler = lstm_results['scaler']
    fig = lstm_results['fig']

    lstm_train_score = train_score[1]
    lstm_val_score = val_score[1] 
    lstm_test_score = test_score[1]

    # averagekwh index
    averagekwh_index = 0

    # Extract the scale and mean used for 'averagekwh'
    averagekwh_scale = scaler.scale_[averagekwh_index]

    # Transform MAE back to to the original scale
    train_score_original = lstm_train_score * averagekwh_scale
    val_score_original = lstm_val_score * averagekwh_scale
    test_score_original = lstm_test_score * averagekwh_scale

    # Set the scores - add scaled train score later
    lstm_train_score = f'Train - Mean Absolute Error (MAE): {train_score_original:.3f}'
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score_original:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score_original:.3f}'

    return lstm_train_score, lstm_val_score, lstm_test_score, fig

# LSTM - multi step
@callback(
    [
        Output('lstm_train_score_multi', 'children'),
        Output('lstm_val_score_multi', 'children'),
        Output('lstm_test_score_multi', 'children'),
        Output('lstm_plot_multi', 'figure'),
    ],
    [
        Input('region_dropdown', 'value')
    ]
)
def update_lstm_multi_step(region):
    request_new_joblib = False  # Change to True for new lstm file
    file_specifier = 1
    shift = 12

    # joblib_filename_lstm_res = f'joblib_files/lstm/{region}_lstm_multi_step_{file_specifier}.joblib'

    # lstm_results = None

    # # Load LSTM scores and predictions
    # if request_new_joblib:
    #     lstm_results = pred_lstm_multi_step(region, file_specifier, shift)
    # else:
    #     if os.path.exists(joblib_filename_lstm_res):
    #         lstm_results = load(joblib_filename_lstm_res)

    lstm_results_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/lstm/{region}_lstm_multi_step_{file_specifier}.joblib"
    lstm_results = load_joblib_from_github(lstm_results_url)            
    
    train_score = lstm_results['train_score']
    val_score = lstm_results['val_score']
    test_score = lstm_results['test_score']
    scaler = lstm_results['scaler']
    fig = lstm_results['fig']


    lstm_train_score = train_score[1]
    lstm_val_score = val_score[1] 
    lstm_test_score = test_score[1]

    # averagekwh index
    averagekwh_index = 0

    # Extract the scale and mean used for 'averagekwh'
    averagekwh_scale = scaler.scale_[averagekwh_index]

    # Transform MAE back to to the original scale
    train_score_original = lstm_train_score * averagekwh_scale
    val_score_original = lstm_val_score * averagekwh_scale
    test_score_original = lstm_test_score * averagekwh_scale

    # Set the scores - add scaled train score later
    lstm_train_score = f'Train - Mean Absolute Error (MAE): {train_score_original:.3f}'
    lstm_val_score = f'Validation - Mean Absolute Error (MAE): {val_score_original:.3f}'
    lstm_test_score = f'Test - Mean Absolute Error (MAE): {test_score_original:.3f}'

    return lstm_train_score, lstm_val_score, lstm_test_score, fig



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

    # joblib_filename_sarima_res = f'joblib_files/sarima/{region}_sarima_{file_specifier}.joblib'

    # plot_title = 'San Jose' if region == 'sj' else 'San Francisco'

    # sarima_results = None

    # # Load SARIMA scores and predictions
    # if request_new_joblib:
    #     sarima_results = pred_sarima(region, file_specifier)
    # else:
    #     if os.path.exists(joblib_filename_sarima_res):
    #         sarima_results = load(joblib_filename_sarima_res)

    sarima_results_url = f"https://raw.githubusercontent.com/gabrield03/data_files/main/joblib_files/sarima/{region}_sarima_{file_specifier}.joblib"
    sarima_results = load_joblib_from_github(sarima_results_url)

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
        legend = dict(x = 0.8, y = 1.25),

        xaxis = dict(
            range = ['2023-05', '2024-07'],
            tickformat = '%Y-%m',
            tickvals = pd.date_range(start = '2023-05', end = '2024-07', freq = 'MS'),
        )
    )

    return mae_score, fig