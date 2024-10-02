import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output

import numpy as np
from scipy.stats import gaussian_kde

dash.register_page(__name__)

# Import and clean data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

# Layout of the Dash app
layout = html.Div([
    html.H1('Passage about the visualizations created:'),
    
    html.Br(), html.Br(),

    ## Plots 1 - SJ Dataset
    html.H3('Visualizations for San Jose'),
    dcc.Dropdown(id='select_sj_option',
                 options=[
                     {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                     {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                     {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}],
                 multi=False,
                 value='averagekwh',
                 style={'width': '55%'}
                 ),
    
    html.Br(), html.Br(),
    
    # Split the layout for the plot and the description
    html.Div([
        dcc.Graph(
            id='sj_data',
            figure={},
            style={'flex': '1', 'height': '50vh', 'width': '100%', 'marginBottom': '200px'}
        ),
        html.Div(
            id='sj_output_container',
            children=[],
            style={'flex': '1', 'padding': '0px 100px'}
        ),
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'}),

    ## Plots 2 - SF Dataset
    html.H3('Visualizations for San Francisco'),
    dcc.Dropdown(id='select_sf_option',
                 options=[
                     {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                     {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                     {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}],
                 multi=False,
                 value='averagekwh',
                 style={'width': '55%'}
                 ),

    html.Br(), html.Br(),

    # Split the layout for the plot and the description
    html.Div([
        dcc.Graph(
            id='sf_data',
            figure={},
            style={'flex': '1', 'height': '50vh', 'width': '100%', 'marginBottom': '20px'}
        ),
        html.Div(
            id='sf_output_container',
            children=[],
            style={'flex': '1', 'padding': '0px 100px'}
        ),
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'}),

    html.Br(), html.Br(),
])


# Descriptions for each plot
sj_averagekwh = html.P(["Plot Description", html.Br(), html.Br(), "This plot shows the distribution of average monthly energy usage in San Jose (kWh)."])
sj_totalkwh = html.P(["Plot Description", html.Br(), html.Br(), "This plot displays the distribution of total energy usage in San Jose over the months (kWh)."])
sj_max_min_temp = html.P(["Plot Description", html.Br(), html.Br(), "This plot presents the average monthly maximum and minimum temperatures in San Jose (°F)."])

sf_averagekwh = html.P(["Plot Description", html.Br(), html.Br(), "This plot shows the distribution of average monthly energy usage in San Francisco (kWh)."])
sf_totalkwh = html.P(["Plot Description", html.Br(), html.Br(), "This plot displays the distribution of total energy usage in San Francisco over the months (kWh)."])
sf_max_min_temp = html.P(["Plot Description", html.Br(), html.Br(), "This plot presents the average monthly maximum and minimum temperatures in San Francisco (°F)."])

# Connect the Plotly graphs with Dash Components
## Plots 1 - SJ
@callback(
    [Output(component_id='sj_output_container', component_property='children'),
     Output(component_id='sj_data', component_property='figure')],
    [Input(component_id='select_sj_option', component_property='value')]
)
def update_sj_graph(option_selected):
    sj_dff = sj_df.copy()

    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SJ - 95110)'
            column = 'averagekwh'
            color = '#6600CC'
            kde_color = '#000000'
            x_range = [250, 625]

            container = sj_averagekwh

        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SJ - 95110)'
            column = 'totalkwh'
            color = '#4C9900'
            kde_color = '#000000'
            x_range = [1750000, 4000000]

            container = sj_totalkwh

        # Create histogram
        fig = px.histogram(sj_dff,
                           x=column,
                           nbins=40,
                           title=plot_title,
                           color_discrete_sequence=[color],
                           height=800)

        # Update the figure to set white outlines for the bars
        fig.update_traces(marker_line_color='white',
                          marker_line_width=1.5)

        # Calculate KDE
        kde = gaussian_kde(sj_dff[column].dropna())
        kde_range = np.linspace(sj_dff[column].min(), sj_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(go.Scatter(
            x=kde_range,
            y=kde(kde_range) * len(sj_dff[column]) * (sj_dff[column].max() - sj_dff[column].min()) / 40,
            mode='lines',
            name=f'{column} KDE',
            line=dict(color=kde_color, width=2, dash='dash')
        ))

        # Adjust layout
        fig.update_layout(
            title=plot_title,
            barmode='overlay',
            yaxis_title='Frequency',
            xaxis_range=x_range,
            height=800,
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
            )
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title='Average Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title='Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SJ - 95110)'
        container = sj_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(go.Histogram(
            x=sj_dff['tmax'],
            nbinsx=40,
            name='Max Temp (tmax)',
            marker_color='#8B0000',
            opacity=0.75
        ))

        # tmin histogram
        fig.add_trace(go.Histogram(
            x=sj_dff['tmin'],
            nbinsx=40,
            name='Min Temp (tmin)',
            marker_color='#003FFF',
            opacity=0.75
        ))

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sj_dff['tmax'].dropna())
        tmax_range = np.linspace(sj_dff['tmax'].min(), sj_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(go.Scatter(
            x=tmax_range,
            y=tmax_kde(tmax_range) * len(sj_dff['tmax']) * (sj_dff['tmax'].max() - sj_dff['tmax'].min()) / 40,
            mode='lines',
            name='Max Temp KDE',
            line=dict(color='#FFCCCB', width=2, dash='dash')
        ))

        # Calculate tmin KDE
        tmin_kde = gaussian_kde(sj_dff['tmin'].dropna())
        tmin_range = np.linspace(sj_dff['tmin'].min(), sj_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(go.Scatter(
            x=tmin_range,
            y=tmin_kde(tmin_range) * len(sj_dff['tmin']) * (sj_dff['tmin'].max() - sj_dff['tmin'].min()) / 40,
            mode='lines',
            name='Min Temp KDE',
            line=dict(color='#A5E5FF', width=2, dash='dash')
        ))

        # Update layout for the plot
        fig.update_layout(
            title=plot_title,
            barmode='overlay',
            xaxis_title='Temperature (F)',
            yaxis_title='Frequency',
            xaxis_range=[30, 90],
            height=800,
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
            )
        )

        fig.update_traces(marker_line_color='white',
                          marker_line_width=1.5)

        return container, fig


## Plots 2 - SF
@callback(
    [Output(component_id='sf_output_container', component_property='children'),
     Output(component_id='sf_data', component_property='figure')],
    [Input(component_id='select_sf_option', component_property='value')]
)
def update_sf_graph(option_selected):
    sf_dff = sf_df.copy()

    container = ''
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SF - 94102)'
            column = 'averagekwh'
            color = '#CC00CC'
            kde_color = '#000000'
            x_range = [200, 450]

            container = sf_averagekwh

        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SF - 94102)'
            column = 'totalkwh'
            color = '#999900'
            kde_color = '#000000'
            x_range = [2500000, 5000000]

            container = sf_totalkwh

        # Create histogram
        fig = px.histogram(sf_dff,
                           x=column,
                           nbins=40,
                           title=plot_title,
                           color_discrete_sequence=[color],
                           height=800)

        # Update the figure to set white outlines for the bars
        fig.update_traces(marker_line_color='white',
                          marker_line_width=1.5)

        # Calculate KDE
        kde = gaussian_kde(sf_dff[column].dropna())
        kde_range = np.linspace(sf_dff[column].min(), sf_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(go.Scatter(
            x=kde_range,
            y=kde(kde_range) * len(sf_dff[column]) * (sf_dff[column].max() - sf_dff[column].min()) / 40,
            mode='lines',
            name=f'{column} KDE',
            line=dict(color=kde_color, width=2, dash='dash')
        ))

        # Adjust layout
        fig.update_layout(
            title=plot_title,
            barmode='overlay',
            yaxis_title='Frequency',
            xaxis_range=x_range,
            height=800,
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
            )
        )

        if column == 'averagekwh':
            fig.update_layout(xaxis_title='Average Energy Usage (kWh)')
        else:
            fig.update_layout(xaxis_title='Total Energy Usage (kWh)')

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SF - 94102)'
        container = sf_max_min_temp

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(go.Histogram(
            x=sf_dff['tmax'],
            nbinsx=40,
            name='Max Temp (tmax)',
            marker_color='#8B0000',
            opacity=0.75
        ))

        # tmin histogram
        fig.add_trace(go.Histogram(
            x=sf_dff['tmin'],
            nbinsx=40,
            name='Min Temp (tmin)',
            marker_color='#003FFF',
            opacity=0.75
        ))

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sf_dff['tmax'].dropna())
        tmax_range = np.linspace(sf_dff['tmax'].min(), sf_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(go.Scatter(
            x=tmax_range,
            y=tmax_kde(tmax_range) * len(sf_dff['tmax']) * (sf_dff['tmax'].max() - sf_dff['tmax'].min()) / 40,
            mode='lines',
            name='Max Temp KDE',
            line=dict(color='#FFCCCB', width=2, dash='dash')
        ))

        # Calculate tmin KDE
        tmin_kde = gaussian_kde(sf_dff['tmin'].dropna())
        tmin_range = np.linspace(sf_dff['tmin'].min(), sf_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(go.Scatter(
            x=tmin_range,
            y=tmin_kde(tmin_range) * len(sf_dff['tmin']) * (sf_dff['tmin'].max() - sf_dff['tmin'].min()) / 40,
            mode='lines',
            name='Min Temp KDE',
            line=dict(color='#A5E5FF', width=2, dash='dash')
        ))

        # Update layout for the plot
        fig.update_layout(
            title=plot_title,
            barmode='overlay',
            xaxis_title='Temperature (F)',
            yaxis_title='Frequency',
            xaxis_range=[30, 90],
            height=800,
            legend=dict(
                x=1,
                y=1,
                xanchor='right',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='black',
                borderwidth=1,
            )
        )

        fig.update_traces(marker_line_color='white',
                          marker_line_width=1.5)

        return container, fig