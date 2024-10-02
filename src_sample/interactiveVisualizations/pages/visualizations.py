import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, dash_table, callback
from dash.dependencies import Input, Output

import numpy as np
from scipy.stats import gaussian_kde

dash.register_page(__name__)

# Import and clean data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

regions_combined = pd.concat([sj_df, sf_df])

# Layout of the Dash app
layout = html.Div([
    html.H1('This is the Visualizations page'),

    html.Br(), html.Br(),

    ## Plots 1 - SJ Dataset
    html.H3('Data Visualizations for San Jose'),
    dcc.Dropdown(id = 'select_sj_option',
                options = [
                    {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                    {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                    {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}],
                multi = False,
                value = 'averagekwh',
                style = {'width': '55%'}
                ),
    html.Div(id = 'sj_output_container', children = []),
    html.Br(),

    dcc.Graph(id = 'sj_data', figure = {}),

    html.Br(), html.Br(),


    ## Plots 2 - SF Dataset
    html.H3('Data Visualizations for San Francisco'),
    dcc.Dropdown(id = 'select_sf_option',
                options = [
                    {'label': 'Avg Energy Usage (kWh)', 'value': 'averagekwh'},
                    {'label': 'Total Energy Usage (kWh)', 'value': 'totalkwh'},
                    {'label': 'Average Monthly Max and Min Temperatures', 'value': 'max_min_temp'}],
                multi = False,
                value = 'averagekwh',
                style = {'width': '55%'}
                ),
    html.Div(id = 'sf_output_container', children = []),
    html.Br(),

    dcc.Graph(id = 'sf_data', figure = {}),

    html.Br(), html.Br(),


    # ## Plots 3 - Chloropleth SJ and SF energy
    # html.H3('Energy Usage in the Bay Area (San Jose & San Francisco)'),
    
    # dcc.Graph(id='choropleth'),
    # dcc.Slider(
    #     id='year-month-slider',
    #     min=2013,
    #     max=2024,
    #     value=2023,
    #     marks={i: str(i) for i in range(2013, 2025)},
    #     step=1
    # ),
    # dcc.Dropdown(
    #     id='month-dropdown',
    #     options=[{'label': month, 'value': i} for i, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)],
    #     value=1,  # Default to January
    #     clearable=False
    # ),

    # html.Br(), html.Br(),


])

# Connect the Plotly graphs with Dash Components
## Plots 1 - SJ
@callback(
    [Output(component_id = 'sj_output_container', component_property = 'children'),
     Output(component_id = 'sj_data', component_property = 'figure')],
    [Input(component_id = 'select_sj_option', component_property = 'value')]
)

def update_sj_graph(option_selected):
    sj_dff = sj_df.copy()

    container = f'The user selected: {option_selected}'
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SJ - 95110)'
            column = 'averagekwh'
            color = '#6600CC'
            kde_color = '#000000'
            x_range = [250, 625]
        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SJ - 95110)'
            column = 'totalkwh'
            color = '#4C9900'
            kde_color = '#000000'
            x_range = [1750000, 4000000]

        # Create histogram
        fig = px.histogram(sj_dff,
                        x = column,
                        nbins = 40,
                        title = plot_title,
                        color_discrete_sequence = [color],
                        height = 800)

        # Update the figure to set white outlines for the bars
        fig.update_traces(marker_line_color = 'white',
                        marker_line_width = 1.5)

        # Calculate KDE
        kde = gaussian_kde(sj_dff[column].dropna())
        kde_range = np.linspace(sj_dff[column].min(), sj_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(go.Scatter(
            x = kde_range,
            y = kde(kde_range) * len(sj_dff[column]) * (sj_dff[column].max() - sj_dff[column].min()) / 40,
            mode = 'lines',
            name = f'{column} KDE',
            line = dict(color = kde_color, width = 2, dash = 'dash')
        ))

        # Adjust the x-axis range
        fig.update_layout(xaxis_range=x_range)

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SJ - 95110)'

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(go.Histogram(
            x = sj_dff['tmax'],
            nbinsx = 40,
            name = 'Max Temp (tmax)',
            marker_color = '#8B0000',
            opacity = 0.75
        ))

        # tmin histogram
        fig.add_trace(go.Histogram(
            x = sj_dff['tmin'],
            nbinsx = 40,
            name = 'Min Temp (tmin)',
            marker_color = '#003FFF',
            opacity = 0.75
        ))

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sj_dff['tmax'].dropna())
        tmax_range = np.linspace(sj_dff['tmax'].min(), sj_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(go.Scatter(
            x = tmax_range,
            y = tmax_kde(tmax_range) * len(sj_dff['tmax']) * (sj_dff['tmax'].max() - sj_dff['tmax'].min()) / 40,
            mode = 'lines',
            name = 'Max Temp KDE',
            line = dict(color = '#FFCCCB', width = 2, dash = 'dash')
        ))

        # Calculate tmin KDE
        tmin_kde = gaussian_kde(sj_dff['tmin'].dropna())
        tmin_range = np.linspace(sj_dff['tmin'].min(), sj_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(go.Scatter(
            x = tmin_range,
            y = tmin_kde(tmin_range) * len(sj_dff['tmin']) * (sj_dff['tmin'].max() - sj_dff['tmin'].min()) / 40,
            mode = 'lines',
            name = 'Min Temp KDE',
            line = dict(color = '#A5E5FF', width = 2, dash = 'dash')
        ))

        # Update layout for the plot
        fig.update_layout(
            title = plot_title,
            barmode = 'overlay',
            xaxis_title = 'Temperature (F)',
            yaxis_title = 'Frequency',
            xaxis_range = [30, 90],
            height = 800
        )

        fig.update_traces(marker_line_color = 'white',
                        marker_line_width = 1.5)
        
        return container, fig
    


## Plots 2 - SF
@callback(
    [Output(component_id = 'sf_output_container', component_property = 'children'),
     Output(component_id = 'sf_data', component_property = 'figure')],
    [Input(component_id = 'select_sf_option', component_property = 'value')]
)

def update_sf_graph(option_selected):
    sf_dff = sf_df.copy()

    container = f'The user selected: {option_selected}'
    plot_title = ''

    # Plot energy
    if 'average' in option_selected or 'total' in option_selected:
        if 'average' in option_selected:
            plot_title = 'Distribution of Monthly Average Energy Usage (SF - 94102)'
            column = 'averagekwh'
            color = '#CC00CC'
            kde_color = '#000000'
            x_range = [200, 450]
        else:
            plot_title = 'Distribution of Monthly Total Energy Usage (SF - 94102)'
            column = 'totalkwh'
            color = '#999900'
            kde_color = '#000000'
            x_range = [2500000, 5000000]

        # Create histogram
        fig = px.histogram(sf_dff,
                        x = column,
                        nbins = 40,
                        title = plot_title,
                        color_discrete_sequence = [color],
                        height = 800)

        # Update the figure to set white outlines for the bars
        fig.update_traces(marker_line_color = 'white',
                        marker_line_width = 1.5)

        # Calculate KDE
        kde = gaussian_kde(sf_dff[column].dropna())
        kde_range = np.linspace(sf_dff[column].min(), sf_dff[column].max(), 100)

        # Add KDE line
        fig.add_trace(go.Scatter(
            x = kde_range,
            y = kde(kde_range) * len(sf_dff[column]) * (sf_dff[column].max() - sf_dff[column].min()) / 40,
            mode = 'lines',
            name = f'{column} KDE',
            line = dict(color = kde_color, width = 2, dash = 'dash')
        ))

        # Adjust the x-axis range
        fig.update_layout(xaxis_range=x_range)

        return container, fig

    # Plot weather
    else:
        plot_title = 'Average Monthly Max and Min Temperatures (SF - 94102)'

        fig = go.Figure()

        # tmax histogram
        fig.add_trace(go.Histogram(
            x = sf_dff['tmax'],
            nbinsx = 40,
            name = 'Max Temp (tmax)',
            marker_color = '#8B0000',
            opacity = 0.75
        ))

        # tmin histogram
        fig.add_trace(go.Histogram(
            x = sf_dff['tmin'],
            nbinsx = 40,
            name = 'Min Temp (tmin)',
            marker_color = '#003FFF',
            opacity = 0.75
        ))

        # Calculate tmax KDE
        tmax_kde = gaussian_kde(sf_dff['tmax'].dropna())
        tmax_range = np.linspace(sf_dff['tmax'].min(), sf_dff['tmax'].max(), 100)

        # Add tmax KDE line
        fig.add_trace(go.Scatter(
            x = tmax_range,
            y = tmax_kde(tmax_range) * len(sf_dff['tmax']) * (sf_dff['tmax'].max() - sf_dff['tmax'].min()) / 40,
            mode = 'lines',
            name = 'Max Temp KDE',
            line = dict(color = '#FFCCCB', width = 2, dash = 'dash')
        ))

        # Calculate tmin KDE
        tmin_kde = gaussian_kde(sf_dff['tmin'].dropna())
        tmin_range = np.linspace(sf_dff['tmin'].min(), sf_dff['tmin'].max(), 100)

        # Add tmin KDE line
        fig.add_trace(go.Scatter(
            x = tmin_range,
            y = tmin_kde(tmin_range) * len(sf_dff['tmin']) * (sf_dff['tmin'].max() - sf_dff['tmin'].min()) / 40,
            mode = 'lines',
            name = 'Min Temp KDE',
            line = dict(color = '#A5E5FF', width = 2, dash = 'dash')
        ))

        # Update layout for the plot
        fig.update_layout(
            title = plot_title,
            barmode = 'overlay',
            xaxis_title = 'Temperature (F)',
            yaxis_title = 'Frequency',
            xaxis_range = [40, 80],
            height = 800
        )

        fig.update_traces(marker_line_color = 'white',
                        marker_line_width = 1.5)
        
        return container, fig



# # Callback for the choropleth
# @.callback(
#     Output('choropleth', 'figure'),
#     [Input('year-month-slider', 'value'),
#      Input('month-dropdown', 'value')]
# )
# def update_choropleth(selected_year, selected_month):
#     sf_dff = sf_df.copy()
#     print(sf_dff.dtypes)

#     month_mapping = {
#         1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May',
#         6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct',
#         11: 'Nov', 12: 'Dec'
#     }
#     selected_month_name = month_mapping[selected_month]

#     # Filter data based on year and month
#     filtered_df = sf_dff[
#         (sf_dff['year'] == selected_year) & 
#         (sf_dff['month'] == selected_month_name)
#     ]

#     # Create choropleth map
#     fig = px.choropleth(
#         filtered_df,
#         geojson='https://gist.githubusercontent.com/cdolek/d08cac2fa3f6338d84ea/raw/ebe3d2a4eda405775a860d251974e1f08cbe4f48/SanFrancisco.Neighborhoods.json',
#         locations='zipcode',  # Ensure this column exists in your DataFrame
#         color='averagekwh',  # Color by average energy usage
#         featureidkey='id',  # Use 'id' to match the GeoJSON structure
#         hover_name='zipcode',
#         hover_data=['averagekwh', 'totalkwh'],
#         title=f'Energy Usage in San Francisco Neighborhoods ({selected_month_name} {selected_year})',
#         color_continuous_scale='Viridis'
#     )

#     fig.update_geos(visible=False)
#     fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

#     return fig











# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)
