import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import Dash, dcc, html, callback
from dash.dependencies import Input, Output

import numpy as np
from scipy.stats import gaussian_kde

# FOLIUM DEPENDENCIES 
import folium
#from jobs.routes import *
#from jobs import db

dash.register_page(__name__)

# NEED PREPROCESSING PIPELINE
# Import data
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

sj_df['month_numeric'] = sj_df['month'].map({
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 
    'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 
    'Nov': 11, 'Dec': 12
})

# FOLIUM - base map
# SJ_COORDINATES = (37.35411, -121.95524)
# sj_map = folium.Map(location = SJ_COORDINATES, tiles = "Stamen Toner", attr = "something", zoom_start = 10)
# sj_map_html = sj_map._repr_html_()

# ### DEBUG
# # Create a simple folium map of the US
# def create_map():
#     us_map = folium.Map(location=[37.0902, -95.7129], zoom_start=4)

#     # Add markers for major US cities
#     cities = {
#         "New York": [40.7128, -74.0060],
#         "Los Angeles": [34.0522, -118.2437],
#         "Chicago": [41.8781, -87.6298],
#         "Houston": [29.7604, -95.3698],
#         "Miami": [25.7617, -80.1918]
#     }

#     for city, coord in cities.items():
#         folium.Marker(location=coord, popup=city).add_to(us_map)

#     # Save to a temporary file and return the HTML
#     temp_map = "us_map.html"
#     us_map.save(temp_map)
#     return temp_map

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
            style={'flex': '1', 'height': '50%', 'width': '65%'}
        ),
        html.Div(
            id='sj_output_container',
            children=[],
            style={'flex': '1', 'padding-left': '5%'}
        ),
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'margin-bottom': '5%'}),

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
            style={'flex': '1', 'height': '50%', 'width': '65%'}
        ),
        html.Div(
            id='sf_output_container',
            children=[],
            style={'flex': '1', 'padding-left': '5%'}
        ),
    ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%', 'margin-bottom': '5%'}),

    html.Br(), html.Br(),


    # Plot 3 - Folium SJ
    html.H3('Map of San Jose'),
    # Folium map placeholder
    html.Div(id='sj_map', children=[]),

    # Year and month sliders
    html.Div([
        html.Label('Select Year'),
        dcc.Slider(
            id='year_slider',
            min=sj_df['year'].min(),
            max=sj_df['year'].max(),
            value=sj_df['year'].min(),
            marks={str(year): str(year) for year in range(sj_df['year'].min(), sj_df['year'].max()+1)},
            step=None
        ),
        html.Label('Select Month'),
        dcc.Slider(
            id='month_slider',
            min=1,
            max=12,
            value=1,
            marks={i: str(i) for i in range(1, 13)},
            step=None
        ),
    ], style={'width': '80%', 'margin': 'auto', 'padding': '20px'}),

    html.Br(), html.Br(),


    html.H3("San Jose Correlation Heatmap"),

    # Year Slider
    dcc.Slider(
        id='year-slider',
        min=sj_df['year'].min(),
        max=sj_df['year'].max(),
        value=sj_df['year'].min(),
        marks={str(year): str(year) for year in sj_df['year'].unique()},
        step=None
    ),
    
    # Month Slider
    dcc.Slider(
        id='month-slider',
        min=1,
        max=12,
        value=1,
        marks={i: month for i, month in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)},
        step=1
    ),
    
    # Graph output
    dcc.Graph(id='heatmap'),



    # ### DEBUG
    # html.Div([
    #     html.H1("US States Map"),
    #     html.Iframe(id='map', srcDoc=open(create_map(), 'r').read(), width='100%', height='600'),
    # ]),
    html.Br(), html.Br(),
])


# Descriptions for each plot
sj_averagekwh = html.P(["Plot Description:", html.Br(), html.Br(), "This plot shows the distribution of average monthly energy usage in San Jose (kWh)."])
sj_totalkwh = html.P(["Plot Description:", html.Br(), html.Br(), "This plot displays the distribution of total energy usage in San Jose over the months (kWh)."])
sj_max_min_temp = html.P(["Plot Description:", html.Br(), html.Br(), "This plot presents the average monthly maximum and minimum temperatures in San Jose (°F)."])

sf_averagekwh = html.P(["Plot Description:", html.Br(), html.Br(), "This plot shows the distribution of average monthly energy usage in San Francisco (kWh)."])
sf_totalkwh = html.P(["Plot Description:", html.Br(), html.Br(), "This plot displays the distribution of total energy usage in San Francisco over the months (kWh)."])
sf_max_min_temp = html.P(["Plot Description:", html.Br(), html.Br(), "This plot presents the average monthly maximum and minimum temperatures in San Francisco (°F)."])

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
                           #height=800
                           )

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
            #height=800,
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
            #height=800,
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
                           #height=800
                           )

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
            #height=800,
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
            nbinsx=30,
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
            y=tmin_kde(tmin_range) * len(sf_dff['tmin']) * (sf_dff['tmin'].max() - sf_dff['tmin'].min()) / 30,
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
            #height=800,
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


# SJ Heatmap
# Callback to update heatmap
@callback(
    Output('heatmap', 'figure'),
    [Input('year-slider', 'value'),
     Input('month-slider', 'value')]
)
def update_heatmap(selected_year, selected_month):
    # Filter data based on selected year and month
    filtered_df = sj_df[(sj_df['year'] == selected_year) & (sj_df['month_numeric'] == selected_month)]
    
    # Select relevant columns for correlation (e.g., energy usage, tmax, tmin)
    correlation_data = filtered_df[['averagekwh', 'totalkwh', 'tmax', 'tmin']].corr()
    
    # Create heatmap
    heatmap = go.Figure(data=go.Heatmap(
        z=correlation_data.values,
        x=correlation_data.columns,
        y=correlation_data.columns,
        colorscale='Viridis'
    ))
    
    heatmap.update_layout(
        title=f"Correlation Heatmap for Year: {selected_year}, Month: {selected_month}",
        xaxis_title="Variables",
        yaxis_title="Variables"
    )
    
    return heatmap


# ### DEBUG
# # Define the callback for updating the Folium map
# @callback(
#     Output(component_id='sj_map', component_property='children'),
#     [Input(component_id='year_slider', component_property='value'),
#      Input(component_id='month_slider', component_property='value')]
# )
# def update_sj_map(selected_year, selected_month):
#     # Filter data based on selected year and month
#     filtered_df = sj_df[(sj_df['year'] == selected_year) & (sj_df['month'] == selected_month)]

#     # San Jose coordinates
#     SJ_COORDINATES = (37.35411, -121.95524)
#     sj_map = folium.Map(location=SJ_COORDINATES, tiles="Stamen Toner", attr = "something", zoom_start=10)

#     # Plot each zip code's energy consumption on the map
#     for _, row in filtered_df.iterrows():
#         zip_code = row['zipcode']
#         total_energy = row['totalkwh']  # Assuming 'totalkwh' is the energy consumption column

#         # Adjust coordinates as per actual zip code locations in San Jose (you'll need to add lat/lon for zip codes)
#         zip_coordinates = (row['latitude'], row['longitude'])  # Ensure latitude/longitude columns are in the dataset

#         folium.CircleMarker(
#             location=zip_coordinates,
#             radius=total_energy / 100000,  # Adjust radius scaling based on energy values
#             color='blue',
#             fill=True,
#             fill_color='blue',
#             fill_opacity=0.6,
#             popup=f"Zip Code: {zip_code}<br>Total Energy: {total_energy} kWh"
#         ).add_to(sj_map)

#     return [html.Iframe(srcDoc=sj_map._repr_html_(), width='100%', height='600')]