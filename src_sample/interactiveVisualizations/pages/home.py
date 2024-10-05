import dash
from dash import html, dcc, callback
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc

import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


home_header = html.Div(
    [
        html.H1(
            'Effects of Weather on Energy Consumption in the Bay Area',
            className = 'text-center mt-5',
            #style = {'text-align': 'center'}),
        ),
    ],
)

home_picture = html.Div(
    [
        html.Div(
            style = {
                'maxWidth': '800px',
                'margin': '0 auto'
            },
            children = [
                html.Div(
                    style = {
                        'textAlign': 'center',
                        'margin': '20px'
                    },
                    children = [
                        html.Img(
                            src = '/assets/bayareascenic.jpg',
                            style = {
                                'width': '100%',
                                'maxWidth': '800px',
                                'height': 'auto',
                                'borderRadius': '10px',
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)

intro_text = html.Div(
    [
        html.H2(
            "Introduction",
            style = {
                'text-align': 'center'
            },
        ),
    ],
)

intro_content = html.Div(
    [
        html.P(
            [
                "This project explores the impact of climate change on energy consumption in the California Bay Area, with a focus on San Jose and San Francisco. \
                By analyzing historical weather data and energy usage trends, I aim to identify key weather factors, such as extreme temperatures, that influence \
                electricity demand. Ultimately, this project hopes to shed light on how shifts in climate can affect local energy consumption, with the potential \
                to apply these findings to other regions.",

                html.Br(), html.Br(),
                "Explore the various pages to seehow weather patterns correlate with energy usage in each region!"
            ],
            style = {
                'text-align': 'justify',
                'font-size': '20px',
                'margin': '20px 7%',
                'word-break': 'keep-all'
            },
        ),
    ],
)

animated_graph = html.Div(
    [
        dbc.Row(
            [
                dcc.Graph(
                    id = 'combined_energy_line_by_mo',
                    figure = {},
                )
            ]
        )
    ]
)

layout = dbc.Container(
    [
        home_header,
        home_picture,
        intro_text,
        intro_content,
        animated_graph,
    ],
    fluid = True
)

@callback(
    Output(component_id='combined_energy_line_by_mo', component_property='figure'),
    Input(component_id='combined_energy_line_by_mo', component_property='id')
)
# def animated_energy_line_plot(graph_id):
#     # Load data and add region col
#     sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
#     sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')
#     sj_df['region'] = 'San Jose'
#     sf_df['region'] = 'San Francisco'

#     sf_march = sf_df[sf_df['month'] == 'Mar']
#     sj_march = sj_df[sj_df['month'] == 'Mar']

#     combined_march = pd.concat([sf_march, sj_march])

#     # Create the line plot
#     fig = px.line(
#         combined_march,
#         x='year-month',
#         y='averagekwh',
#         color='region',
#         markers=True,
#         title="Average Energy Usage (kWh) for March in San Francisco and San Jose",
#         labels={'averagekwh': 'Avg Energy Usage (kWh)', 'year-month': 'Year-Month'}
#     )

#     return fig

## ATTEMPT 1
def animated_energy_line_plot(graph_id):
    # Load data and add region column
    sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
    sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')
    
    sj_df['region'] = 'San Jose'
    sf_df['region'] = 'San Francisco'

    # Filter for March data
    sf_march = sf_df[sf_df['month'] == 'Mar']
    sj_march = sj_df[sj_df['month'] == 'Mar']

    march_2014_data = sf_march[sf_march['year'] == 2014].copy()

    # Modify the year to 2013
    march_2014_data['year'] = 2013




    # Append the new record to sf_march
    sf_march = pd.concat([sf_march, march_2014_data], ignore_index=True)
    sf_march.sort_values(by = 'year', inplace = True)


    print(sf_march['year-month'].values[0])


    # Create empty figure for lines
    fig = go.Figure()

    # Plot the full lines for both regions
    fig.add_trace(go.Scatter(x=sf_march['year-month'], y=sf_march['averagekwh'],
                             mode="lines", name="San Francisco",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=sj_march['year-month'], y=sj_march['averagekwh'],
                             mode="lines", name="San Jose",
                             line=dict(color="green")))

    # Add a starting point for each region's moving dot
    fig.add_trace(go.Scatter(x=[sf_march['year-month'].values[0]],
                             y=[sf_march['averagekwh'].values[0]],
                             mode="markers", marker=dict(color="blue", size=10),
                             name="SF moving point"))
    fig.add_trace(go.Scatter(x=[sj_march['year-month'].values[0]],
                             y=[sj_march['averagekwh'].values[0]],
                             mode="markers", marker=dict(color="green", size=10),
                             name="SJ moving point"))

    # Set layout properties for the plot
    fig.update_layout(
        title="Average Energy Usage (kWh) for March (Animated)",
        xaxis_title="Year-Month",
        yaxis_title="Avg Energy Usage (kWh)",
        width=800,
        height=600,
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {
                                            "frame": {"duration": 1000, "redraw": False},
                                            "fromcurrent": True,
                                            "transition": {"duration": 500}}])])]
    )

    # Create frames for the animation
    frames = []
    max_len = min(len(sf_march), len(sj_march))  # Ensure we loop over the shortest data length

    for i in range(1, max_len):
        # Update the position of the moving points along the lines for each region
        frame_data = [
            # Keep the lines static
            go.Scatter(x=sf_march['year-month'], y=sf_march['averagekwh'],
                       mode="lines", line=dict(color="blue")),
            go.Scatter(x=sj_march['year-month'], y=sj_march['averagekwh'],
                       mode="lines", line=dict(color="green")),
            
            # Update the position of the moving dots for both regions
            go.Scatter(x=[sf_march['year-month'].values[i]], 
                       y=[sf_march['averagekwh'].values[i]],
                       mode="markers", marker=dict(color="blue", size=10)),
            go.Scatter(x=[sj_march['year-month'].values[i]], 
                       y=[sj_march['averagekwh'].values[i]],
                       mode="markers", marker=dict(color="green", size=10))
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.update(frames=frames)

    return fig


### ATTEMPT 2
# def animated_energy_line_plot(graph_id):
#     # Load data and add region column
#     sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
#     sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')
    
#     sj_df['region'] = 'San Jose'
#     sf_df['region'] = 'San Francisco'

#     # Filter for March data
#     sf_march = sf_df[sf_df['month'] == 'Mar']
#     sj_march = sj_df[sj_df['month'] == 'Mar']
    
#     # Ensure data is sorted by year-month
#     sf_march = sf_march.sort_values('year-month')
#     sj_march = sj_march.sort_values('year-month')

#     # Create empty figure for the animation
#     fig = go.Figure()

#     # Add initial positions for the moving dots (but no lines yet)
#     fig.add_trace(go.Scatter(x=[sf_march['year-month'].values[0]],
#                              y=[sf_march['averagekwh'].values[0]],
#                              mode="markers", marker=dict(color="blue", size=10),
#                              name="SF moving point"))
#     fig.add_trace(go.Scatter(x=[sj_march['year-month'].values[0]],
#                              y=[sj_march['averagekwh'].values[0]],
#                              mode="markers", marker=dict(color="green", size=10),
#                              name="SJ moving point"))

#     # Set layout properties for the plot
#     fig.update_layout(
#         title="Average Energy Usage (kWh) for March (Animated)",
#         xaxis_title="Year-Month",
#         yaxis_title="Avg Energy Usage (kWh)",
#         width=800,
#         height=600,
#         updatemenus=[dict(type="buttons",
#                           buttons=[dict(label="Play",
#                                         method="animate",
#                                         args=[None, {
#                                             "frame": {"duration": 1000, "redraw": False},
#                                             "fromcurrent": True,
#                                             "transition": {"duration": 500}}])])]
#     )

#     # Create frames for the animation
#     frames = []
#     sf_len = len(sf_march)
#     sj_len = len(sj_march)
#     max_len = max(sf_len, sj_len)  # Ensure we account for the longest series

#     for i in range(max_len):
#         frame_data = []

#         # San Francisco: add line segment and moving point when i is valid for SF
#         if i < sf_len and sf_march['year'].iloc[i] >= 2014:  # SF data starts from 2014
#             frame_data.append(go.Scatter(x=sf_march['year-month'].iloc[:i+1],  # Reveal SF line up to current point
#                                          y=sf_march['averagekwh'].iloc[:i+1],
#                                          mode="lines", line=dict(color="blue"),
#                                          name="San Francisco"))
#             frame_data.append(go.Scatter(x=[sf_march['year-month'].iloc[i]],
#                                          y=[sf_march['averagekwh'].iloc[i]],
#                                          mode="markers", marker=dict(color="blue", size=10),
#                                          name="SF moving point"))

#         # San Jose: add line segment and moving point when i is valid for SJ
#         if i < sj_len and sj_march['year'].iloc[i] >= 2013:  # SJ data starts from 2013
#             frame_data.append(go.Scatter(x=sj_march['year-month'].iloc[:i+1],  # Reveal SJ line up to current point
#                                          y=sj_march['averagekwh'].iloc[:i+1],
#                                          mode="lines", line=dict(color="green"),
#                                          name="San Jose"))
#             frame_data.append(go.Scatter(x=[sj_march['year-month'].iloc[i]],
#                                          y=[sj_march['averagekwh'].iloc[i]],
#                                          mode="markers", marker=dict(color="green", size=10),
#                                          name="SJ moving point"))

#         # Append frame data for this step
#         frames.append(go.Frame(data=frame_data, name=str(i)))

#     # Add the frames to the figure
#     fig.update(frames=frames)

#     return fig







### ATTEMPT 3
# def animated_energy_line_plot(graph_id):
#     # Load data and add region column
#     sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
#     sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')
    
#     sj_df['region'] = 'San Jose'
#     sf_df['region'] = 'San Francisco'

#     # Filter for March data
#     sf_march = sf_df[sf_df['month'] == 'Mar']
#     sj_march = sj_df[sj_df['month'] == 'Mar']

#     # Sync the two datasets by padding SF data to start at 2013
#     years = pd.date_range(start="2013-03-01", end="2024-03-01", freq='YS').strftime('%Y-%m').tolist()
#     sf_march = sf_march.set_index('year-month').reindex(years).reset_index()
#     sj_march = sj_march.set_index('year-month').reindex(years).reset_index()

#     # Create empty figure for lines
#     fig = go.Figure()

#     # Plot initial empty lines for both regions
#     fig.add_trace(go.Scatter(x=[], y=[],
#                              mode="lines", name="San Francisco",
#                              line=dict(color="blue")))
#     fig.add_trace(go.Scatter(x=[], y=[],
#                              mode="lines", name="San Jose",
#                              line=dict(color="green")))

#     # Add a starting point for each region's moving dot
#     fig.add_trace(go.Scatter(x=[sj_march['year-month'].values[0]],
#                              y=[sj_march['averagekwh'].values[0]],
#                              mode="markers", marker=dict(color="green", size=10),
#                              name="SJ moving point"))
#     fig.add_trace(go.Scatter(x=[sf_march['year-month'].values[0]],
#                              y=[sf_march['averagekwh'].values[0]],
#                              mode="markers", marker=dict(color="blue", size=10),
#                              name="SF moving point"))

#     # Set layout properties for the plot
#     fig.update_layout(
#         title="Average Energy Usage (kWh) for March (Animated)",
#         xaxis_title="Year-Month",
#         yaxis_title="Avg Energy Usage (kWh)",
#         width=800,
#         height=600,
#         updatemenus=[dict(type="buttons",
#                           buttons=[dict(label="Play",
#                                         method="animate",
#                                         args=[None, {
#                                             "frame": {"duration": 1000, "redraw": False},
#                                             "fromcurrent": True,
#                                             "transition": {"duration": 500}}])])]
#     )

#     # Create frames for the animation
#     frames = []
#     max_len = len(years)

#     for i in range(1, max_len):
#         # Update the position of the moving points along the lines for each region
#         frame_data = [
#             # Progressively reveal the San Francisco line
#             go.Scatter(x=sf_march['year-month'].values[:i], y=sf_march['averagekwh'].values[:i],
#                        mode="lines", line=dict(color="blue")),
#             # Progressively reveal the San Jose line
#             go.Scatter(x=sj_march['year-month'].values[:i], y=sj_march['averagekwh'].values[:i],
#                        mode="lines", line=dict(color="green")),
            
#             # Update the position of the moving dots for both regions
#             go.Scatter(x=[sf_march['year-month'].values[i]], 
#                        y=[sf_march['averagekwh'].values[i]],
#                        mode="markers", marker=dict(color="blue", size=10)),
#             go.Scatter(x=[sj_march['year-month'].values[i]], 
#                        y=[sj_march['averagekwh'].values[i]],
#                        mode="markers", marker=dict(color="green", size=10))
#         ]
#         frames.append(go.Frame(data=frame_data, name=str(i)))

#     fig.update(frames=frames)

#     return fig