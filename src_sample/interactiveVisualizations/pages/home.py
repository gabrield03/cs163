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

animated_plot_1 = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            id='animated-plot-tabs',
                            value='Jan',
                            children=[
                                dcc.Tab(label = 'January', value = 'Jan'),
                                dcc.Tab(label = 'February', value = 'Feb'),
                                dcc.Tab(label = 'March', value = 'Mar'),
                                dcc.Tab(label = 'April', value = 'Apr'),
                                dcc.Tab(label = 'May', value = 'May'),
                                dcc.Tab(label = 'June', value = 'Jun'),
                                dcc.Tab(label = 'July', value = 'Jul'),
                                dcc.Tab(label = 'August', value = 'Aug'),
                                dcc.Tab(label = 'September', value = 'Sep'),
                                dcc.Tab(label = 'October', value = 'Oct'),
                                dcc.Tab(label = 'November', value = 'Nov'),
                                dcc.Tab(label = 'December', value = 'Dec'),
                            ],
                        ),
                    ],
                    className='mb-2',
                ),
            ],
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id='combined_energy_line_by_mo',
                            figure={},
                            style={
                                'width': '95vh',
                                'height': '50vh'
                            },
                        ),
                    ],
                    width=7,
                ),
                dbc.Col(
                    [
                        html.Div(
                            id='animated_output_container',
                            children=[],
                            style={
                                'color': '#0f0f0f',
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
    className='mb-5',
)

layout = dbc.Container(
    [
        home_header,
        home_picture,
        intro_text,
        intro_content,
        animated_plot_1,
    ],
    fluid = True
)


# DEBUG
sj_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SJ_Combined.csv')
sf_df = pd.read_csv('https://raw.githubusercontent.com/gabrield03/cs163/refs/heads/main/src_sample/interactiveVisualizations/Data/SF_Combined.csv')

sj_df['region'] = 'San Jose'
sf_df['region'] = 'San Francisco'

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

sf_filled_df = sf_df.copy()

for month in months:
    # Filter data for the current month and the year 2014
    sf_month_2014 = sf_filled_df[(sf_filled_df['month'] == month) & (sf_filled_df['year'] == 2014)].copy()
    
    # Modify the year to 2013 for this data
    sf_month_2014['year'] = 2013
    
    # Append this "filler" data to the dataframe
    sf_filled_df = pd.concat([sf_filled_df, sf_month_2014], ignore_index=True)

# Sort the dataframe by year to maintain the chronological order
sf_filled_df.sort_values(by='year', inplace=True)

### Plot Descriptions ###
comb_energy_jan = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of January."])
comb_energy_feb = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of February."])
comb_energy_mar = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of March."])
comb_energy_apr = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of April."])
comb_energy_may = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of May."])
comb_energy_jun = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of June."])
comb_energy_jul = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of July."])
comb_energy_aug = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of August."])
comb_energy_sep = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of September."])
comb_energy_oct = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of October."])
comb_energy_nov = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of November."])
comb_energy_dec = html.P([html.Br(), html.Br(), "Plot Description:", html.Br(), html.Br(), "This animated plot shows the average energy usage in both SJ and SF regions for the month of December."])



@callback(
    [Output('animated_output_container', 'children'),
     Output('combined_energy_line_by_mo', 'figure')],
    Input('animated-plot-tabs', 'value')  # Take the selected tab (month) as input
)
def update_energy_line_plot(selected_month):
    container = ''
    if selected_month == 'Jan':
        container = comb_energy_jan
    elif selected_month == 'Feb':
        container = comb_energy_feb
    elif selected_month == 'Mar':
        container = comb_energy_mar
    elif selected_month == 'Apr':
        container = comb_energy_apr
    elif selected_month == 'May':
        container = comb_energy_may
    elif selected_month == 'Jun':
        container = comb_energy_jun
    elif selected_month == 'Jul':
        container = comb_energy_jul
    elif selected_month == 'Aug':
        container = comb_energy_aug
    elif selected_month == 'Sep':
        container = comb_energy_sep
    elif selected_month == 'Oct':
        container = comb_energy_oct
    elif selected_month == 'Nov':
        container = comb_energy_nov
    elif selected_month == 'Dec':
        container = comb_energy_dec
    

    # Filter data for the selected month
    sf_filtered = sf_filled_df[sf_filled_df['month'] == selected_month]
    sj_filtered = sj_df[sj_df['month'] == selected_month]

    # Create figure for the selected month
    fig = go.Figure()

    # Plot the full lines for both regions
    fig.add_trace(go.Scatter(x=sf_filtered['year-month'], y=sf_filtered['averagekwh'],
                             mode="lines", name="San Francisco",
                             line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=sj_filtered['year-month'], y=sj_filtered['averagekwh'],
                             mode="lines", name="San Jose",
                             line=dict(color="green")))

    # Add a starting point for each region's moving dot
    fig.add_trace(go.Scatter(x=[sf_filtered['year-month'].values[0]],
                             y=[sf_filtered['averagekwh'].values[0]],
                             mode="markers",
                             marker=dict(color="blue", size=10),
                             name="SF moving point"))

    fig.add_trace(go.Scatter(x=[sj_filtered['year-month'].values[0]],
                             y=[sj_filtered['averagekwh'].values[0]],
                             mode="markers",
                             marker=dict(color="green", size=10),
                             name="SJ moving point"))

    # Set layout properties for the plot
    fig.update_layout(
        title=f"Average Energy Usage (kWh) for {selected_month} (Animated)",
        xaxis_title="Year-Month",
        yaxis_title="Avg Energy Usage (kWh)",
        xaxis_range=['2012', '2025'],  # Set the x-axis range to end in 2025
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None, {
                                            "frame": {"duration": 500, "redraw": False},
                                            "fromcurrent": True,
                                            "transition": {"duration": 500}}])])]
    )

    # Create frames for the animation
    frames = []
    max_len = min(len(sf_filtered), len(sj_filtered))  # Ensure we loop over the shortest data length

    for i in range(0, max_len):
        frame_data = [
            # Keep the lines static
            go.Scatter(x=sf_filtered['year-month'], y=sf_filtered['averagekwh'],
                       mode="lines", line=dict(color="blue")),
            go.Scatter(x=sj_filtered['year-month'], y=sj_filtered['averagekwh'],
                       mode="lines", line=dict(color="green")),

            # Update the position of the moving dots for both regions
            go.Scatter(x=[sf_filtered['year-month'].values[i]],
                       y=[sf_filtered['averagekwh'].values[i]],
                       mode="markers", marker=dict(color="blue", size=10)),
            go.Scatter(x=[sj_filtered['year-month'].values[i]],
                       y=[sj_filtered['averagekwh'].values[i]],
                       mode="markers", marker=dict(color="green", size=10))
        ]
        frames.append(go.Frame(data=frame_data, name=str(i)))

    fig.update(frames=frames)

    return container, fig