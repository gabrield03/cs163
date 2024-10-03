import dash
from dash import html, dcc, callback
from dash.dependencies import Output, Input
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd

dash.register_page(__name__, path = '/')

layout = html.Div(

    children = [
        html.Br(), html.Br(),
        html.H1('Effects of Weather on Energy Consumption in the Bay Area', style = {'text-align': 'center'}),

        # Image styles
        html.Div(
            style = {'maxWidth': '800px', 'margin': '0 auto'},
            children = [
                html.Div(
                    style = {'textAlign': 'center', 'margin': '20px'},
                    children = [
                        html.Img(
                            src = '/assets/bayareascenic.jpg',
                            style = {
                                'width': '100%',
                                'maxWidth': '800px',
                                'height': 'auto',
                                'borderRadius': '10px',
                            }
                        )
                    ]
                )
            ]
        ),
        
        html.H2("Introduction", style = {'text-align': 'center'}),
            
        html.P([
            "This project explores the impact of climate change on energy consumption in the California Bay Area, with a focus on San Jose and San Francisco. \
            By analyzing historical weather data and energy usage trends, I aim to identify key weather factors, such as extreme temperatures, that influence \
            electricity demand. Ultimately, this project hopes to shed light on how shifts in climate can affect local energy consumption, with the potential \
            to apply these findings to other regions.",

            html.Br(), html.Br(),
            "Explore the various pages to seehow weather patterns correlate with energy usage in each region!"
        ], style = {'text-align': 'justify', 'font-size': '20px', 'margin': '20px 7%', 'word-break': 'keep-all'}),

        html.Br(), html.Br(),

        #dcc.Interval(id='interval', interval=1*1000, n_intervals=0),

        html.Div([
            dcc.Graph(
                id = 'ca_data',
                figure = {},
                style = {'flex': '1', 'height': '50%', 'width': '50%'}
            ),
        ]),
    ],

)

@callback(
    Output(component_id = 'ca_data', component_property = 'figure'),
    Input(component_id='ca_data', component_property='id')
)
def update_ca_graph(graph_id):
    triggered = dash.callback_context.triggered

    if not triggered:
        df_sample = pd.read_csv(
            'https://raw.githubusercontent.com/plotly/datasets/master/minoritymajority.csv'
        )
        df_sample_r = df_sample[df_sample['STNAME'] == 'California']

        # Get all FIPS values
        all_fips = df_sample_r['FIPS'].tolist()
        
        # Create a list for values where only Santa Clara and San Francisco have actual population values
        values = []
        for fips in all_fips:
            if fips == "06085" or fips == "06075":
                values.append(df_sample_r.loc[df_sample_r['FIPS'] == fips, 'TOT_POP'].values[0])
            else:
                values.append(0)  # Use 0 or some default value for counties that aren't being highlighted

        # Define the color scale
        colorscale = [
            'rgb(193, 193, 193)',  # Default color
            'rgb(239, 239, 239)',
            'rgb(195, 196, 222)',
            'rgb(144, 148, 194)',
            'rgb(101, 104, 168)',
            'rgb(65, 53, 132)'
        ]

        # Create the choropleth map
        fig = ff.create_choropleth(
            fips=all_fips, values=values, colorscale=colorscale,
            scope=['CA'],
            county_outline={'color': 'rgb(255, 255, 255)', 'width': 0.5},
            legend_title='California Counties',
            title='Population of Selected California Counties'
        )

        return fig

