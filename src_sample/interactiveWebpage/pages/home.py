from dash import html, dcc, Input, Output, clientside_callback
import dash_bootstrap_components as dbc
import plotly.express as px

from joblib import load

# Define the layout for the home page
home_front = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Energy & Weather Impact Analysis',
                        style = {
                            'color': 'white',
                            'textAlign': 'center',
                            'fontSize': '3vw',
                            'fontweight': 'bold',
                            'height': '100%',
                            'text-shadow': '2px 2px 4px #000000',
                        },
                    ),
                    width = 12,
                ),
            ],
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.Div("Max Temperature", id="word-max-temp", className="fade-word"),
                            html.Div("Min Temperature", id="word-min-temp", className="fade-word"),
                            html.Div("Precipitation", id="word-precipitation", className="fade-word"),
                            html.Div("Wind Speed", id="word-wind-speed", className="fade-word"),
                            html.Div("Energy Consumption", id="word-energy-consumption", className="fade-word")
                        ],
                        style = {
                            'color': 'white',
                            'textAlign': 'center',
                            'fontSize': '3vw',
                            'fontWeight': 'bold',
                            'height': '100%',
                            "fontFamily": "monospace",
                            'display': 'flex',
                            'flexDirection': 'column',
                            'justifyContent': 'center',
                            'alignItems': 'center',
                        },
                    ),
                    width=12,
                ),
            ],
        ),
                
        # Video background
        html.Div(
            children = [
                html.Video(
                    id = 'home-front-video',
                    src = '/assets/sky-video2.mp4',
                    autoPlay = True,
                    loop = False,
                    muted = True,
                    controls = False,
                    style = {
                        'width': '100vw',
                        'maxWidth': '100vw',
                        'height': '75vh',
                        'objectFit': 'cover',
                        'position': 'fixed',
                        'top': '0',
                        'left': '0',
                        'zIndex': '-1',
                        'margin': 0,
                    },
                ),
            ],
        ),
        # Fade to black bottom section
        html.Div(
            id = 'home-front-gradient',
            style = {
                'height': '35vh',
                'background': 'linear-gradient(to bottom, rgba(0,0,0,0), rgba(0,0,0,.3), rgba(0,0,0,.6), rgba(0,0,0,0.9), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1), rgba(0,0,0,1))',
                'width': '100vw',
                'position': 'fixed',
                'bottom': '0',
                'left': '0',
                'zIndex': '-1',
            },
        ),

        dcc.Interval(id="page-load-interval", interval=5000, n_intervals=0),
    ],
    style={
        'height': '100vh',
        #'position': 'relative',
    },
)


# SHAP Dot plot function
def shap_parallel_coord_plot(loc):
    base_df = load(f'joblib_files/base_data/{loc}_combined.joblib')

    plot_title = 'San Jose' if loc == 'sj' else 'San Francisco'

    # Which columns for sj_df? All or only ones matching sf_df?
    dim_cols = [
        'zipcode', 'year', 'totalcustomers', 'totalkwh', 'averagekwh',
        'month-numeric', 'prcp', 'tmax', 'tmin'
    ]
    
    fig = px.parallel_coordinates(
        base_df,
        color = 'averagekwh',
        # dimensions = base_df.columns,
        dimensions = dim_cols,
        color_continuous_scale=px.colors.diverging.Tealrose,
    )

    fig.update_layout(
        title = f'Data Distributions for {plot_title}',
        xaxis_title = 'Features',
        yaxis_title = 'Values',
    )

    return fig

# SHAP Dot Plots
shap_parallel_coord_plot_section = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Graph(
                            id = 'shap_parallel_coord_plot',
                            figure = shap_parallel_coord_plot('sj'),
                        ),
                    ],
                    width = 12,
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
                            id = 'shap_parallel_coord_plot',
                            figure = shap_parallel_coord_plot('sf'),
                        ),
                    ],
                    width = 12,
                    align = 'center',
                ),
            ],
            className = 'mb-3',
        ),
    ],
)



layout = dbc.Container(
    [
        home_front,
        shap_parallel_coord_plot_section,
        html.Div(
            id = 'home-page-content',
        ),
    ],
    fluid = True
)

# Client-side callback for fading in and out words
from dash import Input, Output, clientside_callback

# Client-side callback for fading in and out words
clientside_callback(
    """
    function(n_intervals) {
        const words = [
            "word-max-temp", 
            "word-min-temp", 
            "word-precipitation", 
            "word-wind-speed", 
            "word-energy-consumption"
        ];

        // Hide all words
        words.forEach(function(id) {
            const element = document.getElementById(id);
            if (element) {
                element.classList.remove("visible");
            }
        });

        // Show the next word
        const index = n_intervals % words.length;
        const nextWord = document.getElementById(words[index]);
        if (nextWord) {
            nextWord.classList.add("visible");
        }

        // Return the n_intervals to continue counting for the next cycle
        return n_intervals;
    }
    """,
    Output('page-load-interval', 'n_intervals'),
    Input('page-load-interval', 'n_intervals')
)
