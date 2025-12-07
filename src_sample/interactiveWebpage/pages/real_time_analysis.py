from dash import html
import dash_bootstrap_components as dbc


# sj_df = pd.DataFrame()
# df_df = pd.DataFrame()

# if os.path.exists('joblib_files/base_data/sj_combined.joblib'):
#     sj_df = load('joblib_files/base_data/sj_combined.joblib')

# if os.path.exists('joblib_files/base_data/sf_combined.joblib'):
#     sf_df = load('joblib_files/base_data/sf_combined.joblib')

rt_analysis = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        'Real-Time Analysis',
                    ),
                    className = 'text-center mb-5 mt-5',
                    width = 12,
                    style = {
                        'font-size': '50px',
                        'height': '100%',
                        'text-shadow': '2px 2px 4px #000000',
                    },
                ),
            ],
            className = 'mb-5',
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            'This page will gather real-time weather data and present energy consumption predictions.',
                            style = {
                                'text-align': 'center',
                                'font-size': '40px',
                                'font-variant': 'small-caps',
                                'text-shadow': '2px 2px 4px #000000'
                            },
                        ),
                    ],
                ),
            ],
        ),
    ],
)




layout = dbc.Container(
    [
        rt_analysis,
    ],
    fluid = True,
)