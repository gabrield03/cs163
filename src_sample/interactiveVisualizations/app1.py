import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True)

# # original Layout
# app.layout = html.Div([
#     html.Div([
#         html.Div(
#             dcc.Link(f"{page['name']}", href=page["relative_path"])
#         ) for page in dash.page_registry.values()
#     ]),
#     dash.page_container
# ])

# Horizontal Navigation Bard
app.layout = html.Div([
    html.Div(
        className='nav-bar',
        children=[
            dcc.Link(
                f"{page['name']}",
                href=page["relative_path"],
                style={
                    'margin': '10px',
                    'textDecoration': 'none',
                    'color': 'blue',
                    'fontSize': '20px',
                }
            ) for page in dash.page_registry.values()
        ],
        style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
    ),
    dash.page_container
])


# # Button Style Links
# app.layout = html.Div([
#     html.Div(
#         className='nav-bar',
#         children=[
#             dcc.Link(
#                 f"{page['name']}",
#                 href=page["relative_path"],
#                 style={
#                     'margin': '10px',
#                     'padding': '10px 15px',
#                     'backgroundColor': '#D0D0D0',
#                     'color': 'black',
#                     'borderRadius': '5px',
#                     'textDecoration': 'none',
#                     'display': 'inline-block',
#                     'fontSize': '20px',
#                 }
#             ) for page in dash.page_registry.values()
#         ],
#         style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
#     ),
#     dash.page_container
# ])


# # Vertical Sidebar Navigation
# app.layout = html.Div([
#     html.Div(
#         className='sidebar',
#         style={
#             'width': '200px',
#             'height': '100vh',
#             'backgroundColor': '#f8f9fa',
#             'position': 'fixed',
#             'padding': '20px',
#             'boxShadow': '2px 0 5px rgba(0,0,0,0.1)'
#         },
#         children=[
#             dcc.Link(
#                 f"{page['name']}",
#                 href=page["relative_path"],
#                 style={
#                     'display': 'block',
#                     'padding': '10px',
#                     'margin': '10px 0',
#                     'textDecoration': 'none',
#                     'color': '#007BFF'
#                 }
#             ) for page in dash.page_registry.values()
#         ]
#     ),
#     html.Div(
#         style={'marginLeft': '220px'},  # Margin to accommodate the sidebar
#         children=[dash.page_container]
#     )
# ])

# # Tabs Navigation
# app.layout = html.Div([
#     dcc.Tabs(
#         id='tabs',
#         value=list(dash.page_registry.values())[0]['relative_path'],  # Convert to list for indexing
#         children=[
#             dcc.Tab(
#                 label=page['name'],
#                 value=page["relative_path"],
#                 style={'margin': '10px'},
#                 selected_style={'fontWeight': 'bold', 'color': 'blue'}
#             ) for page in dash.page_registry.values()
#         ]
#     ),
#     dash.page_container
# ])


if __name__ == '__main__':
    app.run(debug=True)