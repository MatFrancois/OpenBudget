import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from navbar import Navbar

nav = Navbar('Home', '/home')

p_zone = dcc.Markdown('''
Upload data = send new data to db

Dashboard beside = current year's data, can chose
''')
upload_data = dcc.Upload(
    id='upload-data',
    children=html.Div(
        'Drag & Drop or Select Files',
    ),
    multiple=True,
    className="space text-position-upload",
)

def Home():
    return html.Div(
        children = [
            nav,
            
            html.Div([
                html.Div(className='col-3'),
                html.Div([
                    html.H1('Home', className='row'),
                    
                    html.Div(dcc.Store(id='family_color')),
                    html.Div(p_zone, className='row'),
                    html.Div(upload_data, className='row'),
                    html.Div(html.Div(id="upload-status"), className='row'),
                    html.Div([
                        dcc.Dropdown(id="year_dropdown", placeholder="Année de référence")
                    ], className='row space'),
                    html.Div(className='row', id="graph_zone"),
                    html.Div(className='row', id="main_metrics"),
                    html.Div(className='row', id="waffle"),
                ], className='col-6'),
                html.Div(className='col-3'),
                html.Div(className='row', id="pie_graph"),
            ],className='row'),

            html.Div(
                id = 'table_zone'
            ),
            
        ]
    )
