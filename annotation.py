import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from navbar import Navbar

nav = Navbar('Annotation', '/annotation')
p_zone = dcc.Markdown('''
hello
''')

def Annotation():
    return html.Div(
        children = [
            nav,
            
            html.Div([
                html.Div(className='col-3'),
                html.Div(
                    [
                        html.Div(html.H1('Annotation zone'), className="row"),
                        html.Div(p_zone, className="row"),
                        html.Div(dbc.Button("AutoClassify", color="primary", className="me-1"), className="row"),
                        html.Div(className="row", id="text_to_classify"),
                        html.Div(className="row", id="button_for_classification"),
                        # html.Div([dbc.Alert('Data Imported', color="info")]),
                    ],
                    className='col-6'
                ),
                
                html.Div(className='col-3'),
            ],className='row'),

            html.Div(
                id = 'table_zone'
            ),
            
        ]
    )
