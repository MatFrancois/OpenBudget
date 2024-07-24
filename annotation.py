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
                        html.Div(html.H1('Annotation zone'), className="row space"),
                        html.Div(dcc.Store(id='current-data'), id="current-data-div"),

                        html.Div(p_zone, className="row"),
                        
                        
                        html.Div(html.Div(id="text_to_classify", style={'display': 'flex', 'justify-content': 'space-between'}), className="row", id="text_to_classify_div"),
                        html.Div(html.Div(className="d-grid gap-2 d-md-block", id="button_for_classification")),
                        
                        dbc.Input(id="new_fam", placeholder="Nouvelle famille", type="text", debounce=True, className="space"),

                        
                        html.Div(dbc.Button("AutoClassify", color="info", id="auto_classify", n_clicks=0), className="row space"),
                        html.Div(className="row space",id="autoclassify_status")
                        # dbc.Input(id="hello", debounce=True),
                        # html.Div(html.P(id="output"), id="coco"),
                        # html.Div(id='dynamic-content')
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
