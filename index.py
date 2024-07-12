# from dash import dcc, html
from dash import dcc, html
import callbacks

from server import app

app.layout = html.Div(
   [
    #    html.Div(
    #        [
    #            dcc.Store(id='table1_inv')
    #        ],
    #        style={'display': 'none'}
    #    ),
       dcc.Location(id = 'url', refresh=False),
       
       html.Div(id = 'content')

   ] 
)

