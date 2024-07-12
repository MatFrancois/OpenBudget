import dash_bootstrap_components as dbc
from dash import Dash

### Création de l'objet de type dash
app = Dash(__name__,
    external_stylesheets=[dbc.themes.MINTY],
    suppress_callback_exceptions = True
)
app.title = 'My budget'
