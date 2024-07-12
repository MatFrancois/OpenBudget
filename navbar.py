import dash_bootstrap_components as dbc

### Fonction qui créé la barre de navigation
def Navbar(page, lien):

    return dbc.NavbarSimple(
        children=[
            # Affichage du nom de la page active
            dbc.NavItem(
                dbc.NavLink(page, href=lien, className="main_menu"),
                style={'display': 'none'},
            ),
            # Menu déroulant permettant de séléctionner la page
            dbc.NavItem(dbc.NavLink("Home", href="/home")),
            dbc.NavItem(dbc.NavLink("Annotation", href="/annotation")),
        ],
        brand=page,
        sticky="top",
        # brand_style={"color": "white"},
        color="primary",
        dark=True,
    )
