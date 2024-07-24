import json
import logging
import re
import sqlite3

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots
import plotly.express as px

import utils as f
from annotation import Annotation
from Home import Home

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def _get_con():
    return sqlite3.connect('test.db', check_same_thread=False)

def _init_db(con):
    # ajouter un identifiant unique 
    con.execute(
        """CREATE TABLE IF NOT EXISTS transition(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annee INT,
            mois INT,
            jour INT,
            operation TEXT,
            montant REAL);"""
    )
    con.commit()
    con.execute(
        """CREATE TABLE IF NOT EXISTS budget(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annee INT,
            mois INT,
            jour INT,
            operation TEXT,
            montant REAL,
            categorie TEXT);"""
    )
    con.commit()
    logging.info('db initialised')

con = _get_con()
logging.info('connected to db')
_init_db(con)



# Init function callback
@callback(
    Output('content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/annotation':
        return Annotation()
    else:
        return Home()

# chargement des données dans la base
@callback(  
    [
        Output("upload-status", 'children'),
        Output("year_dropdown", 'options'),
        Output("year_dropdown", 'value'),
        Output('family_color', 'data')
    ],
    [
        Input('upload-data', 'contents')
    ],
    [
        State('upload-data', 'filename'),
        State('family_color', 'data')
    ]
)
def starter(list_of_contents, list_of_names, current_colors):
    if list_of_contents is not None:
        logging.info("gneu")
        try:
            dfs = [f.parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
            for d in dfs:
                r = process_data_input(d)
                
                # TODO:  add update form
                annees = get_annees(con)
                annees_dict = {an: an for an in annees}
                
                return [r, annees_dict, max(annees), current_colors] # return status of the import
        except Exception as e:
            print(f'error: {e}')
            return [None, None, None, current_colors]
    
    annees = get_annees(con)
    with open('famille.json', 'r') as f:
        famille = json.load(f)
    colors = {fam: px.colors.qualitative.Pastel[i] for i, fam in enumerate(set(famille.values())) if i < (len(px.colors.qualitative.Pastel)-1)}

    if not annees : 
        return [None, {}, None, colors]
    
    annees_dict = {an: an for an in annees}
    return [None, annees_dict, max(annees), colors]


@callback(
    [
        Output("pie_graph", "children")    
    ],
    [
        Input("year_dropdown", "value"),
        Input("months_selection", "value")
    ],
    State('family_color', 'data')
)
def get_pie_plot(selected_year, selected_months, family_colors):
    if selected_year and selected_months:
        df = get_data(con, selected_year)
        
        nrows = (len(selected_months)-1)//3+1
        ncols = len(selected_months)%3 or 3 if len(selected_months) <= 3 else 3
        
        # build specs for pie chart
        specs = [[{'type':'domain'}]*ncols for _ in range(nrows)]
        
        fig = make_subplots(rows=nrows, cols=ncols, specs=specs)
        for i, month in enumerate(selected_months):
            
            df_filtered = df[df.mois == month]
            lab, montant = credit_per_fam(df_filtered)
            
            # compute hoverlay data = details about aggregation
            meta_text = [str(df_filtered[df_filtered.categorie==l][["jour", "operation", "montant"]]).replace('\n', '<br>') for l in lab]
            
            fig.add_trace(go.Pie(
                name=month,
                labels=lab, values=montant, hole=.3,
                hovertemplate ='%{text}',
                text = meta_text,
                textinfo='value',
                color_discrete_map=family_colors
            ), row = (i)//3+1, col=(i+1)%3 or 3)
            
        return [dcc.Graph(figure=fig)]
    return [None]
    
@callback(
    [
        Output("graph_zone", "children"),
        Output("main_metrics", "children"),
        Output("waffle", "children")
    ],
    Input("year_dropdown", "value"), 
    State('family_color', 'data')
)
def get_plot(selected_year, family_color):
    if selected_year:
        
        df = get_data(con, selected_year)
        positif_x, positif_y, negatif_x, negatif_y = get_tot_per_month(df)
        
        # barplot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=positif_x, y=[abs(neg) for neg in negatif_y],
                        base=negatif_y,
                        marker_color='#F39B6D',
                        name='Dépenses',
        ))
        fig.add_trace(go.Bar(x=positif_x, y=positif_y,
            base=0,
            marker_color='#72C453',
            name='Revenues',
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.update_xaxes(insiderange=[0, 13], title="Date")

        # metrics
        fig2 = go.Figure()
        fig2.add_trace(go.Indicator(
            mode="delta",
            value=sum(positif_y),
            title="entrées",
            delta={"reference": 0},
            domain = {'row': 0, 'column': 0}
        ))
        fig2.add_trace(go.Indicator(
            mode="delta",
            value=sum(negatif_y),
            title="sorties",
            delta={"reference": 0},
            domain = {'row': 0, 'column': 1}
        ))
        fig2.add_trace(go.Indicator(
            mode="delta",
            value=sum(positif_y)+sum(negatif_y),
            title="reste",
            delta={"reference": 0},
            domain = {'row': 0, 'column': 2}
        ))
        fig2.update_layout(grid = {'rows': 1, 'columns': 3, 'pattern': "independent"})
        
        ## viz du bar plot gen horizontal
        y, x = credit_per_fam(df)
        fig3 = go.Figure()
        for xi, yi in zip(x, y):
            args = {}
            if yi in family_color:
                args = {"marker_color": family_color.get(yi)}
            fig3.add_trace(go.Bar(
                name=yi, x=[xi], y=[""],
                orientation='h',width=[0.3],
                **args
            ))
            
        reste_col = "green" if sum(positif_y)+sum(negatif_y) > 0 else "red"
        fig3.add_vline(x=sum(positif_y), line_width=3, line_dash="dash", line_color=reste_col, annotation_text=f"reste : {sum(positif_y)+sum(negatif_y):,.0f}€", annotation_position="top left",annotation_font_color=reste_col)
        fig3.update_layout(barmode='stack',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig3.update_xaxes(insiderange=[0, max(abs(sum(negatif_y))+1000, sum(positif_y))+1000], title="Date")
        
        return [dcc.Graph(figure=fig), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3)]
    return [None, None, None]



##############################
##############################
# ANNOTATION

# button for annotation creation
@callback(
    [
        Output("button_for_classification", "children"),
        Output("text_to_classify_div", 'children'),
        Output("current-data-div", "children")
    ],
    [
        # Input("auto_classify", "n_clicks"),
        Input('new_fam', 'value')
    ],
    [State("current-data", 'data')]
)
def annotation_load(new_fam, text_to_classify):
    df, famille = get_annoation_data(con)
    if df is None: 
        return [None, html.Div(dbc.Alert('Nothing to annotate', color="success", dismissable=True, fade=True), style={'display': 'flex', 'justify-content': 'space-between'}, id="text_to_classify"), None]
    if new_fam:
        logging.info(f"{text_to_classify} is {new_fam}")
        
        df = update_data(df, text_to_classify.get('first_id'), categorie=new_fam, famille=famille)
        
    first_id = df.id.tolist()[0]
    color = "green" if df[df.id == first_id].montant.tolist()[0] > 0 else "red"
    date = f"{df[df.id == first_id].jour.tolist()[0]}/{df[df.id == first_id].mois.tolist()[0]}/{df[df.id == first_id].annee.tolist()[0]}" 
    
    text = [
        html.Span(date, style={'float': 'left', 'margin-right': '20px'}),
        html.Span(df[df.id == first_id].operation.tolist()[0], style={'margin': '0 auto'}),
        html.Span(f"{df[df.id == first_id].montant.tolist()[0]:.2f}", style={'color': color, 'float': 'right'})
    ]
    data_store = {
        'first_id': first_id,
        # 'data': df.loc[first_id].to_dict()
    }
    
    new_component = [dbc.Button(fam, color="light", id={'type': 'dynamic-button', 'index': fam}, className="me-1 space") for fam in set(famille.values())]
    return [new_component, html.Div(text, style={'display': 'flex', 'justify-content': 'space-between'}, id="text_to_classify"), dcc.Store(data=data_store,id='current-data')]
    

# annotation process
@callback(
    [
        Output("text_to_classify", "children"),
        Output('current-data', 'data')
    ],
    Input({'type': 'dynamic-button', 'index': dash.dependencies.ALL}, 'n_clicks'),
    State("current-data", "data")
)
def update_annotation(n_clicks, text_to_classify):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    else:
        if sum(n or 0 for n in n_clicks) == 0 : 
            return dash.no_update
        
        df, famille = get_annoation_data(con)
        if df is None: 
            return [dbc.Alert('Nothing to annotate', color="success", dismissable=True, fade=True), {}]
        
        categorie = eval(ctx.triggered[0]['prop_id'].split('.')[0]).get('index')
        
        logging.info(f"{text_to_classify} is {categorie}")
        
        df = update_data(df, text_to_classify.get('first_id'), categorie=categorie, famille=famille)
        
        first_id = df.id.tolist()[0]
        logging.info(f'new firstid is {first_id}')
        
        color = "green" if df[df.id == first_id].montant.tolist()[0] > 0 else "red"
        date = f"{df[df.id == first_id].jour.tolist()[0]}/{df[df.id == first_id].mois.tolist()[0]}/{df[df.id == first_id].annee.tolist()[0]}" 
        
        text = [
            html.Span(date, style={'float': 'left', 'margin-right': '20px'}),
            html.Span(df[df.id == first_id].operation.tolist()[0], style={'margin': '0 auto'}),
            html.Span(f"{df[df.id == first_id].montant.tolist()[0]:.2f}", style={'color': color, 'float': 'right'})
        ]
        
        data_store = {
            'first_id': first_id,
            # 'data': df.loc[first_id].to_dict()
        }
        
        # update famille.json
        # update annotation.db
        # update budget.db
        
        return [text, data_store]
        
@callback(
    Output("autoclassify_status", "children"),
    Input("auto_classify", "n_clicks"),
    State("current-data", "data")
)
def autoclassify(n_clicks, text_to_classify):
    if n_clicks>0:
        df, _ = get_annoation_data(con)
        if df is None:
            return None
        # remove text to classify from df to not create duplicate
        df = df[df.id != text_to_classify.get('first_id')]
        r = auto_classify_processing(df, _con=con)
        return r
    else:
        return None
        
def credit_per_fam(df):
    negatif = df[df.montant<0].groupby(['categorie']).sum()
    lab = negatif.index.tolist()
    montant = abs(negatif.montant).tolist()
    return lab, montant

def preprocess_df(df):
    df = df.drop(columns="Unnamed: 5")

    # merge des 2 colonnes crédit / débit en 1 seule
    df['montant'] = np.where(~df.Débit.isna(), df.Débit, df.Crédit)
    df.montant = df.montant.str.strip().str.replace(',','.').astype(float)
    assert df.montant.dtype == float
    
    logging.info("montant updated")
    
    # suppression des colonnes inutiles
    df = df.drop(columns=['Date valeur', 'Débit', 'Crédit'])
    df.columns = ['date', 'operation', 'montant']

    # split de date en  annee INT, mois INT, jour INT,
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

    # Créer de nouvelles colonnes pour le jour, le mois et l'année
    df["jour"] = df["date"].dt.day
    df["mois"] = df["date"].dt.month
    df["annee"] = df["date"].dt.year
    df = df.drop(columns=["date"])
    return df

def auto_classify_processing(df, _con):
    with open('famille.json', 'r') as fi:
        famille = json.load(fi)
    
    # => stockage dans la BD "à annoter"
    df_budget, df_transition = _classify(df, famille)

    status_output = []

    if len(df_budget) > 0:
        # add categorie
        print(df_budget)
        
        df_budget['categorie'] = df_budget.apply(lambda x: famille.get(_parse_operation(x.operation)), axis=1)
        logging.info("categories added to budget")
        
        df_budget.drop("id", axis=1).to_sql(name='budget', con=_con, if_exists='append', index=False)
        
        logging.info(f'{len(df_budget)} lignes auto-classifiées')
        
        status_output.append(dbc.Alert(f'{len(df_budget)} lignes auto-classifiées', color="success", dismissable=True, fade=True))
        
        # suppression des id classifié de budget dans transition
        logging.info(f"executing: DELETE FROM transition WHERE id IN ({', '.join(map(str, df_budget.id.tolist()))});")
        _con.execute(f"DELETE FROM transition WHERE id IN ({', '.join(map(str, df_budget.id.tolist()))});")
        _con.commit()
    
        logging.info(f'{len(df_transition)} lignes restent à annoter')
        status_output.append(dbc.Alert(f'{len(df_transition)} lignes restent à annoter', color="info", dismissable=True, fade=True))
        
    else:
        status_output.append(dbc.Alert('No new family for data having to be annotated', color="warning", dismissable=True, fade=True))
    return status_output

def _parse_operation(operation: str):
    return re.sub(r"^(CARTE \d{2}\/\d{2} )", "", operation)

def _classify(df, famille):
    """
    Split the input dataframe into two sub-dataframes based on the following condition:
    - If the 'operation' value is in the values of the 'famille' dictionary, return True, except if 'operation' is 'course' and 'montant' % 10 == 0 or 'montant' % 10 == 1.
    """
    mask = df.apply(
        lambda x: _parse_operation(x.operation) in famille.keys() and not ((famille.get(_parse_operation(x.operation)) == 'course') and ((x.montant % 10 == 0) | (x.montant % 10 == 1)))
        , axis=1
    )
    df_budget = df[mask]
    df_transition = df[~mask]
    return df_budget, df_transition

def process_data_input(df):
    # split en transition et budget

    df = preprocess_df(df)
    
    with open('famille.json', 'r') as fi:
        famille = json.load(fi)
    
    # => stockage dans la BD "à annoter"
    df_budget, df_transition = _classify(df, famille)

    status_output = []

    if len(df_budget) > 0:
        # add categorie
        print(df_budget)
        df_budget['categorie'] = df_budget.apply(lambda x: famille.get(_parse_operation(x.operation)), axis=1)
        logging.info("categories added to budget")
        
        df_budget.to_sql(name='budget', con=con, if_exists='append', index=False)
        
        logging.info(f'{len(df_budget)} lignes déjà classifiées')
        status_output.append(dbc.Alert(f'{len(df_budget)} lignes déjà classifiées', color="info", dismissable=True, fade=True))
    
    if len(df_transition) > 0:
        df_transition.to_sql(name='transition', con=con, if_exists='append', index=False)
        logging.info(f'{len(df_transition)} lignes ajoutées à annoter')
        status_output.append(dbc.Alert(f'{len(df_transition)} lignes ajouté à annoter', color="info", dismissable=True, fade=True))
    
    
    status_output.append(dbc.Alert('Data Imported', color="success", dismissable=True, fade=True))
    return status_output

def get_tot_per_month(df):
    positif = df[df.montant>0].groupby(['mois']).sum()
    positif_x = positif.index.tolist()
    positif_y = positif.montant.tolist()
    
    negatif = df[df.montant<0].groupby(['mois']).sum()
    negatif_x = negatif.index.tolist()
    negatif_y = negatif.montant.tolist()
    return positif_x, positif_y, negatif_x, negatif_y

# @callback
def get_annees(_con):
    cur = _con.cursor()
    cur.execute("SELECT DISTINCT(annee) FROM budget;")
    return [annee[0] for annee in cur.fetchall()]

# @callback
def get_data(_con, annee):
    query = f"SELECT * FROM budget WHERE annee == {annee};"
    df = pd.read_sql_query(query, _con)
    logging.info('data imported')
    if df.shape[0] > 0 :
        logging.info(f'df imported successfully, shape: {df.shape}')
    else:
        logging.warning('nothing to annotate')
    return df

def get_annoation_data(_con):
    with open('famille.json', 'r') as f:
        famille = json.load(f)
    query = "SELECT * FROM transition;"
    df = pd.read_sql_query(query, _con)
    logging.info('data imported')
    if len(df) > 0 :
        logging.info(f'df imported successfully, shape: {df.shape}')
    else:
        logging.warning('nothing to annotate')
        return None, None
    return df, famille

# @callback
def has_data(_con):
    cur = _con.cursor()
    cur.execute("SELECT COUNT(*) FROM budget;")
    return cur.fetchone()[0] != 0

def update_data(df, first_id, categorie, famille):
    """update data to annotate and annotated db on click

    Args:
        df (pd.DataFrame): transition dataframe
        first_id (int): df id (generated id bdd side) of annotated row
        categorie (str): budget category
    """
    data = df[df.id == first_id]
    data['categorie'] = categorie
    new_df = df[df.id != first_id]
    
    update_famille(famille, operation=data.operation.tolist()[0], categorie=categorie)
    
    ## ajouter un bouton verbose pour voir les lignes insérées et supprimées
    # suppresion des données annotée dans la base de transition
    con.execute(f"DELETE FROM transition WHERE id == {data.id.values[0]};")
    con.commit()
    
    # comptage du nombre de données restantes à annoter
    cur = con.cursor()
    cur.execute("SELECT count(*) FROM transition;")
    nb_lines = cur.fetchall()
    
    # insertion des données dans la base propre
    data = data.drop(columns=['id'])
    
    data["annee"] = data["annee"].astype(int)
    data["mois"] = data["mois"].astype(int)
    data["jour"] = data["jour"].astype(int)
    
    data.to_sql(name='budget', con=con, if_exists='append', index=False)
    
    logging.info(f'{data.values} INSERTED INTO budget - category: {categorie}')    
    logging.info(f'{nb_lines[0][0]} lines to annotate')

    return new_df

def update_famille(famille, operation, categorie):
    if _parse_operation(operation) in famille.keys():
        return
    
    famille[_parse_operation(operation)] = categorie
    with open('famille.json', 'w') as f:
        json.dump(famille, f)
    logging.info('famille.json has been updated')
    