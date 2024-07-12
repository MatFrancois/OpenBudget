import json
import logging
import re
import sqlite3

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from plotly.subplots import make_subplots

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


# à modifier
with open('famille.json', 'r') as fi:
    famille = json.load(fi)

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
    ],
    [
        Input('upload-data', 'contents')
    ],
    [
        State('upload-data', 'filename')
    ]
)
def starter(list_of_contents, list_of_names):
    if list_of_contents is not None:
        try:
            dfs = [f.parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]
            for d in dfs:
                r = _process_data_input(d)
                
                # TODO:  add update form
                annees = get_annees(con)
                annees = {an: an for an in annees}
                
                return [r, annees] # return status of the import
        except Exception as e:
            print(f'error: {e}')
            return [None, None]
    annees = get_annees(con)
    annees = {an: an for an in annees}
    return [None, annees]


@callback(
    [
        Output("pie_graph", "children")    
    ],
    [
        Input("year_dropdown", "value"),
        Input("months_selection", "value")
    ]
)
def get_pie_plot(selected_year, selected_months):
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
                textinfo='value'
            ), row = (i)//3+1, col=(i+1)%3 or 3)
            
        return [dcc.Graph(figure=fig)]
    return [None]
    
@callback(
    [
        Output("graph_zone", "children"),
        Output("main_metrics", "children"),
        Output("waffle", "children")
    ],
    Input("year_dropdown", "value")
)
def get_plot(selected_year):
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
            fig3.add_trace(go.Bar(
                name=yi, x=[xi], y=[""],
                orientation='h',width=[0.3]
            ))
            
        reste_col = "green" if sum(positif_y)+sum(negatif_y) > 0 else "red"
        fig3.add_vline(x=sum(positif_y), line_width=3, line_dash="dash", line_color=reste_col, annotation_text=f"reste : {sum(positif_y)+sum(negatif_y):,.0f}€", annotation_position="top left",annotation_font_color=reste_col)
        fig3.update_layout(barmode='stack',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
        fig3.update_xaxes(insiderange=[0, max(sum(negatif_y), sum(positif_y))+1000], title="Date")
        
        return [dcc.Graph(figure=fig), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3)]
    return [None, None, None]

def credit_per_fam(df):
    negatif = df[df.montant<0].groupby(['categorie']).sum()
    lab = negatif.index.tolist()
    montant = abs(negatif.montant).tolist()
    return lab, montant

def _process_data_input(df):
    # + drop la dernière col
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

    def _parse_operation(operation: str):
        return re.sub(r"^(CARTE \d{2}\/\d{2} )", "", operation)

    # split en transition et budget
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

    # => stockage dans la BD "à annoter"
    df_budget, df_transition = _classify(df, famille)

    status_output = []

    if len(df_budget) > 0:
        # add categorie
        print(df_budget)
        df_budget['categorie'] = df_budget.apply(lambda x: famille.get(x.operation), axis=1)
        logging.info("categories added to budget")
        
        df_budget.to_sql(name='budget', con=con, if_exists='append', index=False)
        
        logging.info(f'{len(df_budget)} lignes déjà classifiées')
        status_output.append(dbc.Alert(f'{len(df_budget)} lignes déjà classifiées', color="info"))
        # st.info(f'{len(df_budget)} lignes déjà classifiées')
    
    if len(df_transition) > 0:
        df_transition.to_sql(name='transition', con=con, if_exists='append', index=False)
        logging.info(f'{len(df_transition)} lignes ajoutées à annoter')
        status_output.append(dbc.Alert(f'{len(df_transition)} lignes ajouté à annoter', color="info"))
        # st.info(f'{len(df_transition)} lignes ajouté à annoter')
    
    
    status_output.append(dbc.Alert('Data Imported', color="success"))
    return status_output
    # st.success('Data Imported', icon="✅")

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
    return df, famille

# @callback
def has_data(_con):
    cur = _con.cursor()
    cur.execute("SELECT COUNT(*) FROM budget;")
    return cur.fetchone()[0] != 0