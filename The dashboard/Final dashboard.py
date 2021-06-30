""" Dashboard of the prediction for the graydon
@author: Zhou Yang, Zheng Bin, Fang AnBing
@date: 26/05/2021
"""
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import download_stocks
from download_stocks import symbol_dict  # Custom import

    
import dash                                # pip install dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc    # pip install dash-bootstrap-components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries # pip install alpha-vantage
import flask
import glob
import os
import base64
from dash.exceptions import PreventUpdate


import dash
from dash_html_components import Br, Div
from dash_table import DataTable
from dash_table.Format import Format, Scheme, Trim
###############################################################################
# Parameters:
###############################################################################
# reference: https://github.com/plotly/dash/issues/71
image_directory = 'C:/Users/MRZHE/Desktop/IESEG SESSION_2/the Hackathon project/5.17_BIN/static' 
list_of_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(image_directory))]
static_image_route = '/static/'
df = pd.read_csv('score_card(sum_log).csv')[['ID','prob', 'default', 'level']].astype('str')
import os
import joblib

import sys
from datetime import datetime
from pandas                 import DataFrame
from pandas                 import read_csv
from numpy                  import array
from numpy                  import random
from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.ensemble       import GradientBoostingClassifier
from sklearn.svm            import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.metrics        import accuracy_score
from sklearn.metrics        import auc
from sklearn.metrics        import roc_auc_score
from matplotlib             import pyplot
import datetime
import sys
from scipy.stats          import pearsonr
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

from pandas               import DataFrame
from pandas               import Series
from pandas               import read_csv
from pandas               import get_dummies
from numpy                import array
from numpy                import random
from numpy                import where
from numpy                import nan
from scipy.stats.mstats   import winsorize
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics      import auc
from sklearn.metrics      import roc_auc_score
from matplotlib           import pyplot

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, GridSearchCV
import matplotlib.pyplot
import seaborn as sns


import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats          import pearsonr
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet

# Use SQL in pandas DataFrame
# Ref: https://pypi.python.org/pypi/pandasql
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())  # Allow sqldf to access global environment
logistic = joblib.load("logistic.pkl")
test=read_csv("test.csv")
a=test['default']
test = test.drop('level', axis = 1)
test = test.drop('default', axis = 1)
test = test.drop('prob', axis = 1)
# get the experiment data
t=test


test = test.set_index('ID')
probabilities_logistic = DataFrame(logistic.predict_proba(test))[1]
test['prob'] = probabilities_logistic.tolist()
for v in test['prob'].tolist():
    if v <0.05:
        test.loc[test['prob']==v,'level'] = 'AAA'
    elif (0.05< v and v < 0.1):
        test.loc[test['prob']==v,'level'] = 'AA'
    elif (0.1< v and v < 0.2):
        test.loc[test['prob']==v,'level'] = 'A'
    elif (0.2< v and v < 0.3):
        test.loc[test['prob']==v,'level'] = 'BBB'
    elif (0.3< v and v < 0.4):
        test.loc[test['prob']==v,'level'] = 'BB'
    elif (0.4< v and v < 0.5):
        test.loc[test['prob']==v,'level'] = 'B'
    elif (0.5< v and v < 0.6):
        test.loc[test['prob']==v,'level'] = 'CCC'
    elif (0.6< v and v < 0.7):
        test.loc[test['prob']==v,'level'] = 'CC'
    elif (0.7< v and v < 0.8):
        test.loc[test['prob']==v,'level'] = 'C'
    else:
        test.loc[test['prob']==v,'level'] = 'D'
test['default']=a
test.reset_index()
# get the result
r=test.reset_index()[['ID','prob','level']]
r

###############################################################################
# set the app:
###############################################################################


app = dash.Dash(external_stylesheets=['<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">'])

###############################################################################
# Contructing main app layout:
###############################################################################
app.layout = html.Div([
    
    ######### navigation#################
    html.Div([
        html.Img(id='logo',
        src="https://media-exp1.licdn.com/dms/image/C4D0BAQEs6S_9EaO4Vg/company-logo_200_200/0/1614196553408?e=2159024400&v=beta&t=YBxa7Mia2mkyDwCVuutxgPuhx4WlxcR-ag4i2OQXpS8"
        ),
        html.H2('Choose a dataset',className='start'),   
        dcc.Dropdown('Dropdown_tickers', options=[
            {'label':'Financial Data','value':'fin'},
            {'label':'Financial & Job Data','value':'fin_job'}
        ]),        
#         dcc.Dropdown('Dropdown_tickers', options=[
#             {'label':'Logistic Regression','value':'logistic'},
#             {'label':'Gradient Boosting Classifier','value':'boostedtree'}
#         ]),
        html.H2('Choose a Model',className='start2'),
        dcc.RadioItems(id='radio-items',
                      options=[
                          {'label':'Logistic Regression','value':'logistic'},
                          {'label':'Gradient Boosting Classifier','value':'boostedtree'}
                      ],className='RadioItems'),

        
############ model buttons###################################################################################        
#         html.Div([
#             html.Button("Logistic Regression", className="LR-btn", id="LR",n_clicks=0),
#             html.Button("Gradient Boosting Classifier", className="GB-btn", id="GB",n_clicks=0),
#         ], className="Buttons"),
    ],className='navigation'),
    
    ######### content #################
#     html.H2('---------------------------------------------------------------------------------------',className='start3'),
    html.Div([
        html.Img( id='logo3',
            src="https://media-exp1.licdn.com/dms/image/C4D1BAQFWlXSuowfHAw/company-background_10000/0/1614196612515?e=2159024400&v=beta&t=ZGSgkyD4PJXt8jvL0DBmoSgtEFbsSldWR0EOCWS-2NA"
        ),
#         html.Img(id='logo2',src='/assets/a.png'),
        # all the graphs##############
        # add tab 1
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
            dcc.Tab(label='Models Information', value='tab-1', children=[
                html.H3(id="ticker1"),
                html.Div([
                    html.Img(id="image1"),
                    html.Img(id="image2"),
                    ], className="header1"), 
                    #         # image2
                    #         html.Div([
                    #                 html.P(id="ticker2"),
                    #                 html.Img(id="image2"),
                    #         ], className="header2"),  
                    # image3
                html.Div([
                    html.P(id="ticker3"),
                    html.Img(id="image3"),
                    html.Img(id="image4"),
                    ], className="header3"),            
            ]),
            # add tab 2 
            dcc.Tab(label='Grading Information', value='tab-2', children=[
                html.Img(id="image5"),
                html.Div([
                        Br(),
                        # Add data table
                        html.H2('The level of all company ',className='tablea'),
                        dash_table.DataTable(
                            id='datatable-interactivity',
                            columns=[
                                {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
                            ],
                            data=df.to_dict('records'),
                            editable=False,
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            column_selectable="single",
                            row_selectable="multi",
                            row_deletable=False,
                            selected_columns=[],
                            selected_rows=[],
                            page_action="native",
                            page_current= 0,
                            page_size= 5,
                            style_table={'overflowX': 'auto'},
                            style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                                'whiteSpace': 'normal'
                                },
                            ),
                ], className="header4"),
                Br(),
                html.H2('The experimental data ',className='tableb'),
                dash_table.DataTable(
                    id='table1',
                    columns=[{"name": i, "id": i} for i in t.columns],
                    data=t.to_dict('records'),
                    style_table={'overflowX': 'auto'},
                    style_cell={
                                'height': 'auto',
                                # all three widths are needed
                                'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                                'whiteSpace': 'normal'
                                },
                ),
                Br(),
                html.H2('The prediction of the experimental data ',className='tableb'),
#                 dash_table.DataTable(
#                     id='table2',
#                     columns=[{"name": i, "id": i} for i in r.columns],
#                     data= r.to_dict('records'),
#                     style_table={'overflowX': 'auto'},
#                     style_cell={
#                                 'height': 'auto',
#                                #all three widths are needed
#                                 'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
#                                 'whiteSpace': 'normal'
#                                 },
#                 ), 
#                 Br(),
                html.Div([
                    html.Button('Click to predict', id='btn-nclicks-1', n_clicks=0)
                ], className="Buttons"),
                dash_table.DataTable(
                    id='datatable',
                    columns=[{'name': 'ID', 'id': 'ID'},
                             {'name': 'prob', 'id': 'prob'},
                             {'name': 'level', 'id': 'level'}],
                    
                    page_current=0,
                    page_size=8,
                    page_action='custom')                  
#                 dash_table.DataTable(
#                      columns=[{"name": i, "id": i} for i in value.columns],
#                      row_selectable=False,
#                      filterable=True,
#                      sortable=True,
#                      id='datatable'
#                 ),
            ]),
        ]),
        # image1


#         # image4
#         html.Div([
#                 html.P(id="ticker4"),
#                 html.Img(id="image4"),
#         ], className="header4"), 
        ##################################################
#         html.Div([
#             dash_table.DataTable(
#                 id='datatable-interactivity',
#                 columns=[
#                     {"name": i, "id": i, "deletable": False, "selectable": True} for i in df.columns
#                 ],
#                 data=df.to_dict('records'),
#                 editable=False,
#                 filter_action="native",
#                 sort_action="native",
#                 sort_mode="multi",
#                 column_selectable="single",
#                 row_selectable="multi",
#                 row_deletable=False,
#                 selected_columns=[],
#                 selected_rows=[],
#                 page_action="native",
#                 page_current= 100,
#                 page_size= 10,
#             ),
#             html.Div(id='datatable-interactivity-container')
#         ]),
        html.Div(id="description", className="decription_ticker"),
        html.Div([
            html.Div([], id="graphs-content"),
        ], id="main-content")
    ],className="content"),  
    html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'}),
],className='container')

###############################################################################
# Get all the data
###############################################################################


@app.callback(
    [dash.dependencies.Output('image1', 'src'),
     dash.dependencies.Output("image2", "src"),
     dash.dependencies.Output("image3", "src"),
     dash.dependencies.Output("image4", "src"),
     dash.dependencies.Output("ticker1", "children")],
    [dash.dependencies.Input('Dropdown_tickers', 'value'),
     dash.dependencies.Input('radio-items', 'value')])

def update_image_src(v,M):
    if v== None:
        raise PreventUpdate    
    elif v== 'fin_job':
        if M== None:
            raise PreventUpdate
        elif M == 'logistic':
            ticker='Logistic Regression'
            p1= 'FJ_LR_AUC-VAR.png'
            p2= 'FJ_LR_ROC.png'
            p3= 'FJ_Fisher.png'
            p4= 'var.png'
        else:
            ticker='Gradient Boosting Classifier'
            p1= 'FJ_BT_AUC-VAR.png'
            p2= 'FJ_BT_ROC.png'
            p3= 'FJ_Fisher.png'
            p4= 'var.png'
    else :
        if M== None:
            raise PreventUpdate
        elif M == 'logistic':
            ticker='Logistic Regression'
            p1= 'Fin_LR_AUC-VAR.png'
            p2= 'Fin_LG_test_15var.png'
            p3= 'Fin_LG_LC_15var1.png'
            p4= 'Fin_LR_PLevel.png'
        else:
            ticker='Gradient Boosting Classifier'
            p1= 'Fin_BT_AUC-VAR.png'
            p2= 'Fin_BT_test_15var.png'
            p3= 'Fin_BT_LC_15var1.png'
            p4= 'Fin_BT_PLevel.png'
    return static_image_route + p1,static_image_route + p2,static_image_route + p3,static_image_route + p4,ticker

# from your computer or server
    @app.server.route('{}<image_path>.png'.format(static_image_route))
    def serve_image(image_path1,image_path2,image_path3,image_path4,ticker):
        image_name1 = '{}.png'.format(image_path1)
        image_name2 = '{}.png'.format(image_path2)
        image_name3 = '{}.png'.format(image_path3)
        image_name4 = '{}.png'.format(image_path4)
        ticker=ticker
#         if image_name not in list_of_images:
#             raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
        return flask.send_from_directory(image_directory, image_name1) ,flask.send_from_directory(image_directory, image_name2),flask.send_from_directory(image_directory, image_name3),flask.send_from_directory(image_directory, image_name4),ticker

# get the data table 
@app.callback(
    [dash.dependencies.Output('datatable-interactivity', 'style_data_conditional'),
     dash.dependencies.Output('image5', 'src')],
    [dash.dependencies.Input('datatable-interactivity', 'selected_columns')]
)
def update_styles(selected_columns):
    a=[{
         'if': { 'column_id': i },
        'background_color': '#D2F3FF'
     } for i in selected_columns]    
    p5= 'level1.png'
    return a,static_image_route+p5

# from your computer or server
    @app.server.route('{}<image_path>.png'.format(static_image_route))
    def serve_image1(a,image_path5):
        image_name5 = '{}.png'.format(image_path5)
        a=a
#         if image_name not in list_of_images:
#             raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
        return a, flask.send_from_directory(image_directory, image_name5)
# the data table
@app.callback(
    [dash.dependencies.Output('datatable', 'data'),dash.dependencies.Output('datatable', 'columns')],
    [dash.dependencies.Input('btn-nclicks-1', 'n_clicks')])
def displayClick(btn1):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        return r.to_dict('records'),[{"name": i, "id": i} for i in r.columns]
        
#         return sheet_to_df_map.parse(2).to_dict('records')
# flask.send_from_directory(image_directory, '{}.png'.format('Fin_BT_PLevel.png'))
###################################################################################
#### RUN THE APP################################################################
app.run_server(debug=False)
