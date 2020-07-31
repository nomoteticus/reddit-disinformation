# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly_express as px
from dash.dependencies import Input, Output
import dash_table
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from itertools import chain
import re
import os

def range_end(minval, maxval, stepval):
    i = []
    while minval < maxval:
        i.append(minval)
        minval += stepval
    #if maxval > max(i):
    #    i.append(maxval)
    return(i)

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation'
#rootfold = '/home/ubuntu'

### Read data

df_subr = pd.read_csv(rootfold+"/dashboard/data/app_subr_week.csv")
df_subm = pd.read_csv(rootfold+"/dashboard/data/app_subm_day.csv")
df_flags = pd.read_csv(rootfold+"/dashboard/data/app_flags_day.csv")

df_accuser = pd.read_csv(rootfold+"/dashboard/data/app_authFlaggers_week.csv")
df_suspect = pd.read_csv(rootfold+"/dashboard/data/app_authFlagged_week.csv")

df_domain = pd.read_csv(rootfold+"/dashboard/data/app_domain_week.csv")


df_subr['ISFLAG'] = df_subr['flag']
df_subm['ISFLAG'] = df_subm['flag']
df_subm['cluster'] = '1'
df_subr['week'] = df_subr['week'].astype('int')

df_subr = df_subr[df_subr.week<29]
df_subm = df_subm[df_subm.week<29]

flagtypes = list(df_subr.loc[:,"disinformation":"other"].columns)

min_week = int(df_subr['week'].min())
max_week = int(df_subr['week'].max())
max_week_lst = [max_week]
range_weeks0 = range(min_week, max_week,4)
if max_week not in range_weeks0:
    range_weeks = [*range_weeks0] + [max_week]
week_day_df = df_subm.groupby('week')['day'].agg(['min','max'])
week_day_df['label'] = [r[1]['min'][5:7] + '/' + r[1]['min'][8:10]
                        for r in week_day_df.iterrows()]
range_weeks1 = range_end(min_week, max_week, 4)
range_weeks2 = range(min_week, max_week+1,4)
range_weeks3 = {1:'1',29:'29'}
aaa = int(df_subr['week'].max())
AAA = [aaa]
range_weeks4 = chain(range(1,29,4), AAA)

top20subreddits = list(df_subr.subreddit.value_counts().index[:20])


### Init APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
server = app.server



### Sliders and Buttons
slider_week =dcc.RangeSlider(
                id = 'sl_week',
                min = min_week, max = max_week,
                step=1,
                marks = {i:str(i) for i in range_weeks},#{ind:week_day_df.loc[ind].loc['label'] for ind in range_weeks2 },
                value = [max_week-4,max_week-1] )

checklist_flagtypes = dcc.Checklist(
                id = 'cl_flagtypes', 
                value=['disinformation', 'fakenews', 'unreliable', 'propaganda'],
                options= [{'label':t,'value':t} for t in flagtypes] )

radio_coronatopic = dcc.RadioItems(
                id = 'rd_coronatopic', 
                value="yes",
                options= [{'label': 'yes', 'value':"yes"},
                          {'label': 'no','value':"no"}] )
@app.callback(
    Output(component_id='show_dates', component_property='children'),
    [Input(component_id='sl_week', component_property='value')]
)
def update_output_div1(input_value):
    return('%s' % range_weeks)


### LAYOUT


style_layout = {'width': '25%', 'display': 'inline-block',
                'position': 'fixed', 'padding-top': '50px',
                'background-color':'lightyellow',
                'height':'1200px'}

app.layout = \
    html.Div(
        children=[
            ### hidden
            html.Div(id='filtered_json_week', style={'display': 'none'}),
            
            html.P(id='show_dates'),
            ### LEFT PANE (filters)
            html.Div([html.H4('Choose period: ', style = {'padding-top': '50px'}),
                      slider_week,
                      #html.Div(id='show_dates', style={'text-align':'center', 'font-size':'150%'}),
                      html.H4('Choose type of flag', style = {'padding-top': '50px'}),
                      checklist_flagtypes,
                      html.H4('Coronavirus only?', style = {'padding-top': '20px'}),
                      radio_coronatopic])])


radio_ntopflags = \
    dcc.RadioItems(
        id = 'rd_ntopflags',
        options=[{'label': x, 'value': x} for x in [10,20,50,100]],
        value=10,
        labelStyle={'display': 'inline-block'})  

drop_subr20 = \
    dcc.Dropdown(
        id='dr_subr20', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')

### Run app
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)
