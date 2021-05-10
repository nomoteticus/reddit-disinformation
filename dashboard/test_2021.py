#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 15:34:12 2021

@author: j0hndoe
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly_express as px
from dash.dependencies import Input, Output, State
import dash_table
from sklearn.cluster import MiniBatchKMeans
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from datetime import datetime
import re
import sys
import time
#import os
import pathlib

#https://dash-bootstrap-components.opensource.faculty.ai/docs/components/input/

current_version =  '0.1.0'

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/dashboard'

#rootfold = pathlib.Path(__file__).parent
#DATA_PATH = rootfold.joinpath("dashboard/onserver").resolve()

sys.path.append(rootfold)
import func_dashboard as fd



#####
##### INITIALIZE APP
#####

#external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

app = dash.Dash(__name__, 
                #external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
server = app.server



#####
##### READ & PREPARE DATA
#####

df_subr = pd.read_csv(rootfold+"/data/app_2021_subr_week.csv")
df_subm = pd.read_csv(rootfold+"/data/app_2021_subm_day.csv")
df_flags = pd.read_csv(rootfold+"/data/app_2021_flags_day.csv")

df_accuser = pd.read_csv(rootfold+"/data/app_2021_authFlaggers_week.csv")
df_suspect = pd.read_csv(rootfold+"/data/app_2021_authFlagged_week.csv")

df_domain = pd.read_csv(rootfold+"/data/app_2021_domain_week.csv")


df_subr['ISFLAG'] = df_subr['flag']
df_subm['ISFLAG'] = df_subm['flag']
df_subm['cluster'] = '1'
df_subr['week'] = df_subr['week'].astype('int')

df_all_subm = df_subr.groupby(['subreddit','week'])[['n_subm_all', 'n_subm_kept']].agg('sum').reset_index()

top20subreddits = df_subr.groupby(['subreddit'])['ISFLAG'].agg('sum').\
    sort_values(ascending=False).head(20).index


#####
##### CALLBACKS 
#####


### FILTERING dataset


@app.callback(
    [Output('flag_filt_json', 'children'),
     Output('subm_filt_json', 'children')],
    [Input('sl_week', 'value'),
      Input('cl_flagtypes', 'value'),
      Input('drop_min_nflags', 'value'),
      Input('dr_subr20_subm', 'value')])
def filter_flags_subm(sl_week, cl_flagtypes, drop_min_nflags, dr_subr20_subm):
    flag_filt = df_flags.copy()
    ### filter by week
    flag_filt = flag_filt[(flag_filt['week'].between(*sl_week))]
    ### filter by flag types
    flag_filt['ISFLAG'] = np.sign(flag_filt[cl_flagtypes].copy().sum(axis=1))
    ### filter by subreddit
    if dr_subr20_subm is not None:
        flag_filt = flag_filt[flag_filt.subreddit.str.contains('|'.join(dr_subr20_subm))]
    ### aggregate at submission level
    agg_flg = flag_filt.groupby('subm_id').agg(nflags = ('ISFLAG', np.sum))

    ### filter by nflags
    agg_flg = agg_flg[agg_flg.nflags >= drop_min_nflags]
    ###
    subm_filt = df_subm.copy()
    subm_filt = subm_filt.set_index('subm_id').join(agg_flg, how = 'inner').reset_index()
    ###
    flag_filt = flag_filt[flag_filt.subm_id.isin(subm_filt.subm_id)]
    ###
    return (flag_filt.to_json(orient='split'),
            subm_filt.to_json(orient='split'))




# @app.callback(
#     [Output('highl_n_subm', 'children'),
#      Output('highl_n_subr', 'children'),
#      #Output('subm_evo_week_json', 'children'),
#      Output('subm_formatted', 'children')],
#      #Output('top_keyword', 'children')],
#     [Input('flag_filt_json', 'children'),
#      Input('subm_filt_json', 'children')]
# )
# def extract_flags_subm(flag_filt_json, subm_filt_json):
#     time.sleep(0.4)
#     flag_filt = pd.read_json(flag_filt_json, orient='split')
#     subm_filt = pd.read_json(subm_filt_json, orient='split')
#     ###
#     #subm_formatted = fd.generate_submissions(flag_filt, 1)
#     ###
#     # subm_evo_week = fd.get_flag_perc(df_all_subm, 
#     #                                  subm_filt, 
#     #                                  groups=['week'])
#     #subm_evo_week = pd.DataFrame({'week':[1,2], 'n_flags':[3,5]})
#     ###
#     #top_keyword = fd.cluster_commonwords(subm_filt.subm_title,1)
#     ###
#     return ( ### highlights / n
#              str(subm_filt.shape[0]),
#              str(len(subm_filt.subreddit.unique())),
#              ### highlights / tendplot
#              #subm_evo_week.to_json(orient='split'),
#              ### highlights / lastsub
#              subm_formatted,
#              ### keywords
#              top_keyword)


#             str(filt_df_domain[filt_df_domain['ISFLAG']>0].shape[0]),
#             str(filt_df_user[filt_df_user['ISFLAG']>0].shape[0]))
# #    return (flag_filt.to_json(orient='split'),
            #subm_filt.to_json(orient='split'))

# @app.callback(
#     Output('cacat','children'),
#     [Input('subm_evo_week_json','children')])
# def muie(qq):
#     filtered_df_week      = pd.read_json(qq, orient='split')
#     return str(filtered_df_week.columns)
 



##### //
##### ELEMENTS
#####

#### //
#### // TOP right

### Slider - week

@app.callback(
    Output(component_id='show_dates',   component_property='children'),
    [Input(component_id='sl_week', component_property='value')]
)
def show_dates(sl_week):
    beg_end =(fd.revdate(week_day_df.loc[sl_week[0]].loc['min'], year = False),
              fd.revdate(week_day_df.loc[sl_week[1]].loc['max'], year = True))
    return '⬐ from %s to %s ⬎' % beg_end

#### // 
#### // BOTTOM left


### Slider - time
week_day_df = fd.generate_slider_df(df_subr, df_subm)
slider_week = fd.dcc_day_slider(df_subr, df_subm, 
                                weeks_shown = 4, step = 1)

### Checklist - flagtypes

flagtypes  = ["disinformation","fakenews","misleading", "unreliable", "propaganda", "bs", "other"]
flaglabels = ["Dis/Misinformation","Fake/False news","Misleading/Clickbait", "Unreliable", "Propaganda","Bullshit", "Other"]
checklist_flagtypes = \
    dcc.Checklist(
        id = 'cl_flagtypes', 
        value=['disinformation', 'fakenews', 'misleading', 'unreliable', 'propaganda'],
        options= [{'label':l,'value':v} for v,l in zip(flagtypes, flaglabels)],
        className = 'checklist_labels'
        )

### Dropdown - nflags
dropdown_min_nflags = \
    dcc.Dropdown(
        id='drop_min_nflags', 
        value = 1,
        options=[{'label': i, 'value': i} for i in [1,2,3,5,10] ], 
        multi=False)

### Dropdown - topic (not yet)

### Dropdown - subreddits
drop_subr20_subm = \
    dcc.Dropdown(
        id='dr_subr20_subm', 
        options=[{'label': sr.capitalize(), 'value': sr} for sr in sorted(top20subreddits)], 
        multi=True, placeholder='Filter by subreddit...')
    
    


##### // 
##### // BOTTOM right


### /// TAB 1 - Highlights

### Info boxes

@app.callback(
    [Output('highl_n_subm', 'children'),
     Output('highl_n_subr', 'children'),
     Output('highl_top_keyw', 'children')],
    [Input('flag_filt_json', 'children'),
     Input('subm_filt_json', 'children')]
)
def highl_info_boxes(flag_filt_json, subm_filt_json):
    flag_filt = pd.read_json(flag_filt_json, orient='split')
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    ###
    ###
    #subm_formatted = fd.generate_submissions(flag_filt, 1)
    ###
    top_keyword = fd.cluster_commonwords(subm_filt.subm_title,1)
    ###
    return (str(subm_filt.shape[0]),
            str(len(subm_filt.subreddit.unique())),
            top_keyword)


 ### highlights / tendplot
 #subm_evo_week.to_json(orient='split'),
 ### highlights / lastsub
 #subm_formatted,
 ### keywords
 #top_keyword
#             )


### PLOT - > line evo
### Radio: n or pct
radio_subr_cat_stat = \
    dcc.RadioItems(
        id = 'rd_evo_week',
        options=[{'label':'Daily (N/day)', 'value': 'n_day'},
                 {'label':'Total (N)', 'value': 'n_flags'},
                 {'label':'Percent (%)', 'value': 'perc_FLAG_pr'}],
        value='n_day', labelStyle={'display': 'inline-block'} ) 
@app.callback(
    Output('plot_evol_line_n','figure'),
    [Input('subm_filt_json','children'),
     Input('rd_evo_week','value')])
def plot_highl_evol_line(subm_filt_json, nperc):
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    # group data by week
    filtered_df_week = fd.get_flag_perc(df_all_subm, 
                                        subm_filt, 
                                        week_day_df,
                                        groups=['week'])    
    # draw plot
    evol_line = fd.plot_highl_evo(filtered_df_week, week_day_df, nperc)
    return evol_line
    
@app.callback(
    Output('highl_subm_detail', 'children'),
    [Input('flag_filt_json', 'children')]
)
def highl_submissions(flag_filt_json):
    flag_filt = pd.read_json(flag_filt_json, orient='split')
    ###
    return fd.generate_submissions(flag_filt, 5, 1, 
                                   class_name = 'shadow_container container_subm')

### Show subm - collapse
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
    )
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


### PLOT - > word frequency
@app.callback(
    Output('plot_highl_word_count','figure'),
    [Input('subm_filt_json','children')])
def plot_highl_word_count(subm_filt_json):
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    # extract common words
    df_subm_words = fd.df_commonwords(subm_filt.subm_title, nwords = 10).sort_values('freq')
    # draw plot
    pl_word_count = fd.plot_highl_word(df_subm_words)
    return pl_word_count

@app.callback(
    Output('plot_two_keywords','figure'),
    [Input('subm_filt_json','children')])
def plot_two_keywords(subm_filt_json):
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    word1 = 'trump'
    word2 = 'biden'
    # group data by week
    df_subm_words_week = subm_filt.groupby('week')['subm_title'].apply(fd.df_commonwords).reset_index()
    df_subm_words_week = df_subm_words_week[df_subm_words_week['word'].str.lower().isin([word1,word2])]
    #fd.df_commonwords(subm_filt.subm_title, nwords = 10).sort_values('freq')
    # draw plot
    pl_two_keywords = fd.plot_two_keywords(df_subm_words_week, word1, word2, week_day_df)
    return pl_two_keywords


##### //
##### LAYOUT 
##### //

app.layout = \
    html.Div(
        className = 'page',
        children = [
        ### hidden
        #html.Div(id='subm_evo_week_json', style={'display': 'none'}),   
        html.Div(id='flag_filt_json', style={'display': 'none'}),   
        html.Div(id='subm_filt_json', style={'display': 'none'}),   
        
        ### TOP
        html.Div(
            className = 'header',
            children=[
            ### TOP LEFT
            html.Div(className = 'item north west',
                     children=[
                      html.Img(src=app.get_asset_url('stroo_v1.png'),
                               className = 'nw_stroo'),
                      html.Span(
                              html.H5(html.B('Is it true?')),
                              className = 'nw_stroo_txt'
                              ),
                      ]),
            ### TOP RIGHT
            html.Div(className = 'item north east',
                     children = [
                         html.Div([
                            html.Span("Monitoring perceived disinformation on Reddit ",
                                      className = 'dates_banner'),
                            html.Span(id='show_dates',
                                      className = 'dates_banner_bold')],
                            className = 'title_ne'),
                      html.Div(
                        slider_week,
                        className = 'title_ne')
                      ])
            ]),
        ### BOTTOM
        html.Div(
            className = 'layout',
            children=[
                ### BOTTOM LEFT
                html.Div(className = 'item south west',
                        children = [html.H5('⬐ Filter data', 
                                    className = 'filter_data'),
                         html.H5('Type of flag'),
                         checklist_flagtypes,
                         html.Span([
                             html.Div(html.H5('Flag count'),
                                     className = 'flag_count'),
                             html.Div(dropdown_min_nflags,
                                      className = 'flag_count')
                             ], className = 'cont_generic',
                             ),
                         html.H5('Topic of posts'),
                         html.H5('Subreddits'),
                         drop_subr20_subm,
                         html.P(html.I('version ' + current_version), 
                                style = {'padding-top': '250px'})
                    ]),
                ### BOTTOM RIGHT
                ### TABS
                html.Div(className = 'item south east content',
                         children = [
                             html.Div([
                                 dcc.Tabs(id='tabs-example', 
                                          value='tab-1', 
                                          parent_className='custom-tabs',
                                          className='custom-tabs-container',
                                          children=[
                                            dcc.Tab(label='Highlights', value='tab-1', className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Submissions',value='tab-2', className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Subreddits', value='tab-3', className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Clustering', value='tab-4', className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Domains',    value='tab-5', className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Method',     value='tab-6', className='custom-tab', selected_className='custom-tab--selected')
                                            ]),
                                 html.Div(id='tabs-example-content')
                                 ])
                        ])
                ])
        ])
    



##### //
##### TABS
##### //


@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value'),
               Input('flag_filt_json', 'children'),
               Input('subm_filt_json', 'children')])
def render_content(tab, flg, subm):
    ### TAB1 - Flags
    if tab == 'tab-1':
        return html.Div([ 
                  html.P(html.Br()),
                  html.P(fd.texts['highlights']['description1']),
                  html.P(fd.texts['highlights']['description2']),
                  html.H5('Posts flagged by Reddit users as false information'),
                  ###
                  ### Posts flagged by Reddit users
                  fd.Row([
                      ### info boxes
                      fd.Row([
                           fd.info_box("# Flagged posts:",   content_id='highl_n_subm'),
                           fd.info_box("# Subreddits:",      content_id='highl_n_subr'),
                           #html.Div("# placeholder:", className = 'shadow_container')
                           fd.info_box("#1 Keyword:", content_id='highl_top_keyw')
                           #fd.info_box("# Flagged domains:", content_id='highl_n_subm')
                           # fd.info_box("# Flagged users:",   content_id='highl_n_user')
                           ],
                          stretch=False),
                      ### evo plot
                      fd.Col([
                          fd.Item(
                              [
                              fd.Row(dcc.Graph(id='plot_evol_line_n')),
                              fd.Row(radio_subr_cat_stat)
                              ],
                              shadow = True)
                      ])
                  ]),
                  ### submission details
                  html.H5('Most recent post flagged by at least 2 users'),
                  fd.Row(
                      [html.Div(id = 'highl_subm_detail'),
                       dbc.Button("Show / Hide more submissions",
                             id="collapse-button", className="mb-3", 
                             color="secondary")
                       ]),
                  ###
                  html.H5('Common keywords in flagged posts'),
                  fd.Row([
                          fd.Item(dcc.Graph(id = 'plot_highl_word_count'), shadow=True),
                          fd.Item(dcc.Graph(id = 'plot_two_keywords'), shadow=True)
                          ]),
                  html.P('Entities!'),
                  ###
                  html.H5('Common domains in flagged posts'),
                  html.P(fd.texts['highlights']['top']),
                  ###
                  html.H5('Common phrases used for flagging'),
                  html.P(fd.texts['highlights']['top'])                                  
                ])#,className = 'tab_div')
    elif tab == 'tab-2':
        return html.Div([

            
               ])




#####
##### RUN APP
#####

if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)



#####
##### Version improvements: 0.1.0
#####
# - added single displays for: n_post, n_subr, #1_keyword,..
# - added left tab (sw) for #flags
# - 
# -
# - switched layout to flex
# - orange theme

#####
##### Known bugs: 0.1.0
#####
# SOLVED - callback error for elements in tab 1
# compression of tabs











