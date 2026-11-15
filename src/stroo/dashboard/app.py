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
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS
from datetime import datetime
import re
import sys
import time
#import os
import pathlib

#https://dash-bootstrap-components.opensource.faculty.ai/docs/components/input/

current_version =  '0.1.2'

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/dashboard'

#rootfold = pathlib.Path(__file__).parent
#DATA_PATH = rootfold.joinpath("dashboard/onserver").resolve()

sys.path.append(rootfold)
import func_dashboard as fd

def perc(x):
    return 100*x/x.sum()

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

df_sources = pd.read_csv(rootfold+"/data/UNITED_SOURCES.csv")

df_subr['ISFLAG'] = df_subr['flag']
df_subm['ISFLAG'] = df_subm['flag']
df_subm['cluster'] = '1'
df_subr['week'] = df_subr['week'].astype('int')

df_all_subm = df_subr.groupby(['subreddit','week'])[['n_subm_all', 'n_subm_kept']].agg('sum').reset_index()

df_flags['domain'] = [fd.simple_url(url, drop_http=True) for url in df_flags['link_url']]
df_flags['domain'] = [fd.remove_http_www(d) for d in df_flags['domain']]
df_subm_domains = df_flags.groupby('subm_id')['domain'].agg('first')
df_subm = df_subm.copy().set_index('subm_id').join(df_subm_domains).reset_index()

top20subreddits = df_subr.groupby(['subreddit'])['ISFLAG'].agg('sum').\
    sort_values(ascending=False).head(20).index

euclid = pd.DataFrame(euclidean_distances(df_subm[['PC01', 'PC02', 'PC03', 'PC04', 'PC05']]))
euclid = pd.concat([df_subm[['subm_id']], euclid], axis=1).set_index('subm_id')



#####
##### CALLBACKS 
#####



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
flaglabels = ["Dis/Misinfo","Fake news","Clickbait", "Unreliable", "Propaganda","Bullshit", "Other"]
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
        value = 2,
        options=[{'label': 'minimum ' + str(i), 'value': i} for i in [1,2,3,5,10] ], 
        multi=False, clearable=False, searchable=False,
        className='left_dropdowns')
        # style={'height': '25px',
        #        'display': 'inline-block',
        #        }
        #)

### Dropdown - topic (not yet)

### Dropdown - subreddits
drop_subr20_subm = \
    dcc.Dropdown(
        id='dr_subr20_subm', 
        options=[{'label': sr.capitalize(), 'value': sr} for sr in sorted(top20subreddits)], 
        multi=True, clearable=False,
        placeholder='Filter by subreddit...',
        className ='left_dropdowns')
    


### Checklist - sources

checklist_sources = \
    dcc.Checklist(
        id = 'cl_sources', 
        value=[1,2],
        options= [{'value':1,'label':'Unreliable'},
                  {'value':2,'label':'Unknown'},
                  {'value':3,'label':'Reliable'}],
        className = 'checklist_labels'
        )



### // FILTERING dataset

@app.callback(
    [Output('flag_filt_json', 'children'),
     Output('subm_filt_json', 'children')],
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('drop_min_nflags', 'value'),
     Input('dr_subr20_subm', 'value'),
     Input('cl_sources', 'value')])
def filter_flags_subm(sl_week, cl_flagtypes, drop_min_nflags, dr_subr20_subm, cl_sources):
    flag_filt = df_flags.copy()
    ### filter by week
    flag_filt = flag_filt[(flag_filt['week'].between(*sl_week))]
    ### filter by flag types
    flag_filt['ISFLAG'] = np.sign(flag_filt[cl_flagtypes].copy().sum(axis=1))
    flag_filt = flag_filt[flag_filt['ISFLAG']==1]
    ### filter by subreddit
    if dr_subr20_subm is not None:
        flag_filt = flag_filt[flag_filt.subreddit.str.contains('|'.join(dr_subr20_subm))]
    ### filter by sources
    domains_rel = df_sources.domain[df_sources.src_reliability.isin(cl_sources)]
    flag_filt = flag_filt[flag_filt.domain.isin(domains_rel) | ~flag_filt.domain.isin(df_sources.domain)]
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




##### // 
##### // BOTTOM right


### T1 - Highlights

# Info boxes

@app.callback(
    [Output('highl_n_subm', 'children'),
     Output('highl_n_subr', 'children'),
     Output('highl_n_wdom', 'children'),
     Output('highl_top_keyw', 'children'),
     Output('highl_top_flag', 'children')],
    [Input('flag_filt_json', 'children'),
     Input('subm_filt_json', 'children')]
)
def highl_info_boxes(flag_filt_json, subm_filt_json):
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    flag_filt = pd.read_json(flag_filt_json, orient='split')
    ###
    #
    top_keyword = fd.df_commonwords(subm_filt.subm_title).word[0].capitalize()
    #
    top_flags   = ' | '.join(flag_filt.sent.str.replace("[.,;]$", "").value_counts().head(2).index)
    ###
    return (str(subm_filt.shape[0]),
            str(len(subm_filt.subreddit.unique())),
            str(len(subm_filt.domain.unique())),
            top_keyword,
            top_flags)




# PLOT - > line evo
# Radio: n or pct
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

# Show subm - collapse
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
    )
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


# PLOT - > word frequency
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



### T2: SUBMISSIONS


@app.callback(
    Output('subm_subm_detail', 'children'),
    [Input('flag_filt_json', 'children')]
)
def subm_submissions(flag_filt_json):
    flag_filt = pd.read_json(flag_filt_json, orient='split')
    ###
    return fd.generate_submissions_subm(flag_filt, df_subm, euclid, 
                                        10, 1, 
                                        class_name = 'shadow_container container_subm')

# Show subm - collapse
@app.callback(Output("collapse_flags0", "is_open"), [Input("collapse-button-flags0", "n_clicks")],[State("collapse_flags0", "is_open")])
def toggle_collapse_flags0(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags1", "is_open"), [Input("collapse-button-flags1", "n_clicks")],[State("collapse_flags1", "is_open")])
def toggle_collapse_flags1(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags2", "is_open"), [Input("collapse-button-flags2", "n_clicks")],[State("collapse_flags2", "is_open")])
def toggle_collapse_flags2(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags3", "is_open"), [Input("collapse-button-flags3", "n_clicks")],[State("collapse_flags3", "is_open")])
def toggle_collapse_flags3(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags4", "is_open"), [Input("collapse-button-flags4", "n_clicks")],[State("collapse_flags4", "is_open")])
def toggle_collapse_flags4(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags5", "is_open"), [Input("collapse-button-flags5", "n_clicks")],[State("collapse_flags5", "is_open")])
def toggle_collapse_flags5(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags6", "is_open"), [Input("collapse-button-flags6", "n_clicks")],[State("collapse_flags6", "is_open")])
def toggle_collapse_flags6(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags7", "is_open"), [Input("collapse-button-flags7", "n_clicks")],[State("collapse_flags7", "is_open")])
def toggle_collapse_flags7(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags8", "is_open"), [Input("collapse-button-flags8", "n_clicks")],[State("collapse_flags8", "is_open")])
def toggle_collapse_flags8(n, is_open):
    return (not is_open) if n else is_open
@app.callback(Output("collapse_flags9", "is_open"), [Input("collapse-button-flags9", "n_clicks")],[State("collapse_flags9", "is_open")])
def toggle_collapse_flags9(n, is_open):
    return (not is_open) if n else is_open





### T3: CLUSTERING

slider_nclust = \
    dcc.Slider(id = 'sl_nclust',
    min=1,  max=10,
    step=1,
    marks={i:str(i) for i in range(1,11)},
    value=5)  

radio_clust_which = \
    dcc.RadioItems(
        id = 'rd_clust_which',
        options=[{'label':'evolution', 'value': 'evolution'},
                 {'label':'differentiation', 'value': 'differentiation'}],
        value='evolution', labelStyle={'display': 'inline-block'} ) 

drop_clustersubm = \
    dcc.Dropdown(
        id='dr_clustersubm', 
        #options=[{'label': i, 'value': i} for i in top20subreddits], 
        value = 1,
        multi=False, placeholder='Choose cluster')


@app.callback(
    Output('dr_clustersubm', 'options'),
    [Input('sl_nclust', 'value')])
def set_nclust(nclust):
    return [{'label': 'Cluster #'+str(i), 'value': i} for i in range(1,nclust+1)]


    
@app.callback(
    [Output('plot_kmeans','figure'),
     Output('badges_topwords','children'),
     #Output('lst_topwords','value'),
     Output('tbl_clustersubm','children')],
    [Input('subm_filt_json','children'),
     Input('sl_nclust','value'),
     Input('dr_clustersubm','value'),
     Input('rd_clust_which','value')])
def plot_kmeans(subm_filt_json, nclust, clusterselected, which_fig):
    subm_filt = pd.read_json(subm_filt_json, orient='split')
    ### Run cluster analysis
    if nclust>1:
        X = subm_filt[['PC01', 'PC02', 'PC03', 'PC04', 'PC05']]
        cluster = MiniBatchKMeans(n_clusters=nclust, batch_size=1000, random_state=777)
        cluster.fit(X)
        subm_filt['cluster'] = (cluster.predict(X)+1).astype(str)
        subm_filt = subm_filt.reset_index()
        ### Sort by cluster to keep graph tidy
        subm_filt=subm_filt.sort_values('cluster')
    ### Build plot
    fig_clust = fd.plot_clusters(subm_filt)
    #
    subm_filt['week'] = subm_filt['week'].astype('int')
    subm_cl_week = subm_filt.groupby(['week','cluster'])['flag'].\
                             agg('count').groupby(level=0).apply(perc).reset_index().\
                             sort_values(['cluster','week'])
    fig_evo_clust = fd.plot_evo_clusters(subm_cl_week, week_day_df) 
    ### Words
    cw=[fd.print_cluster_commonwords(subm_filt, cl, n = 5) \
        for cl in range(1,nclust+1)]
    ### Generate most relevant submissions for each cluster
    #sel_submissions_index = np.argsort(np.inner(X,cluster.cluster_centers_[clusterselected-1,:]))[-100:]
    #sel_submissions_df = subm_filt[['day','subm_title','subreddit','domain']].loc[sel_submissions_index]
    sel_submissions_df = subm_filt[['day','subm_title','subreddit','domain']]\
        [subm_filt['cluster']==str(clusterselected)].sort_values('day',ascending=False).head(100)
    sel_submissions_df['subm_title'] = fd.shorten_titles(sel_submissions_df['subm_title'],75)
    sel_submissions_df['day'] = [day[5:] for day in sel_submissions_df['day']]
    ###
    fig_to_return = fig_evo_clust if which_fig=='evolution' else fig_clust
    return fig_to_return, \
           cw, \
           fd.generate_table(sel_submissions_df, className ='small', max_rows = 30)

##### //
##### LAYOUT 

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
                html.Div(className = 'item south_gradient west',
                         children = [
                         html.Div(html.I('version ' + current_version)),
                         html.H5('⬐ Filter data', className = 'filter_data'),
                         # Settings
                         fd.Row(
                             fd.Col(
                                 [
                                 html.H5('Flag count'),
                                 html.Div(dropdown_min_nflags),
                                 html.H5('Type of flag'),
                                 checklist_flagtypes,
                                 html.H5('Subreddits'),
                                 drop_subr20_subm,
                                 html.H5('Websites linked'),
                                 checklist_sources,
                                 #html.H5('Prestige of flagger'),
                                 fd.ToDo('karma++/tenure')
                                 ]
                            )
                         ),                         
                    ]),
                ### BOTTOM RIGHT
                ### TABS
                html.Div(className = 'item south east content',
                         children = [
                             html.Div([
                                 dcc.Tabs(id='tabs-example', 
                                          value='tab-highl', 
                                          parent_className='custom-tabs',
                                          className='custom-tabs-container',
                                          children=[
                                            dcc.Tab(label='Highlights', value='tab-highl',      className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Submissions',value='tab-subms',      className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Clustering', value='tab-clust',      className='custom-tab', selected_className='custom-tab--selected'),
                                            dcc.Tab(label='Method',     value='tab-methd',      className='custom-tab', selected_className='custom-tab--selected')
                                            ]),
                                 html.Div(id='tabs-example-content')
                                 ])
                        ])
                ])
        ])
    



##### //
##### TABS
##### //


search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", 
                          placeholder="Search by keyword (not working yet)"),
                width = 5),
        dbc.Col(
            dbc.Button(html.Img(src=app.get_asset_url('search_symb.png'),
                                className = 'search_symb'), 
                       color="warning", className="ml-5"),
            width = 1,
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value'),
               Input('flag_filt_json', 'children'),
               Input('subm_filt_json', 'children')])
def render_content(tab, flg, subm):
    ### TAB1 - Highlights
    if tab == 'tab-highl':
        return html.Div([ 
                  html.P(html.Br()),
                  html.P(fd.texts['highl']['description1']),
                  html.P(fd.texts['highl']['description2']),
                  html.H5('Posts flagged by Reddit users as false information'),
                  #
                  fd.Row([
                  ### a. info boxes and evo plot
                      # info boxes
                      fd.Row([
                           fd.info_box("# Flagged posts:",   content_id='highl_n_subm'),
                           fd.info_box("# Subreddits:",      content_id='highl_n_subr'),
                           #html.Div("# placeholder:", className = 'shadow_container')
                           fd.info_box("# Flagged websites:", content_id='highl_n_wdom'),
                           fd.info_box("#1 Keyword:",        content_id='highl_top_keyw'),                           
                           fd.info_box("Top informal flags (comments)", content_id='highl_top_flag', 
                                       big_number = "big_number"),
                           ],
                          stretch=True,
                          size = 7),
                      # evo plot
                      fd.Row(
                          fd.Col(
                              fd.Item(
                                  [
                                  fd.Row(dcc.Graph(id='plot_evol_line_n')),
                                  fd.Row(radio_subr_cat_stat)
                                  ],
                                  shadow = True)
                              ),
                              size = 5)                      
                  ]),
                  ### b. submission details
                  html.H5('Most recent post flagged by at least 2 users'),
                  fd.Row(
                      [html.Div(id = 'highl_subm_detail'),
                       fd.Row(html.Div(
                               dbc.Button("Show / Hide more submissions",
                                           id="collapse-button", 
                                           className="mb-3 panel", 
                                           color="light")),
                              classAdd = 'more_subm_button')
                       ]),
                  #
                  ### c. common keywords
                  html.H5('Common keywords in titles of flagged posts'),
                  fd.Row([
                          fd.Item(dcc.Graph(id = 'plot_highl_word_count'), shadow=True),
                          fd.Item(dcc.Graph(id = 'plot_two_keywords'), shadow=True)
                          ]),
                  fd.ToDo('Choose word1 and word2'),
                  fd.ToDo('Named Entity Recognition!'),
                  ###
                  html.H5('Common websites linked in flagged posts'),
                  #html.P(fd.texts['highl']['top']),
                  ###
                  html.H5('Common phrases used for flagging'),
                  #html.P(fd.texts['highl']['top'])                                  
                ])#,className = 'tab_div')
    
    
    ### TAB2 - submissions
    elif tab == 'tab-subms':
        return html.Div([
                        html.Br(),
                        html.P(fd.texts['subm']['desc1']),
                        html.P(fd.texts['subm']['desc2a']),
                        html.Br(),
                        #
                        #search_bar,
                        html.Div(id='subm_subm_detail'),
                        #
                        html.Br(),
                        fd.ToDo('- filter by keyword'),
                        fd.ToDo('- filter by domain'),
                        fd.ToDo('- list or table toggle'),
                        fd.ToDo('- save table'),
                        fd.ToDo('- I think this is / is not disinformation button'),
                        fd.Row(
                            [
                            # Col1: slider & keywords
                            fd.Item('.'),
                            fd.Item('.')
                            ]
                        )     
            ])
    
    
    ### TAB3 - clustering
    elif tab == 'tab-clust':
        return html.Div([
                    html.Div([html.Br(),
                              html.P('Topics are computed based on the submission titles (Universal Sentence Embeddings + PCA + K-means)')]),
                    fd.Row(
                        [
                        fd.Item(
                            # Col1: slider & keywords
                            fd.Col(
                            [
                                fd.Item(html.H5('1. Choose number of clusters')),
                                fd.Item(html.Div(slider_nclust)),
                                fd.Item(html.H6('Top keywords / cluster')),
                                fd.Item(html.Div(id='badges_topwords')),
                            ]),
                            size = 5
                            ),
                            # Col2: PCA MDS plot
                            fd.Col([
                                fd.Item(dcc.Graph(id='plot_kmeans'),
                                        shadow=True),
                                fd.Row([
                                    fd.Item('Choose type of plot:', size =2),
                                    fd.Item(radio_clust_which, size =3)
                                    ])
                                   ],
                                   size = 6)
                            ]
                        ),
                    #
                    html.Br(),
                    fd.Row([fd.Item(html.H5('2. Choose cluster for inspection:'), size = 5),
                            fd.Item(drop_clustersubm                            , size = 2),
                            fd.Item(''                                          , size = 4)]),
                    #
                    html.H6('The most relevant titles for each cluster:'),
                    fd.ToDo('choose: table or list + save table'),
                    html.Div(id='tbl_clustersubm'),

                ])
    
    ### TAB4 - method
    elif tab == 'tab-methd':
        return html.Div([
                    
                    html.Br(),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['strt']]),
                    
                    html.Br(),
                    html.H5('1. Data collection'),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['col'][:3]]),
                    html.Ul([html.Li(children=t) for t in fd.texts['methd']['subr_criteria']]),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['col'][3:]]),

                    html.Br(),
                    html.H5('2. Detecting informal flags'),
                    html.Div([html.P(t) for t in fd.texts['methd']['flg'][:1]]),
                    html.Div(
                        [html.Span([dbc.Badge(x, className="ml-1", color='warning'), arrow ]) for x, arrow in \
                         zip(['keyword filtering','sentence tokenization','pattern matching','aggregating to submissions', 'deciding if flag'],
                             ['➝','➝','➝','➝',''] )]
                        ),
                    html.Br(),
                    
                    dbc.Row([
                        dbc.Col([
                            html.Br(),
                            html.H6('2a. Keyword filtering'),
                            html.Div([html.P(children=t) for t in fd.texts['methd']['keyw'][:1]]),
                            html.Ul([html.Li(children=t) for t in fd.texts['methd']['keyw'][1:]]),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                            html.H6('2b. Pattern matching')    
                        ], 
                        width = 5),
                        
                        dbc.Col(html.Img(src=app.get_asset_url('method_pipeline.png'),
                                className = 'method_pipeline'),
                        width = 7),
                            
                    ]),
                    
                    html.Div([html.P(children=t) for t in fd.texts['methd']['patt'][:3]]),
                    html.Div(html.Img(src=app.get_asset_url('method_patterns.png'),
                                      className = 'method_patterns')),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['patt'][3:]]),

                    html.H5('3. Clustering submissions'),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['cls']]),
                    
                    html.H5('3. Source credibility (of linked websites)'),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['src'][:2]]),
                    html.Ul([html.Li(children=t) for t in fd.texts['methd']['src'][2:]]),
                    
                    html.Br(),
                    html.Br(),
                    html.Div([html.P(children=t) for t in fd.texts['methd']['end']]),
                    
                    
                    fd.ToDo('Add links')
            
            
            ])

                    


#####
##### RUN APP
#####

if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)


"""
Version improvements: 0.1.2
 - added source credibility
 - Analyze button for individual news
 - Explained method
 - added single displays for: n_post, n_subr, #1_keyword,..
 - added left tab (sw) for #flags
 - switched layout to flex
 - orange theme

Known bugs: 0.1.0
- Plot with multiwords throws error
- (SOLVED) callback error for elements in tab 1
- (SOLVED) compression of tabs

Important changes to make: 0.1.0
- Filter on cluster

Aesthetical changes to make: 0.1.0
- Button color
- Center plot options below trend plot
"""









