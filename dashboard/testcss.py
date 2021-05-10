#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:21:52 2021

@author: j0hndoe
"""


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
from datetime import datetime
import re
import sys
#import os
import pathlib


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
                suppress_callback_exceptions=True)
server = app.server

app.layout3 = \
    html.Div(
        className = 'page',
        children = 'aaas')

app.layout = \
    html.Div(
        className = 'page',
        children = [
            ###s
            html.Div(
                className = 'header',
                children = [
                    html.Div(className = 'item north west',
                             children = [
                                  html.Img(src=app.get_asset_url('stroo_v1.png'),
                                           style = {'max-width': '100%'}),
                                  html.Span(
                                          html.H5(html.B('Is it true?')),
                                          style = {'text-align': 'center',
                                                   'font-size': '70%'})
                                  ]),
                    html.Div(className = 'item north east',
                             children = [ html.Div([
                                                html.Span("Monitoring perceived disinformation on Reddit ",
                                                          className = 'dates_banner'),
                                                html.Span(id='show_dates',
                                                          className = 'dates_banner_bold')],
                                                className = 'title_ne'),
                                          html.Div(
                                            'slider_week',
                                            className = 'title_ne')]
                             )]),
            html.Div(
                className = 'layout',
                children = [
                    html.Div(className = 'item south west',
                             children = 'dsa'),
                    html.Div(className = 'item south east',
                             children = 'dsa')])
  ])
    
    
    
#####
##### RUN APP
#####

if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)



children = [
                            dcc.Tabs(id='tabs-example', value='tab-1', 
                                     children=[
                                        dcc.Tab(label='Highlights', value='tab-1'),
                                        dcc.Tab(label='Submissions', value='tab-2'),
                                        dcc.Tab(label='Subreddits', value='tab-3'),
                                        dcc.Tab(label='Clustering', value='tab-4'),
                                        dcc.Tab(label='Domains', value='tab-5'),
                                        dcc.Tab(label='Method', value='tab-6')
                                        ]),
                            html.Div([html.P(html.Br()),
                                      html.P(fd.texts['highlights']['description1']),
                                      html.P(fd.texts['highlights']['description2']),
                                      html.H5('Posts flagged by Reddit users as false information'),
                                      ###
                                      html.Span(id='show_dates2'),
                                      html.Div([
                                          fd.info_box("# Flagged posts:","x"),
                                          fd.info_box("# Subreddits:","x"),
                                          fd.info_box("# Flagged domains:","x"),
                                          fd.info_box("# Flagged users:","x")
                                          ],
                                          className="row container-display"),
                                      ###
                                      html.H5('Most recent post flagged by at least 2 users'),
                                      html.P(fd.texts['highlights']['top']),
                                      ###
                                      html.H5('Evolution of flagged posts'),
                                      html.P(fd.texts['highlights']['top']),
                                      html.Div([
                                          dcc.Graph(id='plot_evol_line_n', 
                                                    className = 'width30'),
                                          dcc.Graph(id='plot_evol_line_pct', 
                                                    className = 'width30')],
                                          className = 'flex-display'
                                          ),                                  
                                      ###
                                      html.H5('Common keywords in flagged posts'),
                                      html.P(fd.texts['highlights']['top']),
                                      ###
                                      html.H5('Common domains in flagged posts'),
                                      html.P(fd.texts['highlights']['top'])                                  
                                    ],className = 'tab_div')
                            ]