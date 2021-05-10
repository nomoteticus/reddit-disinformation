#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 18:15:17 2021

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


rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/dashboard'
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


##### //
##### ELEMENTS 
##### //


navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Title", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem([html.H6("Flags"), html.Div('flaaag')], href="#"),
                dbc.DropdownMenuItem("Similar posts", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="This is a long title of an article flagged as false",
    brand_href="#",
    color="primary",
    dark=True,
)


##### LAYOUT 
##### //

app.layout = \
    html.Div(
            [
                fd.Row(
                    [
                        fd.Item(navbar, shadow=True)                    
                    ]
                )
                

                # fd.Col([
                #     fd.Item(
                #     fd.Row(
                #         [
                #         fd.Item('asdaaaaaaaaaaaaaaa', shadow=True),
                #         fd.Item('asd', shadow=True),
                #         ], 
                #         size=6, stretch=False,
                #         )),
                #     fd.Col(
                #         [
                #         fd.Item([html.P('X'),html.P('X'),html.P('X'),html.P('X')]),
                #         fd.Item([html.P('O'),html.P('O'),html.P('O')])
                #         ],
                #         size = 6
                #         )
                #     ]),
            ]
        )


##### RUN APP
#####

if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)
