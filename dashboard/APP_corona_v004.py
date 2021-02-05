# -*- coding: utf-8 -*-

# sudo -lsof -i:8050

#https://plotly.com/javascript/configuration-options/
#https://jsfiddle.net/uarz7s1t/8/

#python ~/Documents/git/reddit-disinformation/dashboard/APP_corona_v03.py

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
#from itertools import chain
from datetime import datetime
import re
#import os

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

# Default slider values
flagtypes = list(df_subr.loc[:,"disinformation":"other"].columns)

min_week, max_week = int(df_subr['week'].min()), int(df_subr['week'].max())
max_week_lst = [max_week]
range_weeks = range(min_week, max_week,4)
if max_week not in range_weeks:
    range_weeks = [*range_weeks] + [max_week]
week_day_df = df_subm.groupby('week')['day'].agg(['min','max'])
week_day_df['label'] = [r[1]['min'][5:7] + '/' + r[1]['min'][8:10]
                        for r in week_day_df.iterrows()]
# + ' - ' + r[1]['max'][5:7] + '/' + r[1]['max'][8:10]

top20subreddits = list(df_subr.subreddit.value_counts().index[:20])

### Init APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/dZVMbK.css']

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)
server = app.server

### Functions

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


def user_link(author):
    return 'http://www.reddit.com/user/'+author

def generate_table_authors(dataframe, max_rows=10, col_link = 'subm_author'):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(html.A(dataframe.iloc[i][col], href = user_link(dataframe.iloc[i][col]))) \
                if col==col_link else html.Td(dataframe.iloc[i][col])\
                for col in dataframe.columns                 
             ]) for i in range(min(len(dataframe), max_rows))]
    )

def revdate(datestring, year = False):
    if year:
        return datetime.strptime(datestring, "%Y-%m-%d").strftime("%b %d, %Y")
    else:
        return datetime.strptime(datestring, "%Y-%m-%d").strftime("%b %d")

#def add_extremities(lst, dfvar):
#    return sorted(list(set(lst).union(set([dfvar.min(), dfvar.max()]))))

def get_xlabs(xrange, nticks =5):    
    rng = xrange[1] - xrange[0]
    ticklst = list(range(xrange[0], xrange[1], rng // nticks+1)) 
    return ticklst + [xrange[1]]
    

#html.Table(generate_table(y, max_rows=10))



### Sliders and Buttons
slider_week =dcc.RangeSlider(
                id = 'sl_week',
                min = min_week, max = max_week,
                value = [max_week-4,max_week], 
                step=1,
                marks = {ind:{'label': week_day_df.loc[ind].loc['label'],
                              'style':{'transform':'rotate(45deg)',
                                       'text-orientation': 'sideways'
                                       }} 
                         for ind in range_weeks }) 
                #range(min_week, max_week + 1, 4) })
                #marks = {ind:str(ind) for ind in (list(range(min_week, max_week, 4)) + [max_week] )} )

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
    return('Selected: from %s to %s' % 
           (revdate(week_day_df.loc[input_value[0]].loc['min'], year = False),
            revdate(week_day_df.loc[input_value[1]].loc['max'], year = True)))


### LAYOUT


style_leftpane = {'width': '25%', 'display': 'inline-block',
                 'position': 'fixed', 'padding-top': '5px',
                 'background-color':'lightyellow',
                 'height':'1200px'}
style_rightpane ={'width': '74%', 'display': 'inline-block',
                  'float':'right', 'padding-left': '50px'}

app.layout = \
    html.Div(
        children=[
            ### hidden
            html.Div(id='filtered_json_week', style={'display': 'none'}),
            
            ### LEFT PANE (filters)
            html.Div([html.Img(src=app.get_asset_url('stroo_v1.png'),
                               style = {'max-width': '100%'}),
                      html.H4('Choose period: ', style = {'padding-top': '50px'}),
                      slider_week,
                      html.Div(id='show_dates', 
                               style={'text-align':'left', 
                                      'padding-top':'20px',
                                      #'padding-bottom':'20px',
                                      'padding-left':'20px',
                                      'color':'darkblue',
                                      'font-size':'115%'}),
                      html.H4('Choose type of flag', style = {'padding-top': '50px'}),
                      checklist_flagtypes,
                      html.H4('Coronavirus only?', style = {'padding-top': '20px'}),
                      radio_coronatopic],
                      style = style_leftpane),
            
            ### RIGHT PANE (tabs)
            html.Div([
                dcc.Tabs(id='tabs-example', value='tab-1', children=[
                    dcc.Tab(label='Informal flags', value='tab-1'),
                    dcc.Tab(label='Submissions', value='tab-2'),
                    dcc.Tab(label='Subreddits', value='tab-3'),
                    dcc.Tab(label='Clustering', value='tab-4'),
                    dcc.Tab(label='Domains', value='tab-5'),
                    dcc.Tab(label='Authors', value='tab-6')
                ]),
                html.Div(id='tabs-example-content')],
                style=style_rightpane)
    ])
    
    
### Content of TABS     
              
@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    ### TAB1 - Flags
    if tab == 'tab-1':
        return html.Div([
            html.H5('Number of results to display: ', style = {"margin-top":"75px"}),
            html.Hr(),
            radio_ntopflags,
            
            html.H3('Most common flags', style = {"margin-top":"50px"}),
            html.Div(id='tbl_flags_common'),
            
            html.H3('Most recent flags', style = {"margin-top":"50px"}),
            drop_subr20,
            html.Div(id='tbl_flags_recent'),
            
            #generate_table(df_flags[['sent','subreddit']][:100])
        ])
    
    ### TAB2 - Submissions
    elif tab == 'tab-2':
        return html.Div([
                html.H5('Number of submissions to display: ', style = {"margin-top":"75px"}),
                html.Hr(),
                radio_ntopsubs,
                html.H5('Minimum number of flags: '),
                drop_min_nflags,
                html.H5('Subreddits (leave empty if all) '),
                drop_subr20_subm,
                html.H4('Most recently flagged submissions:', style = {"margin-top":"30px"}),
                html.Div(id = 'flagged_subm_list'),
                
                ],
            style={'width': '75%'})
        
    
    ### TAB3 - Subreddits
    elif tab == 'tab-3':
        return html.Div([
            
            html.H4('Subreddits with most flagged submissions', style = {"margin-top":"75px"}),
            html.Hr(),
            dcc.Graph(id='plot_subr_treemap'),            
            
            # line plots
            html.H4('Evolution of flagged submissions'),
            html.Div([
                html.Div([
                    html.H5('Per subreddit category'),
                    radio_subr_cat_stat,
                    dcc.Graph(id='plot_subr_line_cat')
                    ], style={'width': '45%', 'display': 'inline-block','float':'left'}),
                html.Div([
                    html.H5('Per subreddit'),
                    radio_subr_20_stat,
                    dcc.Graph(id='plot_subr_line_20') 
                    ], style={'width': '45%', 'display': 'inline-block','float':'right'})
                ]),


        ])

    ### TAB4 - clustering
    elif tab == 'tab-4':
        return html.Div([
            html.H4('Clustering of flagged submissions', style = {"margin-top":"75px"}),
            html.Hr(),
            html.Div([
                html.Div([
                        html.H5('Number of clusters (K-means): '),
                        html.Div(slider_nclust),
                        html.Div(drop_subr20_cluster),
                        dcc.Graph(id='plot_kmeans', style={'height':'150%'})],
                    style = {'width': '50%', "margin-top":"25px",
                             "float":"left"}),
                html.Div([
                        html.H5('Most common words in each cluster: '),
                        html.Div(id='tbl_topwords')],
                    style = {'width': '45%', "margin-top":"25px", 'padding-left':'5%',
                             'float':'left'})],
                style = {'min-width': '100%',
                         'display': 'block',
                         'width': '100%',
                         'float':'left'}),
            html.Span([
                html.Br(),
                html.Span([
                        html.H5('Most relevant submissions for ', 
                                style = { 'display': 'inline-block', 'vertical-align':'top'}),
                        html.Div(drop_clustersubm, 
                                 style = {'min-width':'15%', 'display': 'inline-block', 
                                          'padding-left':"10px"})]),
                html.Span(id='tbl_clustersubm')])
            ])
    
    ### TAB5 - domains
    elif tab == 'tab-5':
        return html.Div([
            html.H4('Most flagged domains', style = {"margin-top":"75px"}),
            html.Hr(),
            html.Div([
                html.Div('Display maximum rows: ', title = 'Number of rows to display'),  
                radio_ntopdomains],
                style={'width': '25%', 'display': 'inline-block', 'margin-bottom':'15px'}),
            html.Div([
                html.Div('Sort by: ', title = 'Number of times author was flagged or the percentage flagged from each author'),  
                html.Div(radio_domainspercent)],
                style={'width': '25%', 'display': 'inline-block','margin-bottom':'15px'}),
            html.Div([
                html.Div('Minimum cases for percentage: ', title = 'Only show percentages if number of submissions is at least this number.'),  
                html.Div(radio_mincasesdomains)],
                style={'width': '50%', 'display': 'inline-block', 'margin-bottom':'15px'}),
            html.Br(),
            drop_subr20_dom,
            html.Div(id='tbl_domains'),
            
            ])

    ### TAB6 - authors
    elif tab == 'tab-6':
        return html.Div([
            html.H4('Most flagged submission authors', style = {"margin-top":"75px"}),
            html.Hr(),
            html.Div([
                html.Div('Display maximum rows: ', title = 'Number of rows to display'),  
                radio_ntopauthors],
                style={'width': '25%', 'display': 'inline-block', 'margin-bottom':'15px'}),
            html.Div([
                html.Div('Sort by: ', title = 'Number of times it was flagged or the percentage flagged from each source'),  
                html.Div(radio_authorspercent)],
                style={'width': '25%', 'display': 'inline-block','margin-bottom':'15px'}),
            html.Div([
                html.Div('Minimum cases for percentage: ', title = 'Only show percentages if number of submissions is at least this number.'),  
                html.Div(radio_mincasesauthors)],
                style={'width': '50%', 'display': 'inline-block', 'margin-bottom':'15px'}),
            html.Br(),
            drop_subr20_auth,
            html.Div(id='tbl_authors'),
            
            ])


### Clean data

def get_flag_perc(df, groups):
    return df.\
        groupby(groups)\
            [['n_subm_kept','ISFLAG']].\
                agg(sum).sort_index().reset_index().\
        assign(perc_FLAG = lambda df: df.ISFLAG/df.n_subm_kept)

### aggregation

@app.callback(
    Output('filtered_json_week', 'children'),
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value')])
def filter_data(weekrange, flagtypes, coronatopic):
    filtered_df_subr = df_subr.copy()
    filtered_df_subr = filtered_df_subr[(filtered_df_subr['week'].between(*weekrange))]
    if coronatopic=="yes":
        filtered_df_subr = filtered_df_subr[filtered_df_subr['subm_covid']]
    filtered_df_subr['ISFLAG'] = filtered_df_subr[flagtypes].copy().sum(axis=1)
    ## Aggregate by week
    filtered_df_week = get_flag_perc(
        filtered_df_subr,
        groups = ['subreddit_cat','subreddit','week'])
    ## Calculate X axis labels
    return filtered_df_week.to_json(orient='split')
        

### UPDATE PLOTS+TABS

### T1 Recent flags

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

@app.callback(
    [Output('tbl_flags_common', 'children'),
     Output('tbl_flags_recent', 'children')],
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('rd_ntopflags', 'value'),
     Input('dr_subr20', 'value')])
def table_flags_latest(weekrange, flagtypes, coronatopic, ntopflags, subr20):
    filtered = df_flags.copy()
    filtered = filtered[(filtered['week'].between(*weekrange))]
    filtered['ISFLAG'] = filtered[flagtypes].copy().sum(axis=1)
    filtered = filtered[filtered['ISFLAG']>0]
    if coronatopic=="yes":
        filtered = filtered[filtered['subm_covid']]
    latest_flags_df = filtered[['subreddit','sent','day']]
    common_flags_df = filtered.sent.value_counts()[:ntopflags].reset_index()
    if subr20 is not None:
        latest_flags_df = latest_flags_df[latest_flags_df.subreddit.str.contains('|'.join(subr20))]
    return (generate_table(common_flags_df, max_rows=ntopflags),
            generate_table(latest_flags_df, max_rows=ntopflags))

### T2: SUBMISSIONS list

radio_ntopsubs = \
    dcc.RadioItems(
        id = 'rd_ntopsubs',
        options=[{'label': x, 'value': x} for x in [10,20,50,100]],
        value=10,
        labelStyle={'display': 'inline-block'})  

drop_subr20_subm = \
    dcc.Dropdown(
        id='dr_subr20_subm', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')
    
drop_min_nflags = \
    dcc.Dropdown(
        id='dr_min_nflags', 
        value = 1,
        options=[{'label': i, 'value': i} for i in [1,2,3,5,10] ], 
        multi=False)
    
@app.callback(
    Output('flagged_subm_list', 'children'),
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('rd_ntopsubs', 'value'),
     Input('dr_subr20_subm', 'value'),
     Input('dr_min_nflags', 'value')])
def update_subm(weekrange, flagtypes, coronatopic, ntopflags, subr20, min_nflags):
    filtered = df_flags.copy()
    filtered = filtered[(filtered['week'].between(*weekrange))]
    filtered['ISFLAG'] = filtered[flagtypes].copy().sum(axis=1)
    filtered = filtered[filtered['ISFLAG']>0]
    if coronatopic=="yes":
        filtered = filtered[filtered['subm_covid']]
    filtered['nflags'] = filtered.groupby('subm_id')['ISFLAG'].transform('count')
    filtered = filtered[filtered['nflags']>=min_nflags]
    if subr20 is not None:
        filtered = filtered[filtered.subreddit.str.contains('|'.join(subr20))]
    latest_subs = filtered.set_index('subm_id')[:ntopflags]
    return generate_submissions(latest_subs)

def generate_submission(subm):
    subm_flat = subm.head(1).squeeze()
    return html.Div(children = [            
                html.A('r/' + subm_flat.subreddit,
                       href = subm_flat.subm_link),
                html.H4(subm_flat.subm_title),
                html.A(subm_flat.link_url,
                       href = subm_flat.link_url),
                #html.Hr(),
                html.P(subm_flat.day),
                html.P('Deletion status: ' + subm_flat.subm_removed),
                html.H5('Flags: ', style={'color': 'red'})] + \
            [html.P(ss, style={'color': 'red'}) for ss in subm.sent],
            style = {'border': '1px solid', 
                    'padding': '5px', 'margin-bottom':'15px', 'margin-top':'15px',
                    'box-shadow': '5px 10px inset',
                    })

def generate_submissions(df_subm):
    return [generate_submission(df_subm.loc[[s]]) for s in df_subm.index.unique()]

### T3: SUBREDDIT plots
# Treemap
@app.callback(
    Output('plot_subr_treemap','figure'),
    [Input('filtered_json_week','children')])
def plot_subr_treemap(filtered_json_week):
    filtered = pd.read_json(filtered_json_week, orient='split')
    ## Aggregate overall
    filtered_agg = get_flag_perc(filtered, groups = ['subreddit_cat','subreddit'])
    fig_subr_treemap = px.treemap(filtered_agg, 
                          path=['subreddit_cat', 'subreddit'], 
                          values='ISFLAG',
                          color='subreddit_cat')
    return fig_subr_treemap

# Line plots

radio_subr_cat_stat = \
    dcc.RadioItems(
        id = 'rd_subr_all_stat',
        options=[{'label':'Frequency (N)', 'value': 'n'},
                 {'label':'Percent of subm.(%)', 'value': 'pct'}],
        value='n', labelStyle={'display': 'inline-block'} ) 
    
@app.callback(
    Output('plot_subr_line_cat','figure'),
    [Input('filtered_json_week','children'),
     Input('rd_subr_all_stat','value')])
def plot_subr_line_cat(filtered_json_week, nperc):
    filtered_df_week = pd.read_json(filtered_json_week, orient='split')
    filtered_df_week_cat = get_flag_perc(filtered_df_week, ['subreddit_cat','week'])
    subr_line_cat = px.line(
                        filtered_df_week_cat,
                        x = "week",
                        y = "ISFLAG" if nperc == "n" else "perc_FLAG",
                        color = "subreddit_cat")
    for l in subr_line_cat.data:
        l.update(mode='markers+lines')
    tickvals0 = get_xlabs((min([min(a.x) for a in subr_line_cat.data]),
                           max([max(a.x) for a in subr_line_cat.data])))
    subr_line_cat.update_layout(xaxis = dict(tickmode = 'array',
                                         tickvals = tickvals0,
                                         ticktext = week_day_df['label'][tickvals0]
                                         ))
    return subr_line_cat

radio_subr_20_stat = \
    dcc.RadioItems(
        id = 'rd_subr_20_stat',
        options=[{'label':'Frequency (N)', 'value': 'n'},
                 {'label':'Percent of subm.(%)', 'value': 'pct'}],
        value='n', labelStyle={'display': 'inline-block'} ) 
    

@app.callback(
    Output('plot_subr_line_20','figure'),
    [Input('filtered_json_week','children'),
     Input('rd_subr_20_stat','value')])
def plot_subr_line_20(filtered_json_week, nperc):
    filtered_df_week = pd.read_json(filtered_json_week, orient='split')
    filtered_df_week_20 = get_flag_perc(filtered_df_week[filtered_df_week.subreddit.isin(top20subreddits)], 
                                        groups = ['subreddit','week'])
    subr_line_20 = px.line(
                        filtered_df_week_20,
                        x = "week",
                        y = "ISFLAG" if nperc == "n" else "perc_FLAG",
                        color = "subreddit")
    for l in subr_line_20.data:
        l.update(mode='markers+lines')
    return subr_line_20



### T4: CLUSTERING

def cluster_commonwords(texts, nwords = 10, onlycorona = "yes"):
    ignore_words = STOP_WORDS if onlycorona=="no" else STOP_WORDS.union(['coronavirus','covid','covid19','covid-19'])
    allwords = [w for w in ' '.join(texts.str.lower()).split() if w not in ignore_words and re.search('[a-z]',w)]
    return ', '.join([word for word,cnt in Counter(allwords).most_common(nwords)])

slider_nclust = \
    dcc.Slider(id = 'sl_nclust',
    min=1,  max=10,
    step=1,
    marks={i:str(i) for i in range(1,11)},
    value=3 )  

drop_subr20_cluster = \
    dcc.Dropdown(
        id='dr_subr20_clust', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')

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
    return [{'label': 'Cluster '+str(i), 'value': i} for i in range(1,nclust+1)]


    
@app.callback(
    [Output('plot_kmeans','figure'),
     Output('tbl_topwords','children'),
     #Output('lst_topwords','value'),
     Output('tbl_clustersubm','children')],
    [Input('sl_nclust','value'),
     Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('dr_subr20_clust', 'value'),
     Input('dr_clustersubm','value')])
def plot_kmeans(nclust, weekrange, flagtypes, coronatopic, subr20, clusterselected):
    filtered = df_subm.copy()        
    filtered = filtered[(filtered['week'].between(*weekrange))]
    filtered['ISFLAG'] = filtered[flagtypes].copy().sum(axis=1)
    filtered = filtered[filtered['ISFLAG']>0]
    if coronatopic=="yes":
        filtered = filtered[filtered['subm_covid']]
    if subr20 is not None:
        filtered = filtered[filtered.subreddit.str.contains('|'.join(subr20))]
    ### Run cluster analysis
    if nclust>1:
        X = filtered[['PC01', 'PC02', 'PC03', 'PC04', 'PC05']]
        cluster = MiniBatchKMeans(n_clusters=nclust, batch_size=1000, random_state=777)
        cluster.fit(X)
        filtered['cluster'] = (cluster.predict(X)+1).astype(str)
        filtered=filtered.sort_values('cluster')
    ### Build plot
    fig = px.scatter(
            filtered, 
            x='PC01', y='PC02',
            color='cluster',
            hover_data=['subm_title'],
            opacity=0.75)
    fig.update_traces(marker=dict(size=5))
    ### Keyword table
    cw=[]
    for cl in range(1,nclust+1):
        SCL = filtered.subm_title[filtered.cluster==str(cl)]
        cw.append({'cluster':cl, 
                   'submissions':SCL.shape[0],
                   'kewywords': str(cluster_commonwords(SCL,20,coronatopic)) })  
    sel_submissions_index = np.argsort(np.inner(X,cluster.cluster_centers_[clusterselected-1,:]))[-20:]
    #keywords_clust = cw#[k['keywords'] for k in cw]
    return fig, generate_table(pd.DataFrame(cw)), generate_table(filtered[['subm_title','subreddit']].iloc[sel_submissions_index])

### T5: DOMAINS

radio_ntopdomains = \
    dcc.RadioItems(
        id = 'rd_ntopdomains',
        options=[{'label': x, 'value': x} for x in [10,20,50]],
        value=10,
        labelStyle={'display': 'inline-block'})  

radio_mincasesdomains = \
    dcc.RadioItems(
        id = 'rd_mincasesdomains',
        options=[{'label': x, 'value': x} for x in [10,20,50]],
        value=10,
        labelStyle={'display': 'inline-block'})  


radio_domainspercent = \
    dcc.RadioItems(
        id = 'rd_domainspercent',
        options=[{'label': x, 'value': x} for x in ['frequency','percent']],
        value='percent',
        labelStyle={'display': 'inline-block'})  
    
drop_subr20_dom = \
    dcc.Dropdown(
        id='dr_subr20_dom', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')


@app.callback(
    Output('tbl_domains', 'children'),
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('rd_ntopdomains', 'value'),
     Input('rd_mincasesdomains', 'value'),
     Input('rd_domainspercent', 'value'),
     Input('dr_subr20_dom', 'value')])
def table_domains(weekrange, flagtypes, coronatopic, ntopdomains, mincasesdomains, domainspercent, subr20):
    filtered = df_domain.copy()
    filtered = filtered[(filtered['week'].between(*weekrange))]
    filtered['ISFLAG'] = filtered.copy()[flagtypes].sum(axis=1)
    filtered = filtered[filtered['ISFLAG']>0]
    if coronatopic=="yes":
        filtered = filtered[filtered['subm_covid']]
    if subr20 is not None:
        filtered = filtered[filtered.subreddit.str.contains('|'.join(subr20))]
    top_domains_df = filtered.groupby('link_domain')[['ISFLAG','n_subm_kept']].agg('sum').reset_index()
    top_domains_df.n_subm_kept[top_domains_df.n_subm_kept<top_domains_df.ISFLAG]=top_domains_df.ISFLAG.copy()[top_domains_df.n_subm_kept<top_domains_df.ISFLAG]
    top_domains_df['percent'] = (top_domains_df.ISFLAG / top_domains_df.n_subm_kept).round(decimals=3)
    ###
    top_domains_n = top_domains_df.copy().sort_values('ISFLAG', ascending=False)
    top_domains_perc = top_domains_df[top_domains_df.n_subm_kept >= mincasesdomains]
    top_domains_perc = top_domains_perc.sort_values('percent', ascending = False)
    if domainspercent == 'frequency':
        return generate_table(top_domains_n, max_rows=ntopdomains)
    else:
        return generate_table(top_domains_perc, max_rows=ntopdomains)


### T6: AUTHORS

radio_ntopauthors = \
    dcc.RadioItems(
        id = 'rd_ntopauthors',
        options=[{'label': x, 'value': x} for x in [10,20,50]],
        value=10,
        labelStyle={'display': 'inline-block'})  

radio_mincasesauthors = \
    dcc.RadioItems(
        id = 'rd_mincasesauthors',
        options=[{'label': x, 'value': x} for x in [10,20,50,100]],
        value=10,
        labelStyle={'display': 'inline-block'})  


radio_authorspercent = \
    dcc.RadioItems(
        id = 'rd_authorspercent',
        options=[{'label': x, 'value': x} for x in ['frequency','percent']],
        value='percent',
        labelStyle={'display': 'inline-block'})  
    
drop_subr20_auth = \
    dcc.Dropdown(
        id='dr_subr20_auth', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')


@app.callback(
    Output('tbl_authors', 'children'),
    [Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('rd_ntopauthors', 'value'),
     Input('rd_mincasesauthors', 'value'),
     Input('rd_authorspercent', 'value'),
     Input('dr_subr20_auth', 'value')])
def table_authors(weekrange, flagtypes, coronatopic, ntopauthors, mincasesauthors, authorspercent, subr20):
    filtered = df_suspect.copy()
    filtered = filtered[(filtered['week'].between(*weekrange))]
    filtered['ISFLAG'] = filtered.copy()[flagtypes].sum(axis=1)
    filtered = filtered[filtered['ISFLAG']>0]
    if coronatopic=="yes":
        filtered = filtered[filtered['subm_covid']]
    if subr20 is not None:
        filtered = filtered[filtered.subreddit.str.contains('|'.join(subr20))]
    top_authors_df = filtered.groupby('subm_author')[['ISFLAG','n_subm_kept']].agg('sum').reset_index()
    top_authors_df.n_subm_kept[top_authors_df.n_subm_kept<top_authors_df.ISFLAG]=top_authors_df.ISFLAG.copy()[top_authors_df.n_subm_kept<top_authors_df.ISFLAG]
    top_authors_df['percent'] = (top_authors_df.ISFLAG / top_authors_df.n_subm_kept).round(3)
    ###
    top_authors_n = top_authors_df.copy().sort_values('ISFLAG', ascending=False)
    top_authors_perc = top_authors_df[top_authors_df.n_subm_kept >= mincasesauthors]
    top_authors_perc = top_authors_perc.sort_values('percent', ascending = False)
    if authorspercent == 'frequency':
        return generate_table_authors(top_authors_n, max_rows=ntopauthors)
    else:
        return generate_table_authors(top_authors_perc, max_rows=ntopauthors)

### Run app
if __name__ == '__main__':
    app.run_server(host = '0.0.0.0', debug=True)
