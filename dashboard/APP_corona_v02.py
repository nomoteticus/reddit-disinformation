# -*- coding: utf-8 -*-

# sudo -lsof -i:8050

#https://plotly.com/javascript/configuration-options/

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
import re

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation'


### Read data

df_boundary = pd.read_csv(rootfold+"/dashboard/data/PCA_SUBM_2D_7SR.csv")

df_day = pd.read_csv(rootfold+"/dashboard/data/app_subr_day.csv")
df_subm = pd.read_csv(rootfold+"/dashboard/data/app_subm_day.csv")
df_flags = pd.read_csv(rootfold+"/dashboard/data/app_flags_large.csv")


df_day['ISFLAG'] = df_day['flag']
df_subm['ISFLAG'] = df_subm['flag']
df_subm['cluster'] = '1'
df_day['week'] = df_day['week'].astype('int')

# Default slider values
flagtypes = list(df_day.loc[:,"disinformation":"other"].columns)

min_week, max_week = df_day['week'].min(), df_day['week'].max()
range_weeks = range(min_week, max_week,4)
if max_week not in range_weeks:
    range_weeks = [*range_weeks] + [max_week]
week_day_df = df_day.groupby('week')['day'].agg(['min','max'])

top20subreddits = list(df_day.subreddit.value_counts().index[:20])

### Init APP
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                suppress_callback_exceptions=True)


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

#html.Table(generate_table(y, max_rows=10))



### Sliders and Buttons
slider_week =dcc.RangeSlider(
                id = 'sl_week',
                min = min_week, max = max_week,
                value = [max_week-4,max_week], step=1,
                marks = {ind:str(ind) for ind in range(min_week, max_week,4)})

checklist_flagtypes = dcc.Checklist(
                id = 'cl_flagtypes', 
                value=flagtypes,
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
    return('%s -> %s' % (week_day_df.loc[input_value[0]].loc['min'],
                         week_day_df.loc[input_value[1]].loc['max']))


### LAYOUT

app.layout = \
    html.Div(
        children=[
            ### hidden
            html.Div(id='filtered_json_week', style={'display': 'none'}),
            
            ### LEFT PANE (filters)
            html.Div([html.H4('Choose week', style = {'padding-top': '50px'}),
                      slider_week,
                      html.Div(id='show_dates', style={'text-align':'center', 'font-size':'150%'}),
                      html.H4('Choose type of flag', style = {'padding-top': '50px'}),
                      checklist_flagtypes,
                      html.H4('Coronavirus only?', style = {'padding-top': '20px'}),
                      radio_coronatopic
                      ],
                     style={'width': '25%', 'display': 'inline-block',
                            'position': 'fixed', 'padding-top': '50px',
                            'background-color':'lightyellow',
                            'height':'1200px'}),
            
            ### RIGHT PANE (tabs)
            html.Div([
                dcc.Tabs(id='tabs-example', value='tab-1', children=[
                    dcc.Tab(label='Informal flags', value='tab-1'),
                    dcc.Tab(label='Submissions', value='tab-2'),
                    dcc.Tab(label='Subreddits', value='tab-3'),
                    dcc.Tab(label='Clustering', value='tab-4'),
                    dcc.Tab(label='Boundaries', value='tab-5')
                ]),
                html.Div(id='tabs-example-content')],
                style={'width': '74%', 'display': 'inline-block',
                       'float':'right', 'padding-left': '50px'})
    ])
    
    
### Content of TABS     
              
@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    ### TAB1 - Flags
    if tab == 'tab-1':
        return html.Div([
            html.H5('Number of results to display: ', style = {"margin-top":"75px"}),
            radio_ntopflags,
            
            html.H3('Most common flags', style = {"margin-top":"50px"}),
            html.Div(id='tbl_flags_freq'),
            
            html.H3('Most recent flags', style = {"margin-top":"50px"}),
            drop_subr20,
            html.Div(id='tbl_flags_recent')
            #generate_table(df_flags[['sent','subreddit']][:100])
        ])
    
    ### TAB2 - Submissions
    elif tab == 'tab-2':
        return html.Div([
                html.H5('Number of submissions to display: ', style = {"margin-top":"75px"}),
                radio_ntopsubs,
                html.H5('Minimum number of flags: '),
                drop_min_nflags,
                html.H5('Subreddits (leave empty if all) '),
                drop_subr20_subm,
                html.H4('Most recently flagged submissions:', style = {"margin-top":"30px"}),
                html.Div(id = 'flagged_subm_list')
                ],
            style={'width': '75%'})
        
    
    ### TAB3 - Subreddits
    elif tab == 'tab-3':
        return html.Div([
            
            html.H4('Subreddits with most flagged submissions', style = {"margin-top":"75px"}),
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
                ])
        ])

    ### TAB4 - clustering
    elif tab == 'tab-4':
        return html.Div([
            html.H4('Clustering of flagged submissions', style = {"margin-top":"75px"}),
            html.H5('Number of clusters (K-means): '),
            html.Div(slider_nclust, style={'width': '50%', "margin-top":"25px"}),
            html.Div(drop_subr20_cluster, style={'width': '50%', "margin-top":"25px"}),
            dcc.Graph(id='plot_kmeans', style={'width': '50%', 'height':'150%'}),
            html.H5('Most common words in each cluster: '),
            html.Div(id='tbl_topwords')
            ])
    
    ### TAB5 - boundary
    elif tab == 'tab-5':
        return html.Div([
            html.H4('Boundary plot', style = {"margin-top":"75px"}),
            html.P('Choose subreddit'),
            drop_boundary_subr,
            html.P('Show boundaries?'),
            show_boundary_lines,
            dcc.Graph(id = 'plot_boundary'),
            html.Div(id = 'click_title')
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
    filtered_df_day = df_day.copy()
    filtered_df_day = filtered_df_day[(filtered_df_day['week'].between(*weekrange))]
    if coronatopic=="yes":
        filtered_df_day = filtered_df_day[filtered_df_day['subm_covid']]
    filtered_df_day['ISFLAG'] = filtered_df_day[flagtypes].copy().sum(axis=1)
    ## Aggregate by week
    filtered_df_week = get_flag_perc(
        filtered_df_day,
        groups = ['subreddit_cat','subreddit','week'])
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
    [Output('tbl_flags_recent', 'children'),
     Output('tbl_flags_freq', 'children')],
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
    return (generate_table(latest_flags_df, max_rows=ntopflags),
            generate_table(common_flags_df, max_rows=ntopflags))

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

def cluster_commonwords(texts, nwords = 5):
    allwords = [w for w in ' '.join(texts.str.lower()).split() if w not in STOP_WORDS and re.search('[a-z]',w)]
    return Counter(allwords).most_common(nwords)

slider_nclust = \
    dcc.Slider(id = 'sl_nclust',
    min=1,  max=10,
    step=1,
    marks={i:str(i) for i in range(1,11)},
    value=1 )  

drop_subr20_cluster = \
    dcc.Dropdown(
        id='dr_subr20_clust', 
        options=[{'label': i, 'value': i} for i in top20subreddits], 
        multi=True, placeholder='Filter by subreddit...')
    
@app.callback(
    [Output('plot_kmeans','figure'),
     Output('tbl_topwords','children')],
    [Input('sl_nclust','value'),
     Input('sl_week', 'value'),
     Input('cl_flagtypes', 'value'),
     Input('rd_coronatopic', 'value'),
     Input('dr_subr20_clust', 'value')])
def plot_kmeans(nclust, weekrange, flagtypes, coronatopic, subr20):
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
        X = filtered[['PC01', 'PC02']]
        cluster = MiniBatchKMeans(n_clusters=nclust, batch_size=1000)
        cluster.fit(X)
        filtered['cluster'] = (cluster.predict(X)+1).astype(str)
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
        cw.append({'cluster':cl, 'submissions':SCL.shape[0],'kewywords': str(cluster_commonwords(SCL,10)) })        
    return fig, generate_table(pd.DataFrame(cw))

### T5: BOUNDARIES

import statsmodels.api as sm
def runlogit(df, dv_string, dimnames = ["dim_x_PCA","dim_y_PCA"]):
    logit_model = sm.Logit(df[dv_string], sm.add_constant(df[dimnames]),disp=0)
    #logit_model.fit()
    return logit_model
def line_equation(x1, logit_model):
    fit = logit_model.fit(disp=0)
    intercept, b1, b2 = fit.params
    return(-1* intercept/b2 - (x1*b1)/b2)

drop_boundary_subr = \
    dcc.Dropdown(
        id='dr_boundary_sr', 
        options=[{'label': i, 'value': i} for i in df_boundary.subreddit.unique()], 
        value = 'healthIT',
        multi=False)

show_boundary_lines = \
    dcc.RadioItems(
        id='rd_showboundaries', 
        options=[{'label': 'yes', 'value':'yes'},{'label': 'no','value':'no'}], 
        value = 'no',
        labelStyle={'display': 'inline-block'})

    
@app.callback(
    Output('plot_boundary','figure'),
    [Input('dr_boundary_sr','value'),
     Input('rd_showboundaries','value')])
def plot_boundary(subreddit, showboundaries):
    Selection = df_boundary[df_boundary['subreddit']==subreddit].copy()
    Selection['nplot'] = Selection.month - min(Selection.month) + 1
    dimnames = ['PC01','PC02']
    dv_string = 'removed_all'
    ###
    fig = px.scatter(Selection, 
                x='PC01', y='PC02',
                color=dv_string,
                title=subreddit + '(' + dv_string + ')',
                color_discrete_sequence = ['red','blue'],
                hover_data=['title'],
                opacity=0.25,
                facet_col='month')
    fig.update_traces(marker=dict(size=5))
    if showboundaries == "yes":
        shapes = [{'type': 'line', 'line_width': 2, 'line_color': "green",
                   'x0': min(df[1].PC01), 'y0': line_equation(min(df[1].PC01), logit_model = runlogit(df[1], dv_string +'_c', dimnames)),
                   'x1': max(df[1].PC01), 'y1': line_equation(max(df[1].PC01), logit_model = runlogit(df[1], dv_string +'_c', dimnames)),
                   'xref': 'x'+str(min(df[1].nplot)), 'yref': 'y'+str(min(df[1].nplot))}
                  for df in Selection.groupby('month')]
        fig['layout'].update(shapes=shapes)
    return fig

@app.callback(
    Output('click_title', 'children'),
    [Input('plot_boundary','selectedData')])
def update_output_div2(input_value):
    if input_value is not None:
        show_id =[html.P(sel['customdata'][0]) for i, sel in enumerate(input_value['points']) if i <100]
        #df_flags.set_index('subm_id').loc[show_id]
        return(show_id)
    else:
        return('Click on any point to investigate')
    #except NameError:
    #    show_id = False
    #    return('')



### Run app
if __name__ == '__main__':
    app.run_server(debug=True)