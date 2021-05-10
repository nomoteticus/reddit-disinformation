#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:04:31 2021

@author: j0hndoe
"""

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly_express as px

from spacy.lang.en.stop_words import STOP_WORDS
from urllib.parse import urlparse
from collections import Counter
from datetime import datetime
import pandas as pd
import re



### CONTAINERS

def Container(children, className, classAdd = False, size = False, shadow=False, stretch=True):
    if not stretch:
        className = className + ' nostretch'
    if shadow:
        className = className + ' shadow_container'
    if size:
        className = className + ' flx' + str(size)
    if classAdd:
        className = className + ' ' + classAdd
    return html.Div(children=children, className = className)


def Row(children, className='cont_generic', size = False, classAdd = False, shadow=False, stretch=True):
    return Container(children, className, classAdd, size, shadow, stretch)

def Col(children, className='cont_generic v', size = False, classAdd = False, shadow=False, stretch=True):
    return Container(children, className, classAdd, size, shadow, stretch)

def Item(children, className='panel', size = False, classAdd = False, shadow=False, stretch=True):
    return Container(children, className, classAdd, size, shadow, stretch)

def ToDo(text):
    return html.P(children = [html.Strong('TO DO: '), text], 
                  className = 'hidden_red')
        
    
### SLIDERS
        
def generate_slider_df(df_subr, df_subm):
    min_week, max_week = int(df_subr['week'].min()), int(df_subr['week'].max())
    n_weeks = max_week - min_week
    n_step = 1 + n_weeks // 14
    range_weeks = range(min_week, max_week, n_step)
    if max_week not in range_weeks:
        range_weeks = [*range_weeks] + [max_week]
    week_day_df = df_subm.groupby('week')['day'].agg(['min','max','nunique'])
    week_day_df['label'] = [r[1]['max'][5:7] + '/' + r[1]['max'][8:10]
                            for r in week_day_df.iterrows()]
    return week_day_df

def dcc_day_slider(df_subr, df_subm, weeks_shown = 4, step = 1, oblique = False):
   min_week, max_week = int(df_subr['week'].min()), int(df_subr['week'].max())
   n_weeks = max_week - min_week
   n_step = step + n_weeks // 14
   range_weeks = range(min_week, max_week, n_step)
   if max_week not in range_weeks:
        range_weeks = [*range_weeks] + [max_week]
   week_day_df = generate_slider_df(df_subr, df_subm)
   text_style = {'transform':'rotate(45deg)',
                 'text-orientation': 'sideways'} if oblique else {}
   return dcc.RangeSlider(
                id = 'sl_week', 
                min = min_week, max = max_week,
                value = [max_week - weeks_shown, 
                         max_week], 
                marks = {ind:{'label': week_day_df.loc[ind].loc['label'],
                              'style': text_style } 
                         for ind in range_weeks },
                className = 'time_slider' ) 



### CHECKLISTS




### CLEANING

def get_flag_perc(df_all_subm, df_subm, week_day_df, groups):
    df_all_subm_gr = df_all_subm.groupby(groups)[['n_subm_all','n_subm_kept']].agg(sum)
    return df_subm.groupby(groups).agg(n_flags = ('ISFLAG',sum)).\
                join(df_all_subm_gr). \
                join(week_day_df['nunique']). \
                reset_index(). \
                assign(n_day =        lambda D: round(D.n_flags/D['nunique']).astype('int') ). \
                assign(perc_FLAG    = lambda D: D.n_flags/D.n_subm_kept) . \
                assign(perc_FLAG_pr = lambda D: round(100*D.perc_FLAG,1))

def revdate(datestring, year = False):
    if year:
        return datetime.strptime(datestring, "%Y-%m-%d").strftime("%b %d, %Y")
    else:
        return datetime.strptime(datestring, "%Y-%m-%d").strftime("%b %d")

def get_xlabs(xrange, nticks =5):    
    rng = xrange[1] - xrange[0]
    ticklst = list(range(xrange[0], xrange[1], rng // nticks+1)) 
    return ticklst + [xrange[1]]

# def simple_url(text):    
#     urlparse('www.abc.co.uk/cucumeu')
    
def simple_url(text, drop_http = False):
    found = re.search('^.+?[a-z]\\/', text, re.IGNORECASE)
    if found:
        if drop_http:
            return re.sub('https?\\:\\/\\/','', found.group(0)[:-1])
        else:
            return found.group(0)[:-1]
    else:
        return('-')

def remove_http_www(link):
    no_http = re.sub('https?\\:\\/\\/', '', link)
    no_www  = re.sub('www\\.','',no_http)
    no_slash= re.sub('/.*','',no_www)
    return no_slash
    
def printdf(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def shorten_titles(titles, nchar):
    return [t[:nchar]+'...' for t in titles]
    
    
    

### COMMON WORDS
# Lowercase?

def df_commonwords(texts, nwords = 100, normalize = True):
    ignore_words = STOP_WORDS.union(['new','says'])
    allwords = pd.Series([w.lower() for w in ' '.join(texts).split() if w.lower() not in ignore_words and re.search('[a-z]',w)])
    word_counter = allwords.value_counts(normalize=normalize).reset_index().head(nwords)
    word_counter.columns = ['word','freq']
    if normalize:
        word_counter['freq'] = round(100*word_counter['freq'],2)
    return pd.DataFrame(word_counter)

def cluster_commonwords(texts, nwords = 10):
    ignore_words = STOP_WORDS.union(['new','says'])
    allwords = [w.lower() for w in ' '.join(texts).split() if w.lower() not in ignore_words and re.search('[a-z]',w)]
    return [dbc.Badge(word, color = "Khaki", className="mr-1") \
            for word,cnt in Counter(allwords).most_common(nwords)]
    #return ', '.join([word for word,cnt in Counter(allwords).most_common(nwords)])

def print_cluster_commonwords(df, cl, n = 10):
    commonwords = cluster_commonwords(df.subm_title[df.cluster==str(cl)],nwords=n)
    return Row(
                [Item(html.Strong('CL'+str(cl)+': '), size =1),
                 Item(html.Span(commonwords), size = 10)],
                classAdd = 'cluster_sep',
               )


### LAYOUTS

def info_box(header, content_id, big_number="bigger_number"):
   return  html.Div(
                [html.P(header), 
                 html.Div(id=content_id,
                           className=big_number)],
                 className="shadow_container info_box")



### TABLES

def generate_table(dataframe, max_rows=10, className = 'normaltable'):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))],
        className = className
    )



### SUBMISSIONS

def generate_submission(subm, class_name):
    subm_flat = subm.head(1).squeeze()
    return html.Div(# Submission container
                    children = [           
                        html.Div([
                            html.A('r/' + subm_flat.subreddit,
                                   href = subm_flat.subm_link,
                                   className = 'subcontainer_subm subreddit'),
                            html.Span(subm_flat.day,
                                      className = 'subcontainer_subm date'),
                            html.A(subm_flat.domain,
                                   href = subm_flat.link_url,
                                   className = 'subcontainer_subm domain')     
                            ],
                            className = 'subcontainer_subm'
                            ),
                        html.H6(subm_flat.subm_title,
                                className = 'subm_title'),
                        # Flags container
                        html.Div([
                            html.Div('Flags: ', 
                                    className= 'cont_subm_flags left'),
                            html.Div(
                                html.Div(
                                    [html.P("• "+ss, className = 'cont_subm_flags right item') \
                                     for ss in subm.sent]),
                            className = 'cont_subm_flags right')
                        ],
                        className = 'cont_subm_flags')
                    ],
                    className = class_name) #'shadow_container')


def generate_submission_subm(subm, s, df_subm_full, euclid, class_name, button_code):
    similar_posts_df = find_similar_posts(df_subm_full, s, euclid)
    subm_flat = subm.head(1).squeeze()
    return html.Div(# Submission container
                    children = [           
                        html.Div([
                            html.A('r/' + subm_flat.subreddit,
                                   href = subm_flat.subm_link,
                                   className = 'subcontainer_subm subreddit'),
                            html.Span(subm_flat.day,
                                      className = 'subcontainer_subm date'),
                            html.A(subm_flat.domain,
                                   href = subm_flat.link_url,
                                   className = 'subcontainer_subm domain')     
                            ],
                            className = 'subcontainer_subm'
                            ),
                        Row([
                            Item(html.Div(subm_flat.subm_title,
                                          className = 'subm_title bold smallpadding truncate'),
                                 size = 11),
                            Item(dbc.Button("▼Analyze▼",
                                             id="collapse-button-flags"+str(button_code), 
                                             className="mr-1 panel smallpadding", 
                                             color="danger"))
                            ],
                            classAdd = 'cont_subm_title'),
                        # Flags container
                        dbc.Collapse(
                            [
                            html.Div([
                                html.Div('Flags: ', 
                                        className= 'subm_flags left'),
                                html.Div(
                                    html.Div(
                                        [html.P("• "+ss, className = 'cont_subm_flags right item') \
                                         for ss in subm.sent]),
                                className = 'cont_subm_flags right')
                                ],
                                className = 'cont_subm_flags'),
                            html.Div('Similar posts on Reddit flagged as false information:', 
                                     className= 'subm_title'),
                            generate_table(similar_posts_df, className ='small')
                            ],
                        id = 'collapse_flags'+str(button_code)
                        )
                    ],
                    className = class_name)

        
def generate_submissions(df_flags, n, lim, class_name):
    df_flags_indx  = df_flags.copy().set_index(['subm_id'])
    df_flags_slice = df_flags_indx.loc[df_flags_indx.index.unique()[:n]]
    submissions_shown = [generate_submission(df_flags_slice.loc[[s]], class_name = class_name) for \
                         s in df_flags_slice.index.unique()[:lim]]
    submissions_hidden = dbc.Collapse(
        [generate_submission(df_flags_slice.loc[[s]], class_name = class_name) for \
                             s in df_flags_slice.index.unique()[lim:]],
        id = 'collapse')
    return submissions_shown + [submissions_hidden]    


def generate_submissions_subm(df_flags, df_subm_full, euclid, n, lim, class_name):
    df_flags_indx  = df_flags.copy().set_index(['subm_id'])
    df_flags_slice = df_flags_indx.loc[df_flags_indx.index.unique()[:n]]
    submissions_all = [generate_submission_subm(df_flags_slice.loc[[s]], 
                                                s,
                                                df_subm_full, euclid,
                                                class_name = class_name,
                                                button_code = button_code) for \
                       button_code,s in enumerate(df_flags_slice.index.unique())]
    return submissions_all


def find_similar_posts(df_subm, index_subm, euclid, threshold = 0.005, max_subm = 20):
    similar = euclid.loc[index_subm].sort_values().head(max_subm)
    top_k = df_subm[['day','subm_title','subreddit','domain']].loc[similar[similar<threshold].index]
    top_k['subm_title'] = shorten_titles(top_k['subm_title'],60)
    return top_k.sort_values('day')



#####
##### PLOTS
#####

def format_plots(fig, uniform_col = True):
    fig_mod = fig
    if 'marker' in fig_mod.__dict__['_data'][0].keys() and uniform_col:
        fig_mod.update_traces(marker_color='brown')
    if 'line' in fig_mod.__dict__['_data'][0].keys() and uniform_col:
        fig_mod.update_traces(line_color = 'brown')
    fig_mod.update_xaxes(linecolor='black',gridwidth=0)
    fig_mod.update_layout(plot_bgcolor='LightGoldenRodYellow',
                          paper_bgcolor='LightGoldenRodYellow',
                          margin=dict(l=20, r=20, t=20, b=20)
                          )
    return fig_mod

def plot_highl_evo(df, week_day_df, nperc):
    evol_line = px.line(df,
                        x = "week",
                        y = nperc,
                        width = 300, height=250,
                        )
    # style plot
    ytitle = "Number of flagged posts / day"
    if nperc == "n_flags":
        ytitle = "Number of flagged posts / week"
    elif nperc == "perc_FLAG_pr":
        ytitle = "Percent of flagged posts (%)"
    for l in evol_line.data:
        l.update(mode='markers+lines')
    tickvals0 = get_xlabs((min([min(a.x) for a in evol_line.data]),
                           max([max(a.x) for a in evol_line.data])))
    evol_line.update_yaxes(rangemode="tozero", linecolor='black',gridwidth=1)
    evol_line.update_layout(xaxis = dict(tickmode = 'array',
                                         tickvals = tickvals0,
                                         ticktext = week_day_df['label'][tickvals0]
                                         ),
                            xaxis_title="Month/Day of 2021",
                            yaxis_title= ytitle
                            )
    return format_plots(evol_line)


def plot_highl_word(dfw):
    fig = px.bar(dfw, 
                 y="word", 
                 x="freq", 
                 height = 300, width = 300,
                 orientation='h')
    fig = format_plots(fig)
    # style plot
    fig.update_layout(xaxis_title="Word Frequency (%)",
                      yaxis_title= "Keyword" )
    return format_plots(fig)


def plot_two_keywords(dfw, word1, word2, week_day_df):                   
    fig = px.line(dfw,
                  x="week", 
                  y="freq", 
                  color="word",
                  height = 300, width = 300
                  )
    # style plot
    tickvals0 = get_xlabs((min([min(a.x) for a in fig.data]),
                           max([max(a.x) for a in fig.data])))
    fig.update_layout(xaxis = dict(tickmode = 'array',
                                   tickvals = tickvals0,
                                   ticktext = week_day_df['label'][tickvals0]
                                   ),
                      xaxis_title="Word Frequency (%)",
                      yaxis_title= "Word frequency" )
    return format_plots(fig, uniform_col=False)


def plot_clusters(df):
    df['Title'] = shorten_titles(df['subm_title'],50)
    fig = px.scatter(
            df, 
            x='PC01', y='PC02',
            color='cluster',
            #hover_data=['subm_title'],
            hover_data = {'cluster':True,'Title':True,
                          'PC01':False,'PC02':False},
            opacity=0.75)
    fig.update_traces(marker=dict(size=5))
    return format_plots(fig, uniform_col=False)


def plot_evo_clusters(dfw, week_day_df):
    fig = px.line(dfw,
                  x="week", 
                  y="flag", 
                  color="cluster"
                  )
    tickvals0 = get_xlabs((min([min(a.x) for a in fig.data]),
                           max([max(a.x) for a in fig.data])))
    fig.update_layout(xaxis = dict(tickmode = 'array',
                                   tickvals = tickvals0,
                                   ticktext = week_day_df['label'][tickvals0]
                                   ),
                      xaxis_title="Week",
                      yaxis_title="% of flagged submissions in cluster" )
    return format_plots(fig, uniform_col=False)

### TEXTS

texts = \
    {
    'highl':
        {
        'top':         'H3ll0 World!',
        'description1':'This dashboard shows Reddit posts flagged by Reddit users as false information in the comments.',
        'description2':'The rule-based method of identifying user flags is described in the "Method" tab.'
        },
    'subm':{
        'desc1':  'Here you can inspect each submission, and find similar posts across Reddit.',
        'desc2a': 'You can filter submissions by keywords, linked domains, and subreddits.'
        },
    'methd':{
        'strt':['This dashboard was created by a team at the University of Mannheim, Germany (Faculty of Sociology, Chair of Statistics and Methodology.)',
                """The dashboard can serve as a tool for
                  researchers, journalists, or fact checkers, by helping them filter and 
                  analyze the content and spread of information on Reddit.""",
                  'The dashboard is named STROO? (“Is it true?”), as a nod to the Reddit mascot, the alien SNOO'
                ],
        'col': [
               'Submissions and comments are collected from Reddit via the Pushshift API with Python 3.6 / package PSAW.',
               'Collection started in March 2020 and is daily updated; only 2021 data shown here.',
               'Subreddits were chosen to satisfy the following criteria:',
               #
               'Only submissions with comments are included; only 1000 first level comments are extracted',
               ],
        'subr_criteria':
                [
                'the primary language is English',
                'minimum of 10,000 subscribers',
                'most submissions share links to external websites, not self posts, images, videos',
                'external websites shared are news websites, not images, videos, NSFW',
                'there is at least one comment per post (in average).'
                ],
        'flg':  [
                'An informal flag is a comment replying to a submission suspected by a Reddit user to present false information.',
                """Detecting flagged submissions requires searching for specific sentences
                in the comment (such as "This looks like fake news", or "clickbait!").
                The NLP processing pipeline looks like this:
                """
                ],
        'keyw': [
                'A regex pattern is used to extract comments that contain specific words:',
                'FAKE NEWS: fake, false, bogus, falsehood, fabrication',
                'MISINFORMATION: misinformation, disinformation, malinformation',
                'CLICKBAIT: clickbait, misleading, inaccurate, editorialized, sensationalized',
                'UNRELIABLE: unreliable, untrustworthy, unverified',
                'BULLSHIT: bs, bullshit',
                'PROPAGANDA: propaganda'
                ],
        'patt': [
                """Comments were split into sentences. Then, we created a set of patterns, 
                based on a vocabulary and grammar of the informal flag, and checked if 
                they match each sentence.""",
                'Patterns are based on keywords, POS tags and sentence dependencies.',
                'More than 20 patterns were used. Here is an example for each:',
                #
                '<This is fake news> would match, but not <Fake news means false information>.',
                #
                'If at least one sentence matches at least one pattern, it is considered a flag.',
                'The method is not perfect (>10% of patterns are not flags), but it is precise enough.',
                'Precision of detecting flags is similar to that obtained by manual coding + machine learning',
                'You can change the "definition" of a flag by using the left side menu.'
                ],
        'cls':  [
                'Submission titles were run through the Universal Sentence Encoder (link) to produce sentence embeddings.',
                'The >500 embedding variables were reduced to five Principal Components using PCA',
                'The 5 PCA components were used to compute a similarity measure between submission titles.',
                'The similarity measure was then used in a K-means cluster analysis. You can select the desired number of clusters.'
                ],
        'src':  [
                'The websites that the flagged submissions linked to (e.g. www.wsj.com) were classified into trustworthy/reliable, untrustworthy/unreliable, and unclear/unknown',
                'The classification uses existing databases, such as:',
                'Wikipedia: list of perennial sources',
                'Politifact - Fake News Almanac',
                'OpenSources: False, Misleading, Bias, Clickbait-y, and Satirical Sources',
                'NewsGuard: COVID-19 disinformation tracker'
                ],
        'end':  [
                'Dashboard was built using Python 3.6; pattern matching with spaCy; interface and charts: Dash/Plotly',
                'The dash app is built on top of Flask, served using Gunicorn and Nginx.',
                'You can read more in a peer reviewed article in the journal Information, here.'
                ]
        }
    }