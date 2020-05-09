#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:11:42 2020

@author: j0hndoe
"""
import pandas as pd
from collections import Counter
import os
from urlextract import URLExtract 
import re
import numpy as np
from datetime import datetime

## Generic functions

def has_url(text):
    return URLExtract().has_urls(str(text))
def CounterProc(x):
    c = Counter(x)
    return [{k:round(c[k]/sum([c[k] for k in c]),2)} for k in c]
def CounterProcOrdered(x):
    c = Counter(x).most_common()
    df = pd.DataFrame(c, columns = ['var','n'])
    df['proc'] = round(df['n'] / sum(df['n']),2)
    return df
def remove_spaces(txt):
    return(" ".join(str(txt).split()))


## Define classes for Subreddit, Submission, Comment

class reddit_subr(pd.DataFrame):
    
    @property
    def _constructor(self):
        return reddit_subr
    
    def selected_subreddits(self):
        return self.query('~keep.isnull()').query('keep')
        
    def selected_columns(self):
        return self[['subreddit','category','subscribers_K','total7d']]
    
    def rename_subreddits(self):
        return self.rename(columns = {'subscribers_K':'sr_subscribers', 'category':'sr_category','total7d':'sr_ncovid'})

class reddit_subm(pd.DataFrame):
   
    @property
    def _constructor(self):
        return reddit_subm
    
    def new_names(self):
        return(self.rename(columns = {'id':'subm_id',
                             'created_utc':'subm_created',
                             'full_link':'subm_link',
                             'author':'subm_author',
                             'score':'subm_score',
                             'removed_by_category':'subm_removed',
                             'is_self':'subm_is_self',
                             'selftext':'subm_selftext',
                             'num_comments':'subm_ncomm',
                             'title':'subm_title',
                             'url':'subm_url',
                             'domain':'subm_domain',
                             'link_flair_text':'subm_flair'}))
    def new_order(self):
        return(self[['subreddit','subm_id','subm_title','subm_created',\
                  'subm_url','subm_domain','subm_author',\
                  'subm_score','subm_ncomm','subm_flair','subm_link']])

class reddit_comm(pd.DataFrame):
   
    @property
    def _constructor(self):
            return reddit_comm

    def new_names(self):
        return self.rename(columns = {'id':'comm_id',
                                     'created':'comm_created',
                                     'link_id':'subm_id',
                                     'parent_id':'comm_parent_id',
                                     #'body':'comm_body',
                                     'author':'comm_author',
                                     'score':'comm_score',
                                     'collapsed':'comm_collapsed'})
    
    def define_users(self):
        return self.assign(comm_author_class = lambda df: df['distinguished'].fillna('user')).\
                    assign(comm_body = lambda df: df['body'].astype(str))
        
    def new_order(self):
        return self[['subreddit','subm_id','comm_id',\
                      'comm_body', 'comm_created','comm_parent_id','comm_author',\
                      'comm_score','comm_collapsed','comm_author_class']]


### UNIFICATION functions

def unite_subm_subr(SS,SR, condition_keep_subm, condition_subm_regex ):
    """Function to unite submissions and subreddits"""
    return(\
        ### Read Submissions
        reddit_subm(SS).\
             ### Clean Submissions
             query(condition_keep_subm).\
                    query(condition_subm_regex).\
                    new_names().new_order().\
                    sort_values(['subreddit','subm_id','subm_created']).\
                    set_index(['subreddit','subm_id']).\
        ### Join Subreddits
        join(reddit_subr(SR).\
             ### Clean Subreddits
             selected_subreddits().\
             selected_columns().\
             rename_subreddits().\
             sort_values(['subreddit']).\
             set_index('subreddit'),
            how = 'left'))

def unite_all(CC,SS,SR, condition_keep_subm, condition_subm_regex):
    """Function to unite comments, submissions and subreddits"""
    return(\
        ### Read Comments
        reddit_comm(CC).\
            ### Clean Comments
            new_names().define_users().new_order().\
            sort_values(['subreddit','subm_id','comm_created']).\
            set_index(['subreddit','subm_id','comm_id']).\
        ### Join Submissions
        join(reddit_subm(SS).\
             ### Clean Submissions
             query(condition_keep_subm).\
                    query(condition_subm_regex).\
                    new_names().new_order().\
                    sort_values(['subreddit','subm_id','subm_created']).\
                    set_index(['subreddit','subm_id']),
            how = 'inner').\
        ### Join Subreddits
        join(reddit_subr(SR).\
             ### Clean Subreddits
             selected_subreddits().\
             selected_columns().\
             rename_subreddits().\
             sort_values(['subreddit']).\
             set_index('subreddit'),
            how = 'left'))