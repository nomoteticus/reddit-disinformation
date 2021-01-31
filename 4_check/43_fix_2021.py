#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 22:01:34 2021

@author: j0hndoe
"""


from psaw import PushshiftAPI
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import timeout_decorator
import time
import os
import re
import sys
import numpy as np
import logging

#%pwd

api = PushshiftAPI()

#rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)
rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'

sys.path.append(rootfold+"/2_findflags/all_subreddits")
import functions_pos_match as fpm


### SUBM

S = pd.read_csv(rootfold+'/output/SUBM_2020_01.csv', lineterminator='\n', parse_dates=['created_utc'])

from_2020 = S.id[S.created_utc < '2021-01-01']
from_2021 = S.id[S.created_utc >= '2021-01-01']
#set(from_2020).intersection(set(from_2021))

S_2020 = S[S.id.isin(from_2020)]
S_2021 = S[S.id.isin(from_2021)]

S_2020.to_csv(rootfold+'output/replace/'+'SUBM_2020_01.csv', index = False)
S_2021.to_csv(rootfold+'output/replace/'+'SUBM_2021_01.csv', index = False)



### COMM

C = pd.read_csv(rootfold+'/output/COMM_2020_01.csv', lineterminator='\n')

C_2020 = C[C.link_id.isin(from_2020)]
C_2021 = C[C.link_id.isin(from_2021)]

C_2020.to_csv(rootfold+'output/replace/'+'COMM_2020_01.csv', index = False)
C_2021.to_csv(rootfold+'output/replace/'+'COMM_2021_01.csv', index = False)

c_from_2020 = C_2020.id
c_from_2021 = C_2021.id



### MATCH

M = pd.read_csv(rootfold+'/output/MATCH_2020_01.csv', lineterminator='\n')

M_2020 = M[M.comm_id.isin(c_from_2020)]
M_2021 = M[M.comm_id.isin(c_from_2021)]

M_2020.to_csv(rootfold+'output/replace/'+'MATCH_2020_01.csv', index = False)

patterns = [fpm.nlp.vocab.strings[match_id] for match_id in fpm.flagtypematcher._patterns.keys()]

match_file = 'MATCH_2021_01.csv'
print('Started comm file: %s',match_file)
total_sent = 0
sent_df = pd.DataFrame([], columns = ['comm_id','sent_id','sent']+patterns)
sent_df.to_csv(rootfold+'/output/replace/'+match_file, index = False)
existing_comments = []


### Start scraping
comm_generator = pd.read_csv(rootfold+'/output/replace/COMM_2021_01.csv', chunksize=100000)
for chunk_id, COMM in enumerate(comm_generator):
    COMM = COMM[['subreddit','link_id','id','parent_id','body']]
    COMM.body = COMM.body.astype(str).str.lower()
    ### Remove submissions that are specifically on the topic of disinformation
    COMM = COMM[COMM.body.str.contains(fpm.regex_flag)]
    ### Remove comments that might be irony or sarcasm: "\s", "fake" news
    COMM = COMM[~COMM.body.str.contains(fpm.sarcasm_and_irony_regex)]
    ### Processing only new comments
    remaining_comments = set(COMM.id) - set(existing_comments)
    remaining_comments = [c for c in COMM.id if c in remaining_comments]
    ### Apply matcher
    print('Trying %d comments', len(remaining_comments))
    if len(remaining_comments)>0:
        sent_dict = []
        #total_sent = 0
        COMM = COMM[COMM.id.isin(remaining_comments)]
        for doc_index, doc in enumerate(fpm.nlp.pipe(COMM.body, n_process=10, n_threads = 20, disable = ["ner"])):
            for sent_id, sent in enumerate(doc.sents):
                matched_keywords = sent._.flagtypematched
                if matched_keywords:
                    sent_dict_entry = dict(zip(matched_keywords, [1]*len(matched_keywords)))
                    sent_dict_entry['comm_id']=COMM.id.iloc[doc_index]
                    sent_dict_entry['sent_id']=sent_id
                    sent_dict_entry['sent']=sent.text
                    sent_dict.append(sent_dict_entry)
        ### add to current dataframe
        sent_df_current = pd.DataFrame(sent_dict, columns = ['comm_id','sent_id','sent']+patterns)
        sent_df_current.iloc[:,3:] = sent_df_current.iloc[:,3:].fillna(0).astype('int64')
        total_sent += sent_df_current.shape[0]
        if not sent_df_current.empty:
            ### add to complete dataframe
            sent_df = sent_df_current.append(sent_df).reset_index(drop=True)                
            print('%5dK. Done: %4d comments, %4d sent / %6d total sent. Flags: %s', 
                      (chunk_id+1)*100, 
                      len(set(sent_df_current.comm_id)), 
                      sent_df_current.shape[0], total_sent, 
                      sent_df_current.flag.agg(sum))
        else:
            print('%5dK. No new sentences.', (chunk_id+1)*100)
            break            
    else:
        print('No new comments in comm file: %s',comm_file)
        break
    if not sent_df.empty:
        sent_df.to_csv(rootfold+'/output/replace'+match_file, index = False)
        print('Saved match file: %s',match_file)
#print('Finished comm file: %s',comm_file)