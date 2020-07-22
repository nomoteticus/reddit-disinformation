# -*- coding: utf-8 -*-

import sys
import re
import os
import pandas as pd
import numpy as np

rootfold = re.match('^.*reddit-disinformation', 
                    os.path.dirname(os.path.realpath(__file__))).group(0)
sys.path.append(rootfold+"/2_findflags/all_subreddits")
import functions_pos_match as fpm

subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
comm_files = sorted(re.findall('COMM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
match_files = sorted(re.findall('MATCH_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
match_files

#for comm_file in gen_comm_files:

#comm_file=comm_files[0]
#subm_file=subm_files[0]
#match_file=match_files[0]

flagwords = ['flag', 'disinformation', 'fakenews','bs', 'misleading', 'unreliable', 'propaganda']

COMMALL = pd.DataFrame()

for match_file, comm_file, subm_file in zip(match_files, comm_files, subm_files):
    ##
    MATCH = pd.read_csv(rootfold+'/output/'+ match_file, lineterminator='\n')
    ##
    COMM = pd.read_csv(rootfold+'/output/'+ comm_file, lineterminator='\n')
    COMM.body = COMM.body.astype('str')
    ## Remove bots
    COMM = COMM[~(COMM.body.str.contains(fpm.bot_body_regex) | \
                  COMM.author.str.lower().str.contains(fpm.bot_author_regex))]
    ##
    SUBM = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n', parse_dates=['created_utc'])
    SUBM['title'] = SUBM['title'].astype('str')
    SUBM['fake_regex'] = SUBM['title'].str.lower().str.contains(fpm.fake_regex)
    SUBMRED = SUBM[~SUBM.fake_regex].\
                rename(columns={'id':'subm_id','title':'subm_title'})\
                    [['subm_id', 'subm_title', 'domain']]
    ##
    PREL =  MATCH.groupby('comm_id')[flagwords].agg('sum').\
                  join(COMM.rename(columns = {'id':'comm_id', 'parent_id':'subm_id', 'body':'comm_body'})\
                                   [['comm_id','comm_body','subm_id']].\
                                       set_index(['subm_id','comm_id']), how = 'inner').\
                        query('~comm_body.str.contains(@fpm.sarcasm_and_irony_regex)').\
                        join(SUBMRED.set_index('subm_id'), how = 'inner').\
                        reset_index()
    print('Finished: ' + comm_file)
    ##
    COMMALL = COMMALL.append(PREL)

COMMBODY = COMMALL.groupby('comm_body')[flagwords].agg('sum').\
               join(COMMALL.groupby('comm_body')[['subm_title', 'domain']].agg('first')).\
            reset_index()


COMMBODY['nchar'] = [len(t) for t in COMMBODY.comm_body]
COMMBODY = COMMBODY[COMMBODY.nchar < COMMBODY.nchar.quantile(.95)]


COMMBODY[COMMBODY.fakenews>0].sort_values('fakenews', ascending = False).reset_index().loc[:10,['comm_body','fakenews']]         

TOCLASSIFY = pd.DataFrame()  
for word in flagwords[1:]:
    np.random.seed(0)  
    SLICE = COMMBODY[COMMBODY[word]>0]
    SLICE['word'] = word
    SLICE['prop'] = SLICE[word]/sum(SLICE[word])
    SLICE['class'] = ''
    index_word = np.random.choice(SLICE.index, size=100, replace=False, p=SLICE.prop)
    TOCLASSIFY = TOCLASSIFY.append(SLICE.loc[index_word])

COMMBODY.to_csv("4_check/data/COMMBODY.csv")
TOCLASSIFY[['word','comm_body','class']].to_csv("4_check/data/to_classify_regex.csv")

#x = pd.melt(           value_vars = flagwords, var_name="flagtype", value_name="nflags").\

### Other

np.random.seed(0)         
index_disinformation = np.random.choice(COMMBODY[COMMBODY['disinformation']>0].index, size=100, replace=False)
index_fakenews = np.random.choice(COMMBODY[COMMBODY['fakenews']>0].index, size=100, replace=False)
index_bs = np.random.choice(COMMBODY[COMMBODY['bs']>0].index, size=100, replace=False)
index_misleading = np.random.choice(COMMBODY[COMMBODY['misleading']>0].index, size=100, replace=False)
index_unreliable = np.random.choice(COMMBODY[COMMBODY['unreliable']>0].index, size=100, replace=False)
index_propaganda = np.random.choice(COMMBODY[COMMBODY['propaganda']>0].index, size=100, replace=False)

COMMBODY.loc[index_disinformation]
COMMBODY.loc[index_disinformation].flag.value_counts()
COMMBODY.loc[index_disinformation].query('flag>0')

COMMBODY.loc[index_fakenews]
COMMBODY.loc[index_fakenews].flag.value_counts()
COMMBODY.loc[index_fakenews].query('flag>0')

COMMBODY.loc[index_bs]
COMMBODY.loc[index_bs].flag.value_counts()
COMMBODY.loc[index_bs].query('flag>0')

COMMBODY.loc[index_bs]
COMMBODY.loc[index_bs].flag.value_counts()
COMMBODY.loc[index_bs].query('flag>0')

COMMBODY.loc[index_unreliable]
COMMBODY.loc[index_unreliable].flag.value_counts()
COMMBODY.loc[index_unreliable].query('flag>0')

COMMBODY.loc[index_propaganda]
COMMBODY.loc[index_propaganda].flag.value_counts()
COMMBODY.loc[index_propaganda].query('flag>0')

