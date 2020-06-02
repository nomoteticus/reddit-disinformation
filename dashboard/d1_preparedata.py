#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:25:28 2020

@author: j0hndoe
"""

rootfold = %pwd
import pandas as pd
df_matches = pd.read_csv(rootfold+'/output/comm_matches.csv')
matchedids = set(df_matches.comm_id)
HASREGEXFULL = pd.read_csv(rootfold+'/output/HASREGEX.csv')

COMM_direct = HASREGEXFULL.query('comm_direct')\
    [['subreddit','sr_category','subm_title','subm_created','subm_url','subm_score','comm_id','comm_body']].\
        assign(FNflag = lambda df: df.comm_id.isin(matchedids).astype(int)).\
            query('FNflag==1').\
                sort_values('subm_created', ascending = False)

COMM_direct_1000 = COMM_direct[:1000]

COMM_direct_1000.to_csv(rootfold+"/dashboard/df_test_pos.csv", index=False)

?pd.read_csv
