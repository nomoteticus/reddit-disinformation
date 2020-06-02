# -*- coding: utf-8 -*-

import re
import os
import pandas as pd
import numpy as np

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
comm_files = sorted(re.findall('COMM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))


### Resort all
for subm_file, comm_file in zip(subm_files[:3],comm_files[:3]): 
    SS = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n', date_parser = 'created_utc')
    print(SS[['id','created_utc']])
    print(SS.columns)
    SS.created_utc = pd.to_datetime(SS.created_utc)
    SS = SS[SS.created_utc.dt.month==int(subm_file[11])].sort_values('created_utc', ascending = False)
    SS.to_csv(rootfold+'/output/'+subm_file, index = False)
    print('Reordered: ' + subm_file)
    print(SS[['id','created_utc']])
    print(SS.columns)
    CC = pd.read_csv(rootfold+'/output/'+comm_file, lineterminator='\n', date_parser = 'created')
    print(CC[['id','created']])
    print(CC.columns)
    CC.created = pd.to_datetime(CC.created)
    reorder = CC[['id','link_id','created']].set_index('link_id').\
                join(SS[['id','created_utc']].rename(columns={'id':'link_id'}).\
                     set_index('link_id')).\
                        sort_values(['created_utc','created'], ascending = [False,False])
    reorder = reorder[~reorder.created_utc.isna()].id
    CC = CC.set_index('id').loc[reorder].reset_index()
    CC.to_csv(rootfold+'/output/'+comm_file, index = False)   
    print('Reordered: ' + comm_file)
    print(CC[['id','created']])
    print(CC.columns)