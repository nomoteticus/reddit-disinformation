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

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'
with open('cwd.txt','r') as file:
    rootfold = file.read().rstrip()+'/'

sys.path.append(rootfold+"1_scraping")
import func_submcomm as sc

MAIN = logging.getLogger('MAIN')
MAIN.setLevel(logging.DEBUG)
MAIN_subm = logging.getLogger('MAIN.subm')
MAIN_comm = logging.getLogger('MAIN.comm')


###
MAIN.info('Started syntax to extract new submissions and comments')
### 

"""
SUBREDDITS
"""

subreddits_df = pd.read_csv(rootfold+'input/subr_classification.csv')
subreddits = subreddits_df.subreddit[subreddits_df.keep]
###
MAIN.info('Relevant subreddits: %3d\n', len(subreddits))


"""
SUBMISSIONS
"""


MAIN.info('Started scraping new Submissions' )

subm_files = sorted([filename for filename in os.listdir(rootfold+'output') if re.search('SUBM_2020_[0-9][0-9].csv',filename)], reverse = True)
MAIN.info('Files: %s', subm_files)

current_month = datetime.now().month
current_month_file_subm = 'SUBM_2020_%02d.csv' % current_month

## Create or import dataframe and go back 3 days since last submission

if current_month_file_subm in subm_files:
    subm_current_df = pd.read_csv(rootfold+'output/'+current_month_file_subm, lineterminator='\n')
    subm_current_df.created_utc = pd.to_datetime(subm_current_df.created_utc)
    ### calculate difference from last scraping day
    max_day = pd.to_datetime(subm_current_df.created_utc).max()
    diff_days = (datetime.now() - max_day).days
    # NOT DONE remove last 2 days
    # subm_current_df[subm_current_df.created_utc <= (max_day - timedelta(days=2))]
    existing_submissions = set(subm_current_df.id)
    MAIN.info('Last submission file: %s', current_month_file_subm)

else:
    previous_month_file_subm = 'SUBM_2020_%02d.csv' % (datetime.now().month-1)
    subm_previous_df = pd.read_csv(rootfold+'output/'+previous_month_file_subm, lineterminator='\n')
    subm_current_df = pd.DataFrame({},columns = subm_previous_df.columns)
    diff_days = 0
    existing_submissions = []
    MAIN.info('Added new submission file: ', current_month_file_subm)
    del(subm_previous_df)
MAIN.info('Days since last submission scraped: %d', diff_days)


### Scrape and add new submissions to dataframe

subm_new_lst = []
nrep = 1
while len(subreddits)>0 and nrep<6:
    MAIN_subm.info('Repetition %1d ', nrep)
    for subr in subreddits:
        subm_lst = []
        subm_lst = sc.extract_submissions(subr = subr, srt="asc", lim = 1000000,
                                          bef = "1d", aft = str(diff_days+3)+"d")
        subm_new_lst+=subm_lst
        MAIN_subm.debug('Finished: %30s. Cases: %5d .Overall: %6d', subr, len(subm_lst), len(subm_new_lst))
        subreddits  = set(subreddits).difference(set([s.subreddit for s in subm_new_lst]))
    nrep+=1

MAIN_subm.info('FINISHED scraping submissions. Repetitions: %d. New submissions: %d', 
               nrep, len(subm_new_lst))
subreddits  = set(subreddits).difference(set([s.subreddit for s in subm_new_lst]))
if len(subreddits)>0:
    MAIN_subm.warning('Subreddits not scraped: %d : %s', len(subreddits),subreddits)


### Process and write to file

if len(subm_new_lst)>0:
    subm_new_df = pd.DataFrame([s.d_ for s in subm_new_lst])
    subm_new_df['id'] = 't3_' + subm_new_df['id']
    subm_new_df['created_utc'] = subm_new_df['created_utc'].apply(sc.conv_utc)
    subm_new_df['title'] = subm_new_df['title'].astype('str').apply(sc.remove_spaces)
    subm_new_df = subm_new_df[subm_current_df.columns]
    #subm_new_df = subm_new_df[~subm_new_df.id.isin(subm_current_df.id)]
    ###
    MAIN_subm.info('PROCESSED submissions. Date min: %20s ; Date max %20s', \
          min(subm_new_df.created_utc), max(subm_new_df.created_utc))
    ###
    subm_new_df = subm_new_df[subm_new_df.created_utc.dt.month==current_month]
    MAIN_subm.info('Checked month. Remaining submissions: %d', subm_new_df.shape[0])
    ###
    if subm_new_df.shape[0]>0:
        subm_unite_df = subm_new_df.set_index('id').\
            combine_first(subm_current_df.set_index('id')).\
                reset_index().\
                    sort_values('created_utc', ascending=False)
        #####
        subm_unite_df.to_csv(rootfold+'output/'+current_month_file_subm, index = False)
        MAIN_subm.info('WROTE submissions to file: %s', current_month_file_subm)
        del(subm_unite_df)
    else:
        MAIN_subm.warning('No new submissions to add')
else:
    MAIN_subm.warning('No new submissions to add')

del(subm_unite_df,subm_current_df)

MAIN_comm.info('FINISHED SUBMISSIONS.\n')


"""
COMMENTS
"""

MAIN_comm.info('Starting scraping COMMENTS')

if subm_new_df.shape[0]>0:
    
    keep_columns_comm = ['id', 'created',
                        'subreddit', 'link_id', 'parent_id',
                        'body', 'edited',
                        'author', 
                        'collapsed', 'no_follow', 'score', 'distinguished']
    
    comm_files = sorted([filename for filename in os.listdir(rootfold+'output') if re.search('^COMM_2020_[0-9][0-9].csv',filename)], reverse = True)
    current_month_file_comm = current_month_file_subm.replace('SUBM','COMM')    
    
    ### Read last comments file
    
    if current_month_file_comm in comm_files:
        comm_current_df = pd.read_csv(rootfold+'output/'+current_month_file_comm, lineterminator='\n')
        comm_current_df.created = pd.to_datetime(comm_current_df.created)
        #comm_current_df = comm_current_df[comm_current_df.created<max(comm_current_df.created)-timedelta(days=1)]
    else:
        comm_current_df = pd.DataFrame(columns = comm_previous_df.columns)
    
    subm_toscrape = sc.get_submission_ids(subm_new_df, exclude = set())
    #subm_toscrape = subm_new_ids#[s for s in subm_new_ids if s not in comm_current_df.link_id]
    MAIN_comm.info('Submissions to scrape for comments: %d\n', len(subm_toscrape))
    
    comm_new_df = pd.DataFrame({}, columns = keep_columns_comm)
    
    comm_finished = 0
    subm_limit = 1000000
    chunk_sizes = [200,100,100,50,10]
    nrep_comm = 1
    
    while len(subm_toscrape)>0 and nrep_comm<6:
        MAIN_subm.info('Repetition %1d ', nrep_comm)
        current_subreddit = 'NoneYet'
        subm_finished = 0
        for id_chunk in sc.chunk_generator(subm_toscrape,chunk_sizes[nrep_comm-1]):
             comm_new_df = comm_new_df.\
                 append(sc.scrape_chunk(id_chunk, keep_columns_comm, subm_limit))
             comm_finished=comm_new_df.shape[0]
             subm_finished+=chunk_sizes[nrep_comm-1]
             if comm_new_df.shape[0] > 0:
                 current_subreddit = comm_new_df.iloc[-1].subreddit
             MAIN_comm.debug('%7d / %7d submissions | %10d comments | r/%s',
                            subm_finished, len(subm_toscrape), comm_finished, current_subreddit)
             time.sleep(2)
        subm_toscrape = subm_toscrape[~subm_toscrape.isin(set(comm_new_df.link_id))]
        nrep_comm+=1
    
    MAIN_comm.info('FINISHED scraping comments. Repetitions: %d. New comments: %d', 
                   nrep_comm-1, comm_new_df.shape[0])
    subm_toscrape = subm_toscrape[~subm_toscrape.isin(set(comm_new_df.link_id))]
    
    if len(subm_toscrape)>0:
        MAIN_subm.warning('Submissions not scraped: %d', len(subm_toscrape))

    if comm_new_df.shape[0]>0:
        comm_new_df = sc.process_com_df(comm_new_df, keep_columns_comm)
        #comm_new_df = comm_new_df[~comm_new_df.id.isin(set(comm_current_df.id))]
        MAIN_comm.info('PROCESSED comments. Date min: %20s ; Date max %20s', \
                       min(comm_new_df.created), max(comm_new_df.created))
        comm_unite_df = comm_new_df.set_index('id').\
            combine_first(comm_current_df.set_index('id')).\
                reset_index().\
                    sort_values('created', ascending=False)
        #####
        comm_unite_df.to_csv(rootfold+'output/'+current_month_file_comm, index = False)
        MAIN_comm.info('WROTE comments to file: %s', current_month_file_comm)

    else:
        MAIN_comm.info('No new comments to add')
else:
    MAIN_comm.info('No new comments to add')

MAIN_comm.info('FINISHED.\n\n\n')
del(comm_unite_df, comm_current_df)