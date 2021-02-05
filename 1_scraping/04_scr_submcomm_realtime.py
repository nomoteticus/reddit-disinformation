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

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)
#rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'
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
#subreddits_df = pd.read_csv(rootfold+'input/subr_classification_miss.csv')
subreddits = subreddits_df[~subreddits_df.keep.isna()]
subreddits = subreddits[subreddits.keep].subreddit
###
MAIN.info('Relevant subreddits: %3d\n', len(subreddits))


"""
SUBMISSIONS
"""


MAIN.info('Started scraping new Submissions' )

subm_files = sorted([filename for filename in os.listdir(rootfold+'output') if re.search('SUBM_20[0-9][0-9]_[0-9][0-9].csv',filename)], reverse = True)
MAIN.info('Files: %s', subm_files)

current_year  = datetime.now().year
current_month = datetime.now().month
previous_month_year  = current_year if current_month>1 else current_year-1
previous_month_month = current_month-1 if current_month>1 else 12


current_month_file_subm = 'SUBM_%d_%02d.csv' % (current_year, current_month)
current_month_file_comm = current_month_file_subm.replace('SUBM','COMM')

previous_month_file_subm = 'SUBM_%d_%02d.csv' % (previous_month_year, previous_month_month)
previous_month_file_comm = previous_month_file_subm.replace('SUBM','COMM')


## Create or import dataframe and go back 3 days since last submission

if current_month_file_subm in subm_files:
    subm_current_df = pd.read_csv(rootfold+'output/'+current_month_file_subm, lineterminator='\n')
    subm_current_df.created_utc = pd.to_datetime(subm_current_df.created_utc)
    ### calculate difference from last scraping day
    max_day = pd.to_datetime(subm_current_df.created_utc).max()
    MAIN.info('Last submission file: %s', current_month_file_subm)
else:
    subm_previous_df = pd.read_csv(rootfold+'output/'+previous_month_file_subm, lineterminator='\n')
    subm_current_df = pd.DataFrame({},columns = subm_previous_df.columns)
    ### calculate difference from last scraping day
    max_day = pd.to_datetime(subm_previous_df.created_utc).max()
    MAIN.info('Last submission file: %s', previous_month_file_subm)
    MAIN.info('Added new submission file: %s', current_month_file_subm)

diff_days = (datetime.now() - max_day).days
MAIN.info('Days since last submission scraped: %d', diff_days)


### Scrape and add new submissions to dataframe

subm_new_lst = []
nrep = 1
while len(subreddits)>0 and nrep<6:
    @timeout_decorator.timeout(min(nrep,4)*60) 
    def extract_subm_timeout(*args, **kwargs):
       return sc.extract_submissions(*args, **kwargs)
    MAIN_subm.debug('Repetition %1d ', nrep)
    for subr in subreddits:
        subm_lst = []
        subm_lst = extract_subm_timeout(subr = subr, srt="asc", lim = 1000000,
                                        bef = "10m", aft = str(diff_days+5)+"d")
        subm_new_lst+=subm_lst
        MAIN_subm.debug('Finished: %30s. Cases: %5d .Overall: %6d', subr, len(subm_lst), len(subm_new_lst))
        subreddits  = set(subreddits).difference(set([s.subreddit for s in subm_new_lst]))
    nrep+=1

MAIN_subm.info('FINISHED scraping submissions. Repetitions: %d. New submissions: %d', 
               nrep-1, len(subm_new_lst))
subreddits = set(subreddits).difference(set([s.subreddit for s in subm_new_lst]))
if len(subreddits)>0:
    MAIN_subm.warning('Subreddits not scraped: %d : %s', len(subreddits),subreddits)


### Process and write to file

if len(subm_new_lst)>0:
    subm_allnew_df = pd.DataFrame([s.d_ for s in subm_new_lst])
    subm_allnew_df['id'] = 't3_' + subm_allnew_df['id']
    subm_allnew_df['created_utc'] = subm_allnew_df['created_utc'].apply(sc.conv_utc)
    subm_allnew_df['title'] = subm_allnew_df['title'].astype('str').apply(sc.remove_spaces)
    subm_allnew_df = subm_allnew_df[subm_current_df.columns]
    ###
    MAIN_subm.info('PROCESSED submissions. Date min: %20s ; Date max %20s', \
          min(subm_allnew_df.created_utc), max(subm_allnew_df.created_utc))
    ###
    subm_pre_df = subm_allnew_df[(subm_allnew_df.created_utc.dt.year==previous_month_year) & (subm_allnew_df.created_utc.dt.month==previous_month_month)]
    subm_new_df = subm_allnew_df[(subm_allnew_df.created_utc.dt.year==current_year) & (subm_allnew_df.created_utc.dt.month==current_month)]
    ###
    ### Add submissions to current month dataframe
    if subm_new_df.shape[0]>0:
        subm_unite_df = sc.add_to_old_df(subm_new_df, subm_current_df, sort_var = 'created_utc')
        subm_unite_df.to_csv(rootfold+'output/'+current_month_file_subm, index = False)
        MAIN_subm.info('WROTE %d submissions to file: %s', subm_new_df.shape[0], current_month_file_subm)
        del(subm_unite_df, subm_current_df)
    else:
        MAIN_subm.warning('No new submissions to add')
    ### Add comments to previous month dataframe
    if subm_pre_df.shape[0]>0:
        subm_previous_df = pd.read_csv(rootfold+'output/'+previous_month_file_subm, lineterminator='\n')
        subm_previous_df.created_utc = pd.to_datetime(subm_previous_df.created_utc)
        ###
        subm_unite_df = sc.add_to_old_df(subm_pre_df, subm_previous_df, sort_var = 'created_utc')
        subm_unite_df.to_csv(rootfold+'output/'+previous_month_file_subm, index = False)
        MAIN_subm.info('WROTE %d submissions to file: %s', subm_pre_df.shape[0], previous_month_file_subm)
        del(subm_unite_df, subm_previous_df)
    else:
        MAIN_subm.warning('No new submissions to add for previous month')
else:
    MAIN_subm.warning('No new submissions to add')

MAIN_comm.info('FINISHED SUBMISSIONS.\n')



"""
COMMENTS
"""

MAIN_comm.info('Starting scraping COMMENTS')

if subm_allnew_df.shape[0]>0:
    
    keep_columns_comm = ['id', 'created',
                        'subreddit', 'link_id', 'parent_id',
                        'body', 'edited',
                        'author', 
                        'collapsed', 'no_follow', 'score', 'distinguished']
    
    comm_files = sorted([filename for filename in os.listdir(rootfold+'output') if re.search('^COMM_20[0-9][0-9]_[0-9][0-9].csv',filename)], reverse = True)
    current_month_file_comm = current_month_file_subm.replace('SUBM','COMM')    
    
    ### Read last comments file
    
    if current_month_file_comm in comm_files:
        comm_current_df = pd.read_csv(rootfold+'output/'+current_month_file_comm, lineterminator='\n')
        comm_current_df.created = pd.to_datetime(comm_current_df.created)
    else:
        comm_previous_df = pd.read_csv(rootfold+'output/'+previous_month_file_comm, lineterminator='\n')
        comm_current_df = pd.DataFrame(columns = comm_previous_df.columns)
    
    subm_toscrape = sc.get_submission_ids(subm_allnew_df, exclude = set())

    MAIN_comm.info('Submissions to scrape for comments: %d\n', len(subm_toscrape))
    
    comm_new_df = pd.DataFrame({}, columns = keep_columns_comm)
    
    comm_finished = 0
    subm_limit = 1000000
    chunk_sizes = [200,100,100,50,10,5]
    nrep_comm = 1
    
    while len(subm_toscrape)>0 and nrep_comm<7:
        MAIN_comm.debug('Repetition %1d ', nrep_comm)
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
    
    comm_new_df = sc.process_com_df(comm_new_df, keep_columns_comm)
    MAIN_comm.info('PROCESSED comments. Date min: %20s ; Date max %20s', \
               min(comm_new_df.created), max(comm_new_df.created))

    comm_pre_df = comm_new_df[comm_new_df['link_id'].isin(set(subm_pre_df['id']))]
    comm_new_df = comm_new_df[comm_new_df['link_id'].isin(set(subm_new_df['id']))]

    ### Add comments to current month dataframe
    if comm_new_df.shape[0]>0:
        comm_unite_df = sc.add_to_old_df(comm_new_df, comm_current_df, sort_var = 'created')
        comm_unite_df.to_csv(rootfold+'output/'+current_month_file_comm, index = False)
        MAIN_comm.info('WROTE %d comments to file: %s', comm_new_df.shape[0], current_month_file_comm)
        del(comm_unite_df, comm_current_df)
    else:
        MAIN_comm.info('No new comments to add')
    ### Add comments to previous month dataframe
    if comm_pre_df.shape[0]>0:
        comm_previous_df = pd.read_csv(rootfold+'output/'+previous_month_file_comm, lineterminator='\n')
        comm_previous_df.created = pd.to_datetime(comm_previous_df.created)
        ###
        comm_unite_df = sc.add_to_old_df(comm_pre_df, comm_previous_df, sort_var = 'created')
        comm_unite_df.to_csv(rootfold+'output/'+previous_month_file_comm, index = False)
        MAIN_comm.info('WROTE %d comments to file: %s', comm_pre_df.shape[0], previous_month_file_comm)
        del(comm_unite_df, comm_previous_df)
    else:
        MAIN_comm.info('No new comments to add in previous month')

else:
    MAIN_comm.info('No new comments to add')

MAIN_comm.info('FINISHED.\n\n\n')
