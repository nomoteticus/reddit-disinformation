from psaw import PushshiftAPI
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import timeout_decorator
import time
import os
import re
import numpy as np
import logging

api = PushshiftAPI()
rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'
with open('cwd.txt','r') as file:
    rootfold = file.read().rstrip()+'/'

#with open('cwd.txt','r') as file: rootfold = file.read().rstrip()+'/'

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s %(name)10s [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"logs/subm_comm_realtime.log"),
        logging.StreamHandler()
    ])
SC = logging.getLogger('SC')
SC.setLevel(logging.DEBUG)
SC_subm = logging.getLogger('SC.subm')
SC_comm = logging.getLogger('SC.comm')

def test_logger():
    SC.info('Testing SC - INFO')
    SC_subm.info('Testing SC submission -INFO')
    SC_subm.info('Testing SC comment -INFO')
    SC.debug('Testing SC -DEBUG')
    SC_subm.debug('Testing SC subm -DEBUG')
    SC_comm.debug('Testing SC comm -DEBUG')

def conv_utc(utc, diff_hrs = 0):
    return datetime.fromtimestamp(int(utc))-timedelta(hours = diff_hrs)
def search_yes(regex, text):
    return bool(re.search(regex, text))
def remove_spaces(txt):
    return(" ".join(txt.split()))

#@timeout_decorator.timeout(45)
def extract_submissions(subr, lim, bef, aft, srt = 'desc'):
    subm_lst = []
    time.sleep(3)
    try:
        api = PushshiftAPI()
        corona_generator =  \
             api.search_submissions(subreddit = subr, limit = lim, 
                                    before = bef, after = aft,
                                    sort = srt, lang = 'en',
                                    filter=['id','title', 'subreddit',
                                            'author', 'url','domain', 
                                            'is_self','is_video','is_crosspostable','post_hint',
                                            'num_comments','score','removed_by_category',
                                            'selftext','link_flair_text','full_link'])             
        subm_lst = list(corona_generator)
        #SC_subm.debug('Success!')
    except StopIteration:
        SC_subm.error(subr + ': StopIterrationError')
    except RuntimeError:
        SC_subm.error(subr + ': RuntimeError')
    except timeout_decorator.timeout_decorator.TimeoutError:
        SC_subm.error(subr + ': TimeoutError')
    except:
        SC_subm.error(subr + ': OtherError')
        pass
    return(subm_lst)

def chunk_generator(LST,size):
    return (LST[i:i+size] for i in range(len(LST))[::size])

def process_com_df(com_df, keep_columns_comm):
    if com_df.shape[0]>0:
        com_df = com_df[keep_columns_comm].reset_index().sort_values(['link_id','created']).groupby('link_id').head(1000)
        com_df['created'] = com_df['created'].apply(conv_utc)
        com_df['body'] = com_df['body'].apply(remove_spaces)
        com_df['id'] = 't1_'+com_df['id']
    return(com_df)

def get_submission_ids(submission_df, exclude = set()):
    subm_ids = []
    for i,s in submission_df.iterrows():    
        if not s.is_self and \
        not bool(re.search('image|video|self', str(s.post_hint))) and \
        not bool(re.search('reddit|redd.it|imgur|gfycat|google|twitter|youtu|^self\\.|^i\\.redd', str(s.domain))) and \
        s.num_comments>0:    
            subm_ids.append(s.id)
    subm_ids = pd.Series(subm_ids)
    return(subm_ids[~subm_ids.isin(exclude)])

def add_to_old_df(new_df, old_df, sort_var, index_var = 'id'):
    return(new_df.set_index('id').\
                combine_first(old_df.set_index('id')).\
                    reset_index().sort_values(sort_var, ascending=False))
            
@timeout_decorator.timeout(60,timeout_exception=StopIteration)
def scrape_chunk(id_chunk, keep_columns_comm, subm_limit):
    time.sleep(3)
    com_df = pd.DataFrame({}, columns = keep_columns_comm)
    comments_all = []
    try:
        api = PushshiftAPI()
        comments_all = list(api.search_comments(link_id = ','.join(id_chunk), 
                                                limit=subm_limit, 
                                                filter = keep_columns_comm))
        if len(comments_all)>0:
            comments_level1 = [c.d_ for c in comments_all if c.parent_id[:3]=='t3_']
            comments_level2 = [c.d_ for c in comments_all if c.parent_id in ['t1_'+r['id'] for r in comments_level1] ]
            com_df = pd.DataFrame(comments_level1 + comments_level2)
    except StopIteration:
        SC_comm.error('StopIterration error')
    except RuntimeError:
        SC_comm.error('Runtime error')
    except timeout_decorator.timeout_decorator.TimeoutError:
        SC_subm.error('TimeoutError')
    except:
        SC_comm.error('Other error')
        pass
    return(com_df)


@timeout_decorator.timeout(5,timeout_exception=StopIteration) 
def ff_05seconds(x):
    f05_logger = logging.getLogger("CS_f05")
    f05_logger.setLevel(logging.DEBUG)
    try:
        time.sleep(x)
        f05_logger.info('YEP!   | Number: %2d', x)
    except StopIteration:
        f05_logger.error('NOPE! | Number: %2d', x)

@timeout_decorator.timeout(10,timeout_exception=StopIteration) 
def ff_10seconds(x):
    f10_logger = logging.getLogger("CS_f10")
    f10_logger.setLevel(logging.DEBUG)
    try:
        time.sleep(x)
        f10_logger.info('YEP!   | Number: %2d', x)
    except StopIteration:
        f10_logger.info('NOPE!  | Number: %2d', x)
