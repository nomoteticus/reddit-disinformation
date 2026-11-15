from psaw import PushshiftAPI
from datetime import datetime, timedelta
import os
import re
import pandas as pd
from collections import Counter

api = PushshiftAPI()

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'

subm_files = sorted([filename for filename in os.listdir(rootfold+'output') if search_yes('SUBM_2020_[0-9][0-9].csv',filename)], reverse = True)
subm_files

comm_files = sorted([filename for filename in os.listdir(rootfold+'output') if search_yes('^COMM_2020_[0-9][0-9].csv',filename)], reverse = True)
comm_files

keep_columns = ['id', 'created',
                'subreddit', 'link_id', 'parent_id',
                'body', 'edited',
                'author', 
                'collapsed', 'no_follow', 'score', 'distinguished']

def conv_utc(utc, diff_hrs = 0):
    return datetime.fromtimestamp(int(utc))-timedelta(hours = diff_hrs)
def search_yes(regex, text):
    return bool(re.search(regex, text))
def chunk_generator(LST,size):
    return (LST[i:i+size] for i in range(len(LST))[::size])


def remove_spaces(txt):
    return(" ".join(txt.split()))

def process_com_df(com_df):
    com_df = com_df[keep_columns].sort_values(['link_id','created']).groupby('link_id').head(1000)
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
    return([id for id in subm_ids if id not in exclude])


for subm_file in subm_files:
    ###
    print('\n'+subm_file+'\n---------------')
    print(str(datetime.now())[:19] + '\n')
    ###
    submission_df = pd.read_csv(rootfold+'output/'+subm_file).sort_values(['subreddit','created_utc'])
    existing_comments=set()
    try:
        existing_comments = set(pd.read_csv(rootfold+'output/'+subm_file.replace('SUBM','COMM'), lineterminator='\n').link_id)
    except:
        pd.DataFrame(columns = keep_columns).to_csv(rootfold+'output/'+subm_file.replace('SUBM','COMM'), index = False)
    subm_ids = get_submission_ids(submission_df, existing_comments)
    ###
    subm_finished = 0
    comm_finished = 0
    subm_total = len(subm_ids)
    if subm_total>100:
        for id_chunk in chunk_generator(subm_ids,100):
            comments_level1 = [c.d_ for c in api.search_comments(link_id = ','.join(id_chunk), limit = 1000000, filter = keep_columns) if c.parent_id[:3]=='t3_']
            comments_level2 = [c.d_ for c in api.search_comments(link_id = ','.join(id_chunk), limit = 1000000, filter = keep_columns) if c.parent_id in ['t1_'+r['id'] for r in comments_level1] ]
            d0 = datetime.now()
            com_df = pd.DataFrame(comments_level1 + comments_level2)
            if com_df.shape[0]>0:
                com_df = process_com_df(com_df)
                com_df.to_csv(rootfold+'output/'+subm_file.replace('SUBM','COMM'), mode = 'a', index = False, header = False)
                current_subreddit = com_df.iloc[0].subreddit
            subm_finished+=100
            comm_finished+=com_df.shape[0]
            print('Finished %7d / %7d submissions | %10d comments | (%s) | r/%s' % (subm_finished, subm_total, comm_finished, str(datetime.now())[:19], current_subreddit))



"""
UPLOAD:
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/COMM_2020_01.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/COMM_2020_02.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/COMM_2020_03.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/COMM_2020_04.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/

DOWNLOAD:
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/COMM_2020_01.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/COMM_2020_02.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/COMM_2020_03.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/COMM_2020_04.csv /home/j0hndoe/folder

"""

#existing_comments = []