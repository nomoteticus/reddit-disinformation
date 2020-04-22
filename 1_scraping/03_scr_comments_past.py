from psaw import PushshiftAPI
from datetime import datetime, timedelta
import os
import re
import pandas as pd
from collections import Counter

api = PushshiftAPI()

rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'

def conv_utc(utc, diff_hrs = 0):
    return datetime.fromtimestamp(int(utc))-timedelta(hours = diff_hrs)
def search_yes(regex, text):
    return bool(re.search(regex, text))
def chunk_generator(LST,size):
    return (LST[i:i+size] for i in range(len(LST))[::size])

subm_files = sorted([filename for filename in os.listdir(rootfold+'output') if search_yes('SUBM_2020_',filename)], reverse = True)
subm_files

comm_files = sorted([filename for filename in os.listdir(rootfold+'output') if search_yes('^COMM_2020_',filename)], reverse = True)
comm_files


keep_columns = ['id', 'created_utc',
                'subreddit', 'link_id', 'parent_id',
                'body', 'edited',
                'author', 'author_fullname', 'is_submitter',
                'collapsed', 'no_follow',
                'score', 'distinguished']
    
for subm_file in subm_files:
    print(subm_file+'\n---------------\n')
    print(str(datetime.now())[:19] + '\n')
    submission_df = pd.read_csv(rootfold+'output/'+subm_file).sort_values(['subreddit','created_utc'])
    subm_ids = []
    for i,s in submission_df.iterrows():    
        if not s.is_self and \
        not bool(re.search('image|video|self', str(s.post_hint))) and \
        not bool(re.search('reddit|redd.it|imgur|gfycat|google|twitter|youtu|^self\\.|^i\\.redd', str(s.domain))) and \
        s.num_comments>0:    
            subm_ids.append(s.id)
    pd.DataFrame(columns = keep_columns).to_csv(rootfold+'output/'+subm_file.replace('SUBM','COMM'), index = False)
    subm_finished = 0
    comm_finished = 0
    subm_total = len(subm_ids)
    for id_chunk in chunk_generator(subm_ids,100):
        com_gen = (api.search_comments(link_id = ','.join(id_chunk), limit = 1000000, filter = keep_columns))
        comments_level1 = [c.d_ for c in com_gen if c.link_id[:3]=='t3_']
        com_gen = (api.search_comments(link_id = ','.join(id_chunk), limit = 1000000, filter = keep_columns))
        comments_level2 = [c.d_ for c in com_gen if c.parent_id in ['t1_'+r['id'] for r in comments_level1] ]
        d0 = datetime.now()
        com_df = pd.DataFrame(comments_level1 + comments_level2)[keep_columns]
        com_df['created_utc'] = com_df['created_utc'].apply(conv_utc)
        com_df['body'] = com_df['body'].str.replace('\n',' ').str.replace('  ',' ')
        com_df['id'] = 't1_'+com_df['id']
        com_df.to_csv(rootfold+'output/'+subm_file.replace('SUBM','COMM'), mode = 'a', index = False, header = False)
        subm_finished+=100
        comm_finished+=com_df.shape[0]
        print('Finished %7d / %7d submissions. Comments: %7d (%s)' % (subm_finished, subm_total, comm_finished, str(datetime.now())[:19]))



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