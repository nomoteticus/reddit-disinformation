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

"""
SUBMISSIONS
"""

#subreddits = ['worldnews','news']
subreddits_df = pd.read_csv(rootfold+'input/subr_classification.csv')
subreddits = subreddits_df.subreddit[subreddits_df.keep]

def extract_submissions(subreddits, lim, bef, aft, srt = 'desc'):
    subm_lst = []
    for subr in subreddits:
        err = 0
        corona_generator =  \
             api.search_submissions(subreddit = subr,
                                    limit = lim,
                                    before = bef,
                                    after = aft,
                                    sort = srt,
                                    lang = 'en',
                                    filter=['id','title',
                                            'subreddit',
                                            'author',    
                                            'url','domain', 
                                            'is_self','is_video','is_crosspostable','post_hint',
                                            'num_comments','score','removed_by_category',
                                            'selftext','link_flair_text','full_link'])
        for s in corona_generator:
            try:
                subm_lst.append(s.d_)
            except:
                err+=1
        print('Finished: %s - %d errors , %d overall cases' % (subr, err, len(subm_lst)))
    return(pd.DataFrame(subm_lst))

#subm_df = pd.DataFrame(subm_lst)
#subm_df.shape

subm_files = sorted([filename for filename in os.listdir(rootfold+'output') if search_yes('SUBM_2020_',filename)])
subm_months_existing = [int(filename[10:12]) for filename in subm_files][-1:]
subm_months_needed = list( set(range(1,datetime.now().month+1,1)).difference(set(subm_months_existing)) )
subm_months_needed 

for month in subm_months_needed:
    min_date = (datetime.now()-datetime(2020, month,  1,0,0,0,0)).days+1
    max_date = (datetime.now()-datetime(2020, month+1,1,0,0,0,0)).days
    min_date = min_date if min_date>0 else 0
    max_date = max_date if max_date>0 else 0
    print('Month %3d ; min %5dd , max %5dd' % (month, min_date, max_date))
    df_current_month = extract_submissions(subreddits,
                                            lim = 1000000,
                                            bef = str(max_date)+"d", 
                                            aft = str(min_date)+"d",
                                            srt = "asc")
    df_current_month['id'] = 't3_' + df_current_month['id']
    df_current_month['created_utc'] = df_current_month['created_utc'].apply(conv_utc)
    df_current_month = df_current_month.query('created_utc>="2020-%02d-01" & created_utc<"2020-%02d-01"' % (month, month+1))
    df_current_month.to_csv(rootfold + "output/SUBM_2020_%02d.csv" % month , index = False)
    print(conv_utc(min(df_current_month.created)), conv_utc(max(df_current_month.created)))
    print('Timestamp: ' + str(datetime.now())[:19] + '\n\n')


"""
UPLOAD:
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/SUBM_2020_01.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/SUBM_2020_02.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/SUBM_2020_03.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/
scp /home/j0hndoe/Documents/git/reddit-disinformation/output/SUBM_2020_04.csv ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/

DOWNLOAD:
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/SUBM_2020_01.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/SUBM_2020_02.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/SUBM_2020_03.csv /home/j0hndoe/folder
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/SUBM_2020_04.csv /home/j0hndoe/folder

PAVEL download:
scp ubuntu@134.155.111.117:/home/ubuntu/reddit-disinformation/output/SUBM_2020_01.csv /home/YOURUSERNAME/folder
"""

#existing_submissions = []
#for file in subm_files[-2:]:
#    print('a')
