import sys
import praw
from praw.models import MoreComments
import pandas as pd
import numpy as np
import re
from urlextract import URLExtract 
from datetime import datetime, timedelta
import time

rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'

def subr_todict(s):
    return {'id':s.id, 
            'subr_id': s.subreddit_id, 
            'author_id': s.author_fullname if s.author is not None else 'SUSPENDED', 
            'title':s.title, 
            'body':s.selftext,
            'score':s.score,
            'banned':s.banned_at_utc,
            'reports':len(s.user_reports),
            'num_comments':s.num_comments,
            'quarantine':s.quarantine,
            'url':s.url
            #'duplicates':s.duplicates,
           }
def comm_todict(c):
    return {'id':c.fullname, 'author_id': c.author_fullname if c.author is not None else 'SUSPENDED', 
            'submission_id':c.parent_id, 'body':c.body, 'score':c.score, 'date_created':c.created_utc}
def repl_todict(r, c):
    return {'id':r.fullname, 'author_id': r.author_fullname  if r.author is not None else 'SUSPENDED', 
            'submission_id':c.parent_id, 'parent_id':r.parent_id, 'text':r.body, 'score':r.score, 'date_created':r.created_utc}
def has_url(text):
    return URLExtract().has_urls(text)

APP_url = 'https://ssl.reddit.com/prefs/apps/'
reddit = praw.Reddit(client_id = 'nIjGq9V9Wn_ARw', 
                     client_secret = 'iFJcztvXJX1rwX3VAcNa2QzWmTE', 
                     user_agent = 'extracting news for academic research'
                    )

#pd.DataFrame(columns = ['id', 'subr_id', 'author_id', 'title', 'body', 'score', 'banned',
#                        'reports','num_comments','quarantine','url']).to_csv(rootfold+"R_all_submissions.csv", index = False)
#pd.DataFrame(columns = ['id', 'author_id','submission_id', 'body', 'score','date_created']).to_csv(rootfold+"R_all_comments.csv", index = False)
#pd.DataFrame(columns = ['id', 'author_id','submission_id','parent_id','text','score','date_created']).to_csv(rootfold+"R_all_replies.csv", index = False)

csv_file_submissions = 'http://ec2-3-8-1-23.eu-west-2.compute.amazonaws.com/vlad/all_submissions.csv'
itr = 0
step_print = 100
time_to_sleep = 60*60

while True:
    
    itr+=1
    already_there = list(pd.read_csv(rootfold+"R_all_submissions.csv")['id'])

    ALL = pd.read_csv(csv_file_submissions)
    latest_time = datetime.now()-timedelta(hours=2)
    ALL03 = ALL[(ALL.date_created>='2020-03-01') & 
                (ALL.date_created<str(latest_time))].sort_values(['subr_id','date_created'])
    
    print('Iteration %d. New submissions: %d. Total submissions: %d submissions.\n-----------------------------------' % 
          (itr, ALL03.shape[0]-len(already_there), ALL03.shape[0]))

    dict_subr = []
    dict_comm = []
    dict_repl = []

    submissions_finished = 0
    submissions_todo = ALL03.subm_id
    submissions_todo = submissions_todo[np.logical_not(submissions_todo.isin(already_there))]

    print('Started -> %7d cases ; %s.' % (len(dict_subr) , str(datetime.now())))
    for i,subm_id in enumerate(submissions_todo):
        s = reddit.submission(id = subm_id)
        if s.id not in set(already_there):
            try: 
                dict_subr.append(subr_todict(s))
                for c in s.comments:
                    if not isinstance(c, MoreComments):
                        dict_comm.append(comm_todict(c))
                        if has_url(c.body):
                            for r in c.replies:
                                if not isinstance(r, MoreComments):
                                    dict_repl.append(repl_todict(r,c))
            except: 
                print('Error: ' + str(s))
        if (i % step_print == 0 or i+1 == len(submissions_todo)) and len(dict_subr)>0:
            df_subr = pd.DataFrame(dict_subr)
            df_subr.to_csv(rootfold+"R_all_submissions.csv", index = False, header = False, mode = 'a')
            df_comm = pd.DataFrame(dict_comm)
            df_comm.to_csv(rootfold+"R_all_comments.csv", index = False, header = False, mode = 'a')
            if len(dict_repl)>0:
                df_repl = pd.DataFrame(dict_repl)
                df_repl.to_csv(rootfold+"R_all_replies.csv", index = False, header = False, mode = 'a')
            submissions_finished += len(dict_subr)
            print('Continue -> %7d cases ; %s.' % (submissions_finished , str(datetime.now())))
            already_there += list(df_subr.id)
            dict_subr = []
            dict_comm = []
            dict_repl = []
    print('Finished -> %7d cases ; %s.\n\n\n' % (submissions_finished , str(datetime.now())))
    time.sleep(time_to_sleep)
    print('... 1h pause')