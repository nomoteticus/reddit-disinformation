from psaw import PushshiftAPI
from datetime import datetime, timedelta
import os
import re
import pandas as pd
from collections import Counter

#rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation/'
#os.chdir(rootfold)
with open('cwd.txt','r') as file: rootfold = file.read().rstrip()+'/'

def read_linebyline(file):
    return([line.rstrip() for line in open(file)])
def conv_utc(utc, diff_hrs = 0):
    return datetime.fromtimestamp(int(utc))-timedelta(hours = diff_hrs)
def conv_utcvec(utcvec):
    return [conv_utc(utc) for utc in utcvec]

api = PushshiftAPI()


#####

covid_search_terms = read_linebyline(rootfold+'input/keywords_covid.txt')
covid_search_string = '|'.join(covid_search_terms)
covid_search_string

ls = []
err = []
iter=0
corona_generator =  \
    api.search_submissions(q = covid_search_string,
                           limit = 10000000,
                           after = '60d',
                           sort = 'desc',
                           filter=['id','title','subreddit','subreddit_subscribers',
                                   'author','full_link','url','domain', 'is_self'])
for s in corona_generator:
    iter+=1
    try:
        if s.subreddit_subscribers>1000 and \
           not s.is_self and \
           not bool(re.search('reddit|redd.it|imgur|gfycat', s.domain)):
            ls.append(s)
    except:
        err.append(s.id)
    if iter % 1000 == 1:
        print('%7d / %7d' % (len(ls),iter))

print('Finished scraping. Preparing dataset...')

ls_df = pd.DataFrame( (s.d_ for s in ls) )

if ls_df.shape[0] > 0 :
    ls_df['date_added'] = datetime.now()
    ls_df['created_utc'] = conv_utcvec(ls_df['created_utc'])
    ls_df = ls_df[ls_df['created_utc'] >='2020-03-01']

ls_df.to_csv(rootfold+"output/R_subm_subr_covid_past.csv", 
             index = False)

print('%s / Err: %7d/ Iter: %7d/ Subm: %7d \n' % 
      (str(datetime.now())[:19], len(err), iter, ls_df.shape[0]))