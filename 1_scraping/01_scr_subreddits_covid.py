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

ls_df0 = pd.DataFrame()
existing_submissions = set([])

try:
    ls_df0 = pd.read_csv(rootfold+"output/R_subm_subr_covid.csv")
    existing_submissions = set(ls_df0.iloc[:,4])
    subr_ctr = Counter(ls_df0['subreddit'])
    subr_df = pd.DataFrame.from_dict(subr_ctr, orient='index').sort_values(by = 0, ascending=False)
    subr_df.to_csv(rootfold+"output/R_top_subr_covid.csv")  
except:
    print('No file yet')
    


covid_search_terms = read_linebyline(rootfold+'input/keywords_covid.txt')
covid_search_string = '|'.join(covid_search_terms)
covid_search_string

ls = []
err = []
iter=0
corona_generator =  \
    api.search_submissions(q = covid_search_string,
                           limit = 10000,
                           after = '30m',
                           sort = 'desc',
                           filter=['id','title','subreddit','subreddit_subscribers',
                                   'author','full_link','url','domain', 'is_self'])
for s in corona_generator:
    iter+=1
    try:
        if s.id not in existing_submissions and \
           s.subreddit_subscribers>1000 and \
           not s.is_self and \
           not bool(re.search('reddit|redd.it|imgur|gfycat', s.domain)):
            ls.append(s)
            existing_submissions.append(s.id)
    except:
        err.append(s.id)

ls30m = len(ls)

if iter<10:
    corona_generator =  \
        api.search_submissions(q = covid_search_string,
                               limit = 100000,
                               after = '1d',
                               sort = 'desc',
                               filter=['id','title','subreddit','subreddit_subscribers',
                                   'author','full_link','url','domain', 'is_self'])
    for s in corona_generator:
        iter+=1
        try:
            if s.id not in existing_submissions and \
               s.subreddit_subscribers>1000 and \
               not s.is_self and \
               not bool(re.search('reddit|redd.it|imgur|gfycat', s.domain)):
                   ls.append(s)
        except:
            err.append(s.id)

ls1d = len(ls)

sls = [s.d_ for s in ls]

with open(rootfold+'output/LOG_subm_subr_covid.txt', 'a') as logfile:
    logfile.write('%s / Err: %7d/ Iter: %7d/ Subm: %7d (30m) ; %7d (1d) ; %9d (total) \n' % 
                  (str(datetime.now())[:19], len(err), iter, ls30m, ls1d, ls_df0.shape[0] + ls30m + ls1d))

ls_df = pd.DataFrame(sls)
if ls_df.shape[0] > 0 :
    ls_df['date_added'] = datetime.now()
    ls_df['created_utc'] = conv_utcvec(ls_df['created_utc'])

if ls_df0.shape[0] == 0 :
    ls_df.to_csv(rootfold+"output/R_subm_subr_covid.csv", 
                 index = False, header=True)
else:
    ls_df.to_csv(rootfold+"output/R_subm_subr_covid.csv", 
                 index = False, header = False, mode = 'a')
