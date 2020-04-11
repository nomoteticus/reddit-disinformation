from psaw import PushshiftAPI
from datetime import datetime, timedelta
import os
import re
import pandas as pd
from collections import Counter

#rootfold = '/home/j0hndoe/Documents/git/reddit-disinformation'
#os.chdir(rootfold)
rootfold = '../'

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
except:
    print('No file yet')
    


covid_search_terms = read_linebyline(rootfold+'input/keywords_covid.txt')
covid_search_string = '|'.join(covid_search_terms)
covid_search_string


ls = []
iter=0
corona_generator =  \
    api.search_submissions(q = covid_search_string,
                           limit = 100,
                           after = '2h',
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
            if len(ls) % 1000 == 1:
                print('%10.0d / %10.0d' % (len(ls),iter))
    except:
        print('Error: '+ str(s))
print(len(ls))


sls = [s.d_ for s in ls if s.id not in existing_submissions]

Counter([s.subreddit for s in ls]).most_common()

ls_df = pd.DataFrame(sls)
ls_df['date_added'] = datetime.now()
ls_df['created_utc'] = conv_utcvec(ls_df['created_utc'])

if ls_df0.shape[0] == 0 :
    ls_df.to_csv(rootfold+"output/R_subm_subr_covid.csv", 
                 index = False, header=True)
else:
    ls_df.to_csv(rootfold+"output/R_subm_subr_covid.csv", 
                 index = False, header = False, mode = 'a')  
    
