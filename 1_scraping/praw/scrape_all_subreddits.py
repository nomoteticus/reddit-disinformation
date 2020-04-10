import sys
import praw
import time
import os
import pandas as pd
import numpy as np

rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'

APP_url = 'https://ssl.reddit.com/prefs/apps/'
reddit = praw.Reddit(client_id = 'nIjGq9V9Wn_ARw', 
                     client_secret = 'iFJcztvXJX1rwX3VAcNa2QzWmTE', 
                     user_agent = 'extracting news for academic research'
                    )

### create new file
file = open(rootfold+"covid_submissions.txt","w+")
pd.DataFrame({'id':[],'subreddit':[],'title':[],'date_created':[], 'iter':[]}).to_csv(rootfold+'covid_allsubr_subm.csv', index = False)
file.close()

### define search terms
search_terms = ['coronavirus', 'covid','covid19', 'covid-19','corona','virus', 
                'pandemic','epidemic', 'crisis', 
                'lockdown', 'quarantine', 'isolation', 'social distancing',
                'deaths', 'death toll', 'infected', 'infection', 'the spread', 'patients', 
                'hospital', 'vaccine', 'symptoms','ventilators', 'masks', 'medical supplies',
                'wuhan', 'fauci', 'cdc', 'world health organization',
                'chloroquine', 'hydroxychloroquine']
search_limits = [300]*6 + [150]*(len(search_terms)-6)
sleep_time = 60*15


### loop over search terms every 30 minutes and save to file

i=0
while True:
    with open(rootfold+'covid_submissions.txt','r') as file:
        existing_submissions = [line.strip() for line in file]
    print('ITER: %d\n----------\nExisting submisions: %d' % (i,len(existing_submissions)))
    new_submissions = []
    for term, limit in zip(search_terms, search_limits):
        new_submissions+=[(s.id, s.subreddit.display_name, s.title, s.created_utc, i)
                          for s in reddit.subreddit("all").\
                          search(term, limit=limit, syntax='lucene', sort = 'new') 
                          if s.id not in set(existing_submissions)]
        existing_submissions += [s[0] for s in new_submissions]
        print('%s. Total length: %d' % (term, len(new_submissions)))
    with open(rootfold+'covid_submissions.txt','a') as file:
        for s in new_submissions:
            file.write(s[0] + '\n')
    pd.DataFrame(new_submissions).to_csv(rootfold+'covid_allsubr_subm.csv', index = False, header = False, mode = 'a')
    print('Done. New submissions: %d \n\n' % len(new_submissions))
    time.sleep(sleep_time)
    i+=1