"""
Created on Tue Apr 14 19:42:06 2020

@author: j0hndoe
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from psaw import PushshiftAPI

api = PushshiftAPI()

rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'
#with open('cwd.txt','r') as file:    rootfold = file.read().rstrip()+'/'


subr_classification = pd.read_csv(rootfold+'input/subr_classification.csv').set_index('subreddit')
subr_classification = subr_classification.reset_index().sort_values('subreddit').set_index('subreddit')


subr_classification['category'] = subr_classification['category'].fillna('')
subr_classification['language'] = subr_classification['language'].fillna('')
subr_classification['keep'] = subr_classification['keep'].fillna(-1)


subr_classification['new'] = 0
subr_classification.new[subr_classification['category']==''] = 1


subr_classification.category[[bool(re.search('politic', i.lower())) for i in subr_classification.index]] = "generic_politics"
subr_classification.category[[bool(re.search('news', i.lower())) for i in subr_classification.index]] = "generic_news"
subr_classification.category[[bool(re.search('corona|covid', i.lower())) for i in subr_classification.index]] = "coronavirus"

### Add language
subr_classification['language'][subr_classification['language']==''] = "en"


### Manually classify

### Save
subr_classification.to_csv(rootfold+'input/subr_classification.csv')

subr_classification = pd.read_csv(rootfold+'input/subr_classification.csv').set_index('subreddit')
subr_classification = subr_classification.reset_index().sort_values('subreddit').set_index('subreddit')



### Add language
for i in subr_classification.index:
    if str(subr_classification.loc[i,'category']) == 'local' and str(subr_classification.loc[i,'language']) == 'en':
        for s in api.search_submissions(subreddit = i, limit = 10, after = '30d', sort = 'desc', filter=['id','title']):
            print(s.title)
        print(i)
        print('\nLanguages: en, fr, it, es, pt, cn, oth\n')
        language = input(i+" .Language?\n")
        if language !='':
            subr_classification.loc[i,'language'] = language
        print('\n\n\n')


### Save
subr_classification.to_csv(rootfold+'input/subr_classification.csv')

Counter(subr_classification['category'])

subr_classification['keep'] = \
    (subr_classification['language'] == "en") & \
        (subr_classification['avg_ncomments']<1) & \
            (subr_classification['category'].isin(["generic_news","generic_politics","coronavirus","local","science_health"]))


### Save
subr_classification.to_csv(rootfold+'input/subr_classification.csv')

        
        
