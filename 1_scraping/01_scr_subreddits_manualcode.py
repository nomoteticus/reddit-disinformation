"""
Created on Tue Apr 14 19:42:06 2020

@author: j0hndoe
"""

import pandas as pd
import numpy as np

rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'
#with open('cwd.txt','r') as file:    rootfold = file.read().rstrip()+'/'

pd.set_option('display.max_columns', 15)
pd.set_option('display.max_rows', 15)


subr_classification = pd.read_csv(rootfold+'input/subr_classification.csv').set_index('subreddit')
subr_classification = subr_classification.reset_index().sort_values('subreddit').set_index('subreddit')
subr_classification['category'] = subr_classification.keep.fillna('')
subr_classification['language'] = subr_classification.keep.fillna('')
subr_classification['keep'] = subr_classification.keep.fillna(-1)

new_indexes = []

for i in subr_classification.index[:3]:
    if str(subr_classification.loc[i,'category']) == 'nan':
        print(subr_classification.loc[:i])
        print('\nCategories: '+', '.join([str(el) for el in set(subr_classification.category)]))
        category = input(i+" .Category?\n")
        if category !='':
            subr_classification.loc[i,'category'] = category
        print('\n\n\n')
        new_indexes.append(i)
    if str(subr_classification.loc[i,'language']) == 'nan':
        print('\Languages: en, fr, it, es, pt, cn, oth')
        language = input(i+" .Language?\n")
        if language !='':
            subr_classification.loc[i,'language'] = language
        print('\n\n\n')
        new_indexes.append(i)
    if str(subr_classification.loc[i,'keep']) == 'nan':
        print('\nKeep: 1 = yes, 0 = no')
        keep = input(i+" .Keep scraping?\n")
        if keep !='':
            subr_classification.loc[i,'keep'] = int(keep)
        print('\n\n\n')
        new_indexes.append(i)

print(subr_classification.loc[list(set(new_indexes)),['keep','category','language']])

print('Finished')

old = 
[
'Coronavirus',
'news',
'worldnews',
'worldpolitics',
'worldevents',
'NewsPorn',
'politics',
'uspolitics',
'europe',
'PoliticalDiscussion',
'TrueReddit',
'Positive_News',
'offbeat',
'inthenews'
]

set(old).difference( set(subr_classification.index) )
