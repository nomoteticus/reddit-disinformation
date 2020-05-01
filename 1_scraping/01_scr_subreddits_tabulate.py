"""
Created on Mon Apr 13 18:37:15 2020

@author: j0hndoe
"""
#rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'
with open('cwd.txt','r') as file:
    rootfold = file.read().rstrip()+'/'

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
from psaw import PushshiftAPI

api = PushshiftAPI()

sdf_past = pd.read_csv(rootfold+"output/R_subm_subr_covid_past.csv", lineterminator = '\n')
sdf_present = pd.read_csv(rootfold+"output/R_subm_subr_covid.csv", lineterminator = '\n')
print(sdf_present.shape)

sdf_all = pd.concat([sdf_past,
                    sdf_present[~sdf_present['id'].isin(set(sdf_past['id']))]]).\
          sort_values(['created_utc'], ascending = False)

### separate per day
sdf_all['month_day'] = ['%02d-%02d' % (d.month, d.day) for d in pd.to_datetime(sdf_all.created_utc)]

### tabulate
subr_per_day = sdf_all[['id','subreddit','month_day']].groupby(['subreddit','month_day']).count().query('id>1')
subr_per_day = subr_per_day.unstack(level =1).fillna(0)
subr_per_day.columns = subr_per_day.columns.get_level_values(1)
subr_per_day['total'] = subr_per_day.sum(axis = 1)/(subr_per_day.shape[1])
subr_per_day['total7d'] = subr_per_day.iloc[:,-8:-1].sum(axis = 1)/(subr_per_day.iloc[:,-8:-1].shape[1])


### define condition to keep subreddits in list
condition_keep = np.logical_or(subr_per_day['total']>10, subr_per_day['total7d']>10)
subr_per_day_top = subr_per_day[condition_keep].sort_values('total7d', ascending = False)

### extract metadata for each subreddit
subr_meta = []
for subr in subr_per_day_top.index:
    l = []
    for s in api.search_submissions(subreddit = subr,
                       limit = 1000,
                       after = '30d',
                       sort = 'desc',
                       filter=['subreddit','subreddit_subscribers','num_comments','removed_by_category']):
        l.append(s.d_)
    df = pd.DataFrame(l)
    if 'removed_by_category' in df.columns:
        subr_meta.append((df.subreddit[0], 
                          df.subreddit_subscribers[0], 
                          df.num_comments[df.removed_by_category.isna()].mean(),
                          (~df.removed_by_category.isna()).mean()))
    else:
        subr_meta.append((df.subreddit[0], 
                  df.subreddit_subscribers[0], 
                  df.num_comments.mean(),
                  0))
    print('Finished: ' + subr)

### Edit metadata
subr_meta_df = pd.DataFrame(subr_meta, columns = ['subreddit','subscribers','ncomments','removed_by_category']).set_index('subreddit')
subr_meta_df['subscribers_K'] = [int(v) for v in (subr_meta_df['subscribers'] / 1000)]
subr_meta_df['avg_ncomments'] = round(subr_meta_df['ncomments'], 1)
subr_meta_df['perc_removed'] = round(100*subr_meta_df['removed_by_category'], 1)
subr_meta_df = subr_meta_df[['subscribers_K','avg_ncomments','perc_removed']]

### Import categories
try:
    subr_classification = pd.read_csv(rootfold+'input/subr_classification.csv').set_index('subreddit').iloc[:,:3]
except:
    subr_classification = pd.DataFrame(columns = ['subreddit','keep','category','language']).set_index('subreddit')

### Join categories and metadata
subr_per_day_top_meta = subr_classification.\
    join(subr_meta_df, how = 'right').\
        join(subr_per_day_top.iloc[:,-5:], how = 'left').round(1)

### Save updated list of subreddits
subr_per_day_top_meta.to_csv(rootfold+'output/subr_per_day_top_meta.csv')
subr_per_day_top.to_csv(rootfold+'output/subr_per_day.csv')

### Select uncategorized indexes
categ_new_entries = set(subr_per_day_top_meta[subr_per_day_top_meta.category.isna()].index).difference(subr_classification.index)

### Add uncategorized subreddits to list of categories
if len(categ_new_entries)>0:
    subr_per_day_top_meta.loc[categ_new_entries].to_csv(rootfold+'input/subr_classification.csv', mode ='a', header = False)
