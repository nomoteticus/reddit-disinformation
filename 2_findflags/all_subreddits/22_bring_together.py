# -*- coding: utf-8 -*-

import re
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

##
module_url = rootfold+"/3_embed/model/universal-sentence-encoder_4"
ebmeddingmodel = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return ebmeddingmodel(input)
##


subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
comm_files = sorted(re.findall('COMM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
match_files = sorted(re.findall('MATCH_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))

### set regexes
covid_regex =  ['china_flu'] + [re.sub("[\\*\\(\\)]","",'\\b' +w.rstrip()) \
                for w in open(rootfold+'/input/keywords_covid.txt','r')] 
                
covid_regex = re.compile('|'.join(covid_regex))
fake_regex = re.compile('\\b(dis|mis|mal)info|\\bpropagand|\\bconspira|\\bfalseh|\\bfake (news|info)')
#re.findall(fake_regex, 'this flake news is fake disinfrormation')

### Set defaults for filtering / grouping 

col_subm_keep = ['subreddit','id','title','month','week','day',
                 'full_link','url','domain','removed_by_category',
                 'subm_covid', 'num_comments']
col_comm_keep = ['subreddit','parent_id','link_id','id','score','body']

groupby_subr = ['subreddit','subm_covid','month','week','day']

flag_types = ['disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']

### Functions tp process subm/subr/ 

def process_SUBM(SS, col_subm_keep):
    SS['alltexts'] = SS[['title','subreddit','selftext']].fillna('').assign(comb = lambda df: df.title + df.subreddit + df.selftext).comb.astype('str')
    SS['subm_covid'] = SS['alltexts'].str.lower().str.contains(covid_regex)
    SS['subm_fake'] = SS['alltexts'].str.lower().str.contains(fake_regex)
    SS['removed_by_category'] = SS['removed_by_category'].fillna('visible').astype('str')
    SS['created_utc'] = pd.to_datetime(SS['created_utc'])
    SS['month'] = SS.created_utc.dt.month
    SS['week'] = SS.created_utc.dt.week
    SS['day'] = pd.to_datetime(SS.created_utc).dt.date
    SS = SS[~SS.subm_fake][col_subm_keep].\
         rename(columns = {'id':'subm_id','title':'subm_title',
                           'full_link':'subm_link','url':'link_url','domain':'link_domain',
                           'removed_by_category':'subm_removed', 'num_comments':'subm_ncomments'})
    return SS

def agg_SUBM(SS):
    return SS.reset_index().\
            groupby(groupby_subr).\
                agg(n_subm_all = ('subm_id','count'),
                    n_subm_kept = ('subm_removed', lambda x: (x=='visible').sum()))
        

    
def process_COMM(CC, col_comm_keep):
    CC = CC[col_comm_keep].\
         rename(columns = {'link_id':'subm_id','id':'comm_id',
                           'parent_id':'comm_parent_id','score':'comm_score',
                           'body':'comm_body'})
    return CC

def embed_SUBM(SUBM):
    embeddings=np.empty((0, 512))
    for strings in np.array_split(SUBM.subm_title, 1+SUBM.shape[0]/10000 ):
        embeddings = np.vstack((embeddings, embed(strings).numpy() ))
        print('#',end='')        
    pcamodel = PCA(n_components=2).fit(embeddings.T)
    category_vectors = pcamodel.components_.T
    embeddings_pca_df = pd.DataFrame(category_vectors, columns = ["PC"+str(100+x)[1:] for x in range(1,3)])
    SUBM = pd.concat([SUBM, embeddings_pca_df], axis = 1)
    return SUBM

SR_ALL = pd.read_csv(rootfold+'/input/subr_classification.csv')
SR = SR_ALL.query('keep')[['subreddit','category']].\
        rename(columns = {'category':'subreddit_cat'})


UNITED = pd.DataFrame([])
AGG_SS = pd.DataFrame([])
for subm_file, comm_file, match_file in zip(subm_files,comm_files,match_files):
    SS = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n')
    SS = process_SUBM(SS, col_subm_keep)
    AGG_SS = AGG_SS.append(agg_SUBM(SS))
    print('Processed: ' + subm_file)
    ##
    CC = pd.read_csv(rootfold+'/output/'+comm_file, lineterminator='\n')
    CC = process_COMM(CC, col_comm_keep)
    print('Processed: ' + comm_file)
    ##
    MM = pd.read_csv(rootfold+'/output/'+match_file, lineterminator='\n')
    MM['other'] = ((MM.flag > 0) & (MM[flag_types].sum(axis=1) == 0)).astype(int)
    ### Join
    UNTD = MM.query('flag>0').set_index('comm_id').\
            join(CC.set_index('comm_id'), 
                 on = 'comm_id', how = 'left').\
                reset_index().set_index(['subreddit','subm_id','comm_id']).\
            join(SS.set_index(['subreddit','subm_id']),                 
                 on = ['subreddit','subm_id'], how = 'left').\
            join(SR.set_index('subreddit'),
                 on = 'subreddit', how = 'left')
    ### Select final columns
    UNTD = UNTD[['month','week','day','subreddit_cat', 'sent_id', 'sent', 
                 'flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda','other', 
                 'comm_parent_id', 'subm_title', 'subm_removed', 'subm_covid', 'subm_link', 'link_url',
                 'subm_ncomments']]
    UNITED = UNITED.append(UNTD)
    print('Finished: %s . Shape: %s' % (match_file, UNITED.shape))

#SS.title[:10].str.lower().str.contains(covid_regex)
#SS['subm_covid'].str.lower().str.contains(covid_regex)

UNITED = UNITED.reset_index()
UNITED = UNITED[~UNITED.day.isna()]
UNITED = UNITED[UNITED.comm_parent_id.str.contains('^t3_')]
UNITED = UNITED.sort_values('day', ascending = False)

flag_vars = ['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda','other']

groupby_subr
groupby_subm = ['subreddit_cat','subreddit','subm_covid','month','week','day','subm_id','subm_title']
groupby_comm = groupby_subm + ['comm_id']

### COMMENTS
COMM_day = UNITED.groupby(groupby_comm)[flag_vars].agg(np.sum).reset_index()
COMM_day.loc[:,flag_vars] = COMM_day.loc[:,flag_vars]>0

#### SUBMISSIONS
SUBM_day = COMM_day.groupby(groupby_subm)[flag_vars].agg(np.sum).reset_index()
SUBM_day = embed_SUBM(SUBM_day).sort_values('day', ascending = False)


### SUBREDDIT
SUBR_day = COMM_day.groupby(['subreddit_cat']+groupby_subr)[flag_vars].\
                    agg(np.sum).join(AGG_SS).reset_index()

SUBR_day.to_csv(rootfold+"/dashboard/data/app_subr_day.csv", index = False)
SUBM_day.to_csv(rootfold+"/dashboard/data/app_subm_day.csv", index = False)
UNITED.to_csv(rootfold+"/dashboard/data/app_flags_large.csv", index = False)

#AGG_SS['perc_del'] = AGG_SS.n_subm_kept / AGG_SS.n_subm_all
#pd.pivot_table(AGG_SS.reset_index(), 
#               columns = 'subm_covid', index = 'subreddit', values = 'perc_del')


UNITED.groupby(UNITED.subm_created.dt.month)\
    [['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']].agg(sum).T
UNITED.groupby(UNITED.subreddit_cat)\
    [['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']].agg(sum).T



"""
SUBR: subr, subr_cat
SUBM: covid_related : Y/N, date ; RM: fake_news_related,
COMM: rating: positive; RM: body: bot, sarcasm ; RM: link_id: t1
SENT: flag + 6 flags, method: regex, pos ; flagging: multiple
AUTHOR: ...

1. Do we know comment is flag ? pos regex (+)
2. Do we trust the flag ? author, multiple flagged
3. Is it really disinformation ? (cluster, fact checking validation)


Top flaggers
PCA: flags
Monthly comparison + weekly slider
"""
