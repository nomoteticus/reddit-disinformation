# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.decomposition import PCA
import logging

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

sys.path.append(rootfold+"/2_findflags/all_subreddits")
import functions_pos_match as fpm

### Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s <UTD> [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"/logs/pos_embed_unite.log"),
        logging.StreamHandler()
    ])
LOGU = logging.getLogger('LOGU')
LOGU.setLevel(logging.DEBUG)

LOGU.info('--------------------')
LOGU.info('Started unification.')
LOGU.info('--------------------')

## Load language model
module_url = rootfold+"/3_embed/model/universal-sentence-encoder_4"
ebmeddingmodel = hub.load(module_url)
LOGU.debug("module %s loaded", module_url[-43:])
def embed(input):
  return ebmeddingmodel(input)
##


subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
comm_files = sorted(re.findall('COMM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
match_files = sorted(re.findall('MATCH_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))

#re.findall(fake_regex, 'this flake news is fake disinfrormation')

### Set defaults for filtering / grouping 

col_subm_keep = ['subreddit','id','title','month','week','day',
                 'full_link','url','domain','removed_by_category',
                 'subm_covid', 'subm_fake', 'author', 'num_comments']
col_comm_keep = ['subreddit','parent_id','link_id','id','score','body','author']

#groupby_subr = ['subreddit','subm_covid','month','week','day']
groupby_subr = ['subreddit_cat','subreddit','subm_covid','month','week','day']
#groupby_subr_week = ['subreddit','subm_covid','week']
groupby_subr_week = ['subreddit_cat','subreddit','subm_covid','week']
##
groupby_subm = groupby_subr + ['subm_id','subm_title']#,'subm_link', 'link_url','subm_removed']
groupby_comm = groupby_subm + ['comm_id','comm_body']
groupby_sent = ['subreddit', 'subm_id', 'comm_id','sent_id']

groupby_subm_week = groupby_subr_week + ['subm_id','subm_title']
groupby_auth_week = groupby_subr_week + ['subm_author']
groupby_dom_week = groupby_subr_week + ['link_domain']


flag_types = ['disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']
flag_vars = ['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda','other']
count_vars = ['n_subm_all', 'n_subm_kept']

### Functions tp process subm/subr/ 

def process_SUBM(SS, SR, col_subm_keep):
    SS['alltexts'] = SS[['title','subreddit','selftext']].fillna('').assign(comb = lambda df: df.title + df.subreddit + df.selftext).comb.astype('str')
    SS['subm_covid'] = SS['alltexts'].str.lower().str.contains(fpm.covid_regex)
    SS['subm_fake'] = SS['alltexts'].str.lower().str.contains(fpm.fake_regex)
    SS['removed_by_category'] = SS['removed_by_category'].fillna('visible').astype('str')
    SS['created_utc'] = pd.to_datetime(SS['created_utc'])
    SS['month'] = SS.created_utc.dt.month
    SS['week'] = SS.created_utc.dt.week
    SS['day'] = pd.to_datetime(SS.created_utc).dt.date
    return SS[col_subm_keep].\
             rename(columns = {'id':'subm_id','title':'subm_title','author':'subm_author',
                               'full_link':'subm_link','url':'link_url','domain':'link_domain',
                               'removed_by_category':'subm_removed', 'num_comments':'subm_ncomments'}).\
             set_index('subreddit').join(SR.set_index('subreddit'), how = 'left').\
             reset_index()

def agg_SUBM(SS, grvars):
    SS.day = SS.day.astype('str')
    return SS.reset_index().\
            groupby(grvars).\
                agg(n_subm_all = ('subm_id','count'),
                    n_subm_kept = ('subm_removed', lambda x: (x=='visible').sum()))
     

    
def process_COMM(CC, col_comm_keep):
    CC = CC[col_comm_keep].\
         rename(columns = {'link_id':'subm_id','id':'comm_id',
                           'parent_id':'comm_parent_id','score':'comm_score',
                           'body':'comm_body', 'author':'comm_author'})
    return CC

def embed_SUBM(SUBM, Ncomp=5):
    embeddings=np.empty((0, 512))
    for strings in np.array_split(SUBM.subm_title, 1+SUBM.shape[0]/10000 ):
        embeddings = np.vstack((embeddings, embed(strings).numpy() ))
        print('#',end='')        
    pcamodel = PCA(n_components=Ncomp).fit(embeddings.T)
    category_vectors = pcamodel.components_.T
    embeddings_pca_df = pd.DataFrame(category_vectors, columns = ["PC"+str(100+x)[1:] for x in range(1,Ncomp+1)])
    SUBM = pd.concat([SUBM, embeddings_pca_df], axis = 1)
    return SUBM

def removemin(DF, col, min_val):
    DF = DF.reset_index()
    val_counts = DF[col].value_counts()
    return DF[DF[col].isin(val_counts[val_counts >= min_val].index)]


SR_ALL = pd.read_csv(rootfold+'/input/subr_classification.csv').dropna()
SR_ALL_sh = SR_ALL[['subreddit','category']].\
        rename(columns = {'category':'subreddit_cat'})
SR = SR_ALL.query('keep')[['subreddit','category']].\
        rename(columns = {'category':'subreddit_cat'})

### Check if United files exist
try:
    UNITED = pd.read_csv(rootfold+'/output/UNITED_FLAG.csv', lineterminator='\n').\
                set_index(groupby_sent)
    AGG_SS = pd.read_csv(rootfold+'/output/UNITED_SUBM.csv', lineterminator='\n').\
                set_index(groupby_subr_week)
    AGG_AA = pd.read_csv(rootfold+'/output/UNITED_AUTH.csv', lineterminator='\n').\
                set_index(groupby_auth_week)
    AGG_DD = pd.read_csv(rootfold+'/output/UNITED_DOM.csv', lineterminator='\n').\
                set_index(groupby_dom_week)
    SCM_generator = zip(subm_files[-2:], comm_files[-2:], match_files[-2:])
    ##
    ss1,ss2,aa1,aa2,dd1,dd2 = \
               AGG_SS.index.get_level_values(3).min(), AGG_SS.index.get_level_values(3).max(),\
               AGG_AA.index.get_level_values(3).min(), AGG_AA.index.get_level_values(3).max(),\
               AGG_DD.index.get_level_values(3).min(), AGG_DD.index.get_level_values(3).max()
    max_week = min([ss2,aa2,dd2])    
    LOGU.debug("Last week. SUBM: %d -> %d / AUTH: %d -> %d / DOM: %d -> %d",
               ss1,ss2,aa1,aa2,dd1,dd2)
except FileNotFoundError:
    UNITED = pd.DataFrame([])
    AGG_SS = pd.DataFrame([])
    AGG_AA = pd.DataFrame([])
    AGG_DD = pd.DataFrame([])
    SCM_generator = zip(subm_files, comm_files, match_files) #[:-1]
    LOGU.debug("Created: SUBM / AGG_SS, AUTH / AGG_AA, DOM / AGG_DD")
    max_week = 0


for subm_file, comm_file, match_file in SCM_generator:
    SS = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n', parse_dates=['created_utc'])
    SS = SS[SS.created_utc.dt.week > max_week-3]
    if not SS.empty:
        SS = process_SUBM(SS, SR_ALL_sh, col_subm_keep)
        ##
        CC = pd.read_csv(rootfold+'/output/'+comm_file, lineterminator='\n')
        CC = process_COMM(CC, col_comm_keep)
        LOGU.debug('Opened: %s & %s', subm_file, comm_file)
        LOGU.info('Updating weeks: %s',list(sorted(SS.week.value_counts().index)))
        ##
        MM = pd.read_csv(rootfold+'/output/'+match_file, lineterminator='\n')
        MM['other'] = ((MM.flag > 0) & (MM[flag_types].sum(axis=1) == 0)).astype(int)
        ### Join
        UNTD = MM.query('flag>0').set_index('comm_id').\
                join(CC.set_index('comm_id'), 
                     on = 'comm_id', how = 'left').\
                    reset_index().set_index(['subreddit','subm_id','comm_id','sent_id']).\
                join(SS.set_index(['subreddit','subm_id']),                 
                     on = ['subreddit','subm_id'], how = 'left')
        ### Select final columns
        UNTD = UNTD[['month','week','day','subreddit_cat', 'sent', 
                     'flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda','other', 
                     'subm_title', 'subm_removed', 'subm_covid', 'subm_fake',
                     'subm_link', 'link_url', 'link_domain', 'subm_ncomments', 'subm_author',
                     'comm_parent_id', 'comm_body', 'comm_author']]
        UNITED = UNTD.combine_first(UNITED) if UNITED.shape[0]>0 else UNTD
        ### Update aggregates
        AGG_SS = agg_SUBM(SS, groupby_subr_week).combine_first(AGG_SS) if AGG_SS.shape[0]>0 else agg_SUBM(SS, groupby_subr_week).copy()
        AGG_AA = agg_SUBM(SS, groupby_auth_week).combine_first(AGG_AA) if AGG_AA.shape[0]>0 else agg_SUBM(SS, groupby_auth_week).copy()
        AGG_DD = agg_SUBM(SS, groupby_dom_week).combine_first(AGG_DD) if AGG_DD.shape[0]>0 else agg_SUBM(SS, groupby_dom_week).copy()
        LOGU.info('Finished: %s . Current: %s. Total shape: %s' % (match_file, MM.shape, UNITED.shape))
    else:
        LOGU.info('Nothing to add: %s', subm_file)

del(SS,CC,MM)
#SS.title[:10].str.lower().str.contains(covid_regex)
#SS['subm_covid'].str.lower().str.contains(covid_regex)


## Final cleaning
UNITED = UNITED.reset_index()
UNITED['day'] = UNITED['day'].astype('str')
UNITED = UNITED.sort_values('day', ascending = False)
## Remove missing data, replies to comments (instead of submissions), 
## submissions on the topic of disinformation and comments made by bots
UNITED = UNITED[~UNITED.day.isna()]
UNITED = UNITED[UNITED.comm_parent_id.str.contains('^t3_')]
UNITED = UNITED[~UNITED.subm_fake.astype('bool')]
UNITED = UNITED[~(UNITED.comm_body.str.lower().str.contains(fpm.bot_body_regex) | \
                  UNITED.comm_author.str.lower().str.contains(fpm.bot_author_regex))]

## Saving to dataset
UNITED.to_csv(rootfold+"/output/UNITED_FLAG.csv", index=False)
##
AGG_SS.reset_index().sort_values('week', ascending = False).to_csv(rootfold+"/output/UNITED_SUBM.csv", index=False)
AGG_AA.reset_index().sort_values('week', ascending = False).to_csv(rootfold+"/output/UNITED_AUTH.csv", index=False)
AGG_DD.reset_index().sort_values('week', ascending = False).to_csv(rootfold+"/output/UNITED_DOM.csv", index=False)


#pd.crosstab(index = COMM_week.subreddit, columns = COMM_week.is_bot)
#pd.set_option('display.max_rows', 500) ; pd.reset_option('all')



### FLAGS
FLAGS_day = UNITED.reset_index().\
                loc[:,['subreddit_cat', 'subreddit' , 'subm_id', 'subm_covid',
                       'subm_title','subm_link', 'link_url','subm_removed',
                       'sent_id','sent',
                       'month', 'week', 'day'] + flag_vars].\
                sort_values('day', ascending=False)
LOGU.info('Created flags. Shape: %s', FLAGS_day.shape)

### COMMENTS
COMM_day = UNITED.groupby(groupby_comm)[flag_vars].agg(np.sum).reset_index()
COMM_day.loc[:,flag_vars] = COMM_day.loc[:,flag_vars]>0


#### SUBMISSIONS
SUBM_day = COMM_day.groupby(groupby_subm)[flag_vars].agg(np.sum).reset_index()
SUBM_day = embed_SUBM(SUBM_day, Ncomp=5).sort_values('day', ascending = False)
LOGU.info('Created submissions. Shape: %s', SUBM_day.shape)

### SUBREDDIT
SUBR_week = COMM_day.groupby(groupby_subr_week)[flag_vars].\
                    agg(np.sum).join(AGG_SS).reset_index().\
                        sort_values('week', ascending=False)
LOGU.info('Created subreddits. Shape: %s', SUBR_week.shape)


### AUTHORS
AUTH_FLAGGERS_week = removemin(UNITED.\
                               groupby(groupby_subr + ['comm_author'])[flag_vars].\
                               agg(np.sum).reset_index(), 
                               'comm_author', 2).\
                        sort_values('week', ascending=False)
                     

AUTH_FLAGGED_week = removemin(UNITED.\
                               groupby(groupby_subr + ['subm_author'])[flag_vars].\
                               agg(np.sum).reset_index(), 
                               'subm_author', 2).\
                        set_index(groupby_auth_week).\
                        join(AGG_AA, how = 'left').\
                        reset_index().\
                        sort_values('week', ascending=False)
                    
LOGU.info('Created authors. Flaggers shape: %s; Flagged shape: %s', 
          AUTH_FLAGGERS_week.shape, AUTH_FLAGGED_week.shape)


### DOMAINS
DOM_week = removemin(UNITED.groupby(groupby_dom_week)[flag_vars].\
                     agg(np.sum).reset_index(),
                     'link_domain', 2).\
                  set_index(groupby_dom_week).\
                  join(AGG_DD, how = 'left').\
                  reset_index().\
                  sort_values('week', ascending=False)

LOGU.info('Created domains. Shape: %s', DOM_week.shape)

#x = DOM_week.groupby('link_domain')['disinformation','n_subm_all'].agg('sum')
#x['proc'] = x['disinformation']/x['n_subm_all'].round(3)
#x[x.n_subm_all>=10].sort_values('proc', ascending = False).head(30)


### WRITE FILES

FLAGS_day.to_csv(rootfold+"/dashboard/data/app_flags_day.csv", index = False)
SUBM_day.to_csv(rootfold+"/dashboard/data/app_subm_day.csv", index = False)
SUBR_week.to_csv(rootfold+"/dashboard/data/app_subr_week.csv", index = False)
AUTH_FLAGGERS_week.to_csv(rootfold+"/dashboard/data/app_authFlaggers_week.csv", index = False)
AUTH_FLAGGED_week.to_csv(rootfold+"/dashboard/data/app_authFlagged_week.csv", index = False)
DOM_week.to_csv(rootfold+"/dashboard/data/app_domain_week.csv", index = False)

LOGU.info('Done writing 6 files.')
#AGG_SS['perc_del'] = AGG_SS.n_subm_kept / AGG_SS.n_subm_all
#pd.pivot_table(AGG_SS.reset_index(), 
#               columns = 'subm_covid', index = 'subreddit', values = 'perc_del')


#UNITED.groupby(UNITED.month)\
#   [['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']].agg(sum).T
#UNITED.groupby(UNITED.subreddit_cat)\
#    [['flag', 'disinformation', 'fakenews', 'bs', 'misleading', 'unreliable', 'propaganda']].agg(sum).T



"""
SUBR: subr, subr_cat
SUBM: covid_related : Y/N, date ; RM: fake_news_related,
COMM: rating: positive; RM: body: bot, sarcasm ; RM: link_id: t1
SENT: flag + 6 flags, method: regex, pos ; flagging: multiple
AUTHOR: ...

1. Do we know comment is flag ? pos regex (+)
2. Do we trust the flag ? author, multiple flagged
3. Is it really disinformation ? (cluster, fact checking validation)

https://mediabiasfactcheck.com/left/

Top flaggers
PCA: flags
Monthly comparison + weekly slider
"""
