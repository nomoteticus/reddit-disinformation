# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import logging

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

sys.path.append(rootfold+"/3_embed")

import tensorflow as tf
import tensorflow_hub as hub

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
module_url = rootfold+"/3_embed/model/universal-sentence-encoder_4"
ebmeddingmodel = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return ebmeddingmodel(input)
#embed(['This article is really fake news','You are spreading disinformation.','This is fkae news'])


### Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"/logs/pos_embed.log"),
        logging.StreamHandler()
    ])
LOGE = logging.getLogger('LOGE')
LOGE.setLevel(logging.DEBUG)

#####
LOGE.info('-----------------')
LOGE.info('Started embedding')

selected_subreddits = \
    ["Coronavirus","COVID19","China_Flu", "CoronavirusUS","CoronavirusUK",
     "Conservative", "democrats", "POTUSWatch", "Republican","Libertarian","conspiracy",
     "news", "worldnews", "politics", "POLITIC",
     "europe", "canada","australia","ukpolitics","unitedkingdom","Sino","China",
     "science", "technology"]

#"dataisbeautiful","healthcare","healthIT"
#"CoronavirusRecession", "worldpolitics",
#"europe",    "health-IT", "UpliftingNews", "usanews", "nottheonion",

subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))

SS_SUM = pd.DataFrame()
### Filter subreddits
n_files_loaded=0
for subm_file in subm_files[-2:]: 
    SS = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n', date_parser = 'created_utc')
    SS = SS[SS.subreddit.isin(set(selected_subreddits))][['subreddit','id','title','created_utc','removed_by_category']]
    SS['month'] = int(subm_file[11])
    SS['removed_by_category'] = SS['removed_by_category'].fillna('visible')
    SS = SS.sort_values(['subreddit','month','removed_by_category'])
    SS_SUM = SS_SUM.append(SS)
    #print('\nFinished: ' + subm_file)
    n_files_loaded+=1
SS_SUM = SS_SUM.set_index('subreddit')

needed_subreddits = [s for s in SS_SUM.index.unique() if s in selected_subreddits]
LOGE.info('Loaded %d submission files', n_files_loaded)

### Add kD PCA dim
Ndim = 5

embed_file_name = "PCA_SUBM_"+str(Ndim)+"D.csv"

if SS_SUM.shape[0]>0:
    embed_file = sorted(re.findall(embed_file_name, ' '.join(os.listdir(rootfold+'/output'))))
    ##
    if len(embed_file)>0:
        SS_ALL = pd.read_csv(rootfold+'/output/'+embed_file[0], lineterminator='\n')
        existing_embeddings = SS_ALL.id
        SS_SUM = SS_SUM[~SS_SUM.id.isin(set(existing_embeddings))]
        LOGE.info('Found embedding file: %s', embed_file)
        needed_subreddits = [s for s in SS_SUM.index.unique() if s in selected_subreddits]
        LOGE.info('Needed subreddits: %d', len(needed_subreddits))
    else:
        SS_ALL = pd.DataFrame()
        LOGE.info('Created embedding file.')
    ###    
    for subreddit in needed_subreddits:
        SUBR = SS_SUM.loc[subreddit]
        LOGE.debug('Started subreddit: %20s (size:%8d)', subreddit, SUBR.shape[0])    
        if SUBR.shape[0]>0:
            embeddings=np.empty((0, 512))
            for strings in np.array_split(SUBR.title, 1+SUBR.shape[0]/10000 ):
                embeddings = np.vstack((embeddings, embed(strings).numpy() ))        
            pcamodel = PCA(n_components=Ndim).fit(embeddings.T)
            category_vectors = pcamodel.components_.T
            embeddings_pca_df = pd.DataFrame(category_vectors, columns = ["PC"+str(100+x)[1:] for x in range(1,Ndim+1)])
            SUBR = pd.concat([SUBR.reset_index(),embeddings_pca_df], axis = 1)
            SS_ALL = SS_ALL.append(SUBR)

    LOGE.info('Embedded %d files.', SS_SUM.shape[0])
    
    SS_ALL = SS_ALL.sort_values('created_utc', ascending=False)
    
    SS_ALL.to_csv(rootfold + "/output/" +  embed_file_name, index = False)
    
    LOGE.info('Saved file: %s', embed_file_name)
else:
    LOGE.info('Nothing to embed')

#scp /home/j0hndoe/Documents/git/reddit-disinformation/output/PCA_SUBM_2D.csv ubuntu@134.155.109.205:/home/ubuntu/reddit-disinformation/output/Pavel