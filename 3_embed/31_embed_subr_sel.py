# -*- coding: utf-8 -*-

import re
import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

sys.path.append(rootfold+"/3_embed")
import embed_logger as log

import tensorflow as tf
import tensorflow_hub as hub

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
module_url = rootfold+"/3_embed/model/universal-sentence-encoder_4"
ebmeddingmodel = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return ebmeddingmodel(input)
#embed(['This article is really fake news','You are spreading disinformation.','This is fkae news'])

selected_subreddits = \
    ["China_Flu", "Coronavirus","CoronavirusUS","COVID19","CoronavirusRecession",
     "Conservative", "democrats", "POTUSWatch", "Republican",
     "news", "worldnews", "usanews", "USNEWS", "Sino",
     "politics", "worldpolitics",
     "science", "technology", "dataisbeautiful","healthcare","healthIT"]

#"europe",    "health-IT", "UpliftingNews", "usanews", "nottheonion",

subm_files = sorted(re.findall('SUBM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))



SS_SUM = pd.DataFrame()
### Filter subreddits and Add 2D PCA dim
for subm_file in subm_files: 
    SS = pd.read_csv(rootfold+'/output/'+subm_file, lineterminator='\n', date_parser = 'created_utc')
    SS = SS[SS.subreddit.isin(set(selected_subreddits))][['subreddit','id','title','removed_by_category']]
    SS['month'] = int(subm_file[11])
    SS['removed_by_category'] = SS['removed_by_category'].fillna('visible')
    SS = SS.sort_values(['subreddit','month','removed_by_category'])
    SS_SUM = SS_SUM.append(SS)
    print('\nFinished: ' + subm_file)
SS_SUM = SS_SUM.set_index('subreddit')
needed_subreddits = [s for s in SS_SUM.index.unique() if s in selected_subreddits]

###
SS_ALL = pd.DataFrame()
for subreddit in needed_subreddits:
    SUBR = SS_SUM.loc[subreddit]
    print('Finished: '+ subreddit+ ' . Size ' + str(SUBR.shape))        
    embeddings=np.empty((0, 512))
    for strings in np.array_split(SUBR.title, 1+SUBR.shape[0]/10000 ):
        embeddings = np.vstack((embeddings, embed(strings).numpy() ))        
    pcamodel = PCA(n_components=2).fit(embeddings.T)
    category_vectors = pcamodel.components_.T
    embeddings_pca_df = pd.DataFrame(category_vectors, columns = ["PC"+str(100+x)[1:] for x in range(1,3)])
    SUBR = pd.concat([SUBR.reset_index(),embeddings_pca_df], axis = 1)
    SS_ALL = SS_ALL.append(SUBR)

print('Size: ' + str(SS_ALL.shape))

SS_ALL.to_csv(rootfold+"/output/PCA_SUBM_2D.csv")

#scp /home/j0hndoe/Documents/git/reddit-disinformation/output/PCA_SUBM_2D.csv ubuntu@134.155.109.205:/home/ubuntu/reddit-disinformation/output/Pavel