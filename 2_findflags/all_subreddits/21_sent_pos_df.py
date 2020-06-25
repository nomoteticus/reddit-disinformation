import sys
import re
import os
import pandas as pd
import logging

rootfold = re.match('^.*reddit-disinformation', 
                    os.path.dirname(os.path.realpath(__file__))).group(0)

### Setup logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s [%(levelname)8s ] %(message)s",
    datefmt='%y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(rootfold+"/logs/pos_onetime.log"),
        logging.StreamHandler()
    ])
LOG = logging.getLogger('LOG')
LOG.setLevel(logging.DEBUG)

pd.set_option('mode.chained_assignment', None)

LOG.info('Started Pattern search')
LOG.debug('PATH: %s',rootfold)

sys.path.append(rootfold+"/2_findflags/all_subreddits")
import functions_pos_match as fpm

patterns = [fpm.nlp.vocab.strings[match_id] for match_id in fpm.flagtypematcher._patterns.keys()]

comm_files = sorted(re.findall('COMM_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
match_files = sorted(re.findall('MATCH_2020_[0-9][0-9].csv', ' '.join(os.listdir(rootfold+'/output'))))
LOG.debug('COMM_files: %s', comm_files)
LOG.debug('MATCH_files: %s', match_files)

for comm_file in comm_files:#[-2:]:
    match_file = re.sub('COMM','MATCH',comm_file)
    LOG.info('Started comm file: %s',comm_file)
    if match_file not in match_files:
        total_sent = 0
        sent_df = pd.DataFrame([], columns = ['comm_id','sent_id','sent']+patterns)
        sent_df.to_csv(rootfold+'/output/'+match_file, index = False)
        existing_comments = []
    else:
        sent_df = pd.read_csv(rootfold+'/output/'+match_file)
        existing_comments = sent_df.comm_id
    ### Start scraping
    comm_generator = pd.read_csv(rootfold+'/output/'+comm_file, chunksize=100000)
    for chunk_id, COMM in enumerate(comm_generator):
        COMM = COMM[['subreddit','link_id','id','parent_id','body']]
        COMM.body = COMM.body.astype(str).str.lower()
        COMM = COMM[COMM.body.str.contains(fpm.regex_flag)]
        #remaining_comments = set(COMM.id).difference(existing_comments) #[c for c in COMM.id if c not in set(existing_comments)]
        remaining_comments = set(COMM.id) - set(existing_comments)
        remaining_comments = [c for c in COMM.id if c in remaining_comments]
        ### match words 
        LOG.debug('Trying %d comments', len(remaining_comments))
        if len(remaining_comments)>0:
            sent_dict = []
            total_sent = 0
            COMM = COMM[COMM.id.isin(remaining_comments)]
            for doc_index, doc in enumerate(fpm.nlp.pipe(COMM.body, n_process=10, n_threads = 20, disable = ["ner"])):
                for sent_id, sent in enumerate(doc.sents):
                    matched_keywords = sent._.flagtypematched
                    if matched_keywords:
                        sent_dict_entry = dict(zip(matched_keywords, [1]*len(matched_keywords)))
                        sent_dict_entry['comm_id']=COMM.id.iloc[doc_index]
                        sent_dict_entry['sent_id']=sent_id
                        sent_dict_entry['sent']=sent.text
                        sent_dict.append(sent_dict_entry)
            ### add to current dataframe
            sent_df_current = pd.DataFrame(sent_dict, columns = ['comm_id','sent_id','sent']+patterns)
            sent_df_current.iloc[:,3:] = sent_df_current.iloc[:,3:].fillna(0).astype('int64')
            total_sent += sent_df_current.shape[0]
            #sent_df_current.to_csv(rootfold+'/output/'+match_file, index = False, mode = 'a', header = False)
            if not sent_df_current.empty:
                ### add to complete dataframe
                sent_df = sent_df_current.append(sent_df).reset_index(drop=True)                
                LOG.debug('%5dK. Done: %4d comments, %4d sent / %6d total sent. Flags: %s', 
                          (chunk_id+1)*100, len(set(sent_df_current.comm_id)), sent_df_current.shape[0], total_sent, 
                          sent_df_current.flag.agg(sum))
            else:
                LOG.warning('%5dK. No new sentences.', (chunk_id+1)*100)
                break            
        else:
            LOG.warning('No new comments in comm file: %s',comm_file)
            break
        if not sent_df.empty:
            sent_df.to_csv(rootfold+'/output/'+match_file, index = False)
            LOG.info('Saved match file: %s',match_file)
    LOG.info('Finished comm file: %s',comm_file)