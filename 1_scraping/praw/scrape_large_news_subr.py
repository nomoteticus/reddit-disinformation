import praw
import pandas as pd
import re
import time
from datetime import datetime, timedelta

rootfold = '/home/j0hndoe/Dropbox/Python/reddit/Coronavirus/'

def subr_todict_minimal(s):
    return {'subm_id':s.id, 
            'subr_id': s.subreddit_id, 
            'author_id': str(s.author),
            'date_created': datetime.fromtimestamp(s.created) - timedelta(hours = 8),
            'date_added': datetime.now()
           }

APP_url = 'https://ssl.reddit.com/prefs/apps/'
reddit = praw.Reddit(client_id = 'nIjGq9V9Wn_ARw', 
                     client_secret = 'iFJcztvXJX1rwX3VAcNa2QzWmTE', 
                     user_agent = 'extracting news for academic research'
                    )

lst_subreddits = ['news', 'worldnews', 'Coronavirus', 'worldpolitics', 'worldevents', 
				  'NewsPorn','politics','uspolitics','europe','PoliticalDiscussion',
				  'TrueReddit','Positive_News','offbeat','inthenews']
#lst_subreddits = ['news']


all_submissions = pd.read_csv(rootfold+"all_submissions.csv")
print('Initial submissions: ' + str(all_submissions.shape))

already_there = set(all_submissions.subm_id)
print(already_there)

time_to_sleep = 15*60

if len(already_there)==0:    
    d = []
    for subred in lst_subreddits:
        for s in reddit.subreddit(subred).new(limit = None):
            if s.id not in already_there:
                try:
                    d.append(subr_todict_minimal(s)) 
                except:
                    print('Error: ' + str(s))                
        print('Finished subreddit: ' + subred + '; ncases: ' + str(len(d))  + '; ' + str(datetime.now()))
    all_submissions = pd.DataFrame(d)
    all_submissions.to_csv(rootfold+"all_submissions.csv", index = False)
    already_there = already_there.union(all_submissions.subm_id)
    time.sleep(time_to_sleep)

while len(already_there)>0:    
	d = []
	for subred in lst_subreddits:
		for s in reddit.subreddit(subred).new(limit = 25):
			if s.id not in already_there:
				try:
					d.append(subr_todict_minimal(s)) 
				except:
					print('Error: ' + str(s))
	if len(d)>0:
		all_submissions = pd.DataFrame(d)			
		all_submissions.to_csv(rootfold+"all_submissions.csv", index = False, mode = 'a', header = False)
		already_there = already_there.union(all_submissions.subm_id)
		print('Added: ' + str(all_submissions.shape)  + ' ' + str(datetime.now()) )
	else:
		print('Nothing to add: ' + subred + ' ' + str(datetime.now()) )
	time.sleep(time_to_sleep)