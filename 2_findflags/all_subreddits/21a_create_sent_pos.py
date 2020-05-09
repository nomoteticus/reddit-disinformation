#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:34:21 2020

@author: j0hndoe
"""
import sys
from colorama import Fore, Back, Style
import datetime
import itertools
import re
rootfold = %pwd
sys.path.append(rootfold+"/2_findflags/all_subreddits")

import pandas as pd
        
import spacy
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')

def extract_pos(text, df=False):
    """extracts essential POS from sentence"""
    pos_list = [(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, list(t.ancestors),list(t.children)) for t in nlp(text)]
    if df:
        return(pd.DataFrame(pos_list, columns=['text','lemma','pos','tag','dep','ancestors','children']))   
    else:
        return(pos_list)

def highlight_txt(words, content, style ='', forgr='', backgr=''):
    regex = "|".join(['(\\b' + w + '\\w*)' for w in words])
    replace = "".join(["\\%d" % (i + 1) for i in range(len(words))])
    replace = style + forgr + backgr + replace + Style.RESET_ALL
    return re.sub(regex, replace, content, re.IGNORECASE)

def highlight_keyw(text):
    formatted_text = text
    formatted_text = highlight_txt(subjects, formatted_text, style = Style.DIM, backgr = Back.YELLOW)
    formatted_text = highlight_txt(predicates, formatted_text, style = Style.DIM, backgr = Back.CYAN)
    formatted_text = highlight_txt(objects, formatted_text, style = Style.BRIGHT, forgr = Fore.BLACK, backgr = Back.YELLOW)
    formatted_text = highlight_txt(attributes, formatted_text, style = Style.BRIGHT, forgr = Fore.BLUE, backgr = Back.YELLOW)
    return formatted_text

def print_matches(doc, matches=False, colored = True):
    if not matches:
        doc = nlp(doc)
        matches = matcher(doc)
    for match_id, start, end in matches:
        string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        if colored:
            print(Style.BRIGHT + Back.WHITE + Fore.BLACK + string_id + ':  ' + span.text)
            print(Style.RESET_ALL)
        else:
            print(string_id + ':  ' + span.text)
    return matches


## I. Build vocabulary
##

subject_deps = ["nsubj","nsubjpass"]
object_deps = ["dobj","pobj","attr","acomp","oprd","conj","compound","nsubjpass"]
attribute_deps =  ['amod'] + object_deps
pos_nouns = ['NOUN','PROPN']
#pos.extract_pos(' '.join(attributes))

del(voc)
class voc:
    class subj:        
        article = ['article','submission','sub','post']
        title = ['title','headline','header']
        source = ['source','website','site','url','link']
        ###
        fpers = ['i','me','we']
        secpers = ['you']
        ###
        this = ['it','this','that']
    class pred:
        be = ['be']
        feel = ['look','sound','feel','seem','smell','stink']
        think = ['guess','call','think','beleive','say','know','feel']
        stop = ['stop','quit','do']
        report = ['flag','report']
        spread = ['spread','propagate','spew','distribute','promote','post']
    class obj:
        news = ['news','information','info'] 
        disinfo = ['disinformation','misinformation','malinformation']
        clickbait = ['clickbait','innacuracy','innacuracies']
        falsehood = ['falsehood','fabrication']
        bs = ['bullshit','bs']
        propaganda = ['propaganda']
    class attr:
        false = ['fake','false','bogus','fabricated', 'manipulated','manipulative','inaccurate']
        misleading = ['misleading','mislead','editorialized','clickbait','sensationalized']
        unreliable = ['untrustworthy','unreliable','unverified']
        bs = ['bullshit','bs']
        propaganda = ['propaganda']
        real = ['real','true']
        reliable = ['reliable','verified','credible']
    neg = ['not','no']

# DISINFORMATION, FAKE+FALSEHOOD, MISLEADING+CLICKBAIT, UNRELIABLE, BULLSHIT, PROPAGANDA #
  
## Define wild cards and/or negation
##
WC = [{"OP":"?"}]
NEGATION = [{"LEMMA":{"IN":['not','no']}}]
NONEGWC  = [{"LEMMA":{"NOT_IN":['not','no']},"OP":"?"}]



## II. Build phrase patterns
##

# This is disinformation. Article seems like propaganda.
pattern_svo  = \
   [{"DEP":{"IN":subject_deps}, "OP":"+",
     "LOWER":{"IN":voc.subj.this + voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# This is a misleading title. Article souds like fake news.
pattern_svao  = \
   [{"DEP":{"IN":subject_deps}, "OP":"+",
     "LOWER":{"IN":voc.subj.this + voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":attribute_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# Disinformation! Bullshit!
pattern_o  = \
   [#*NONEGWC*3,
    {"DEP":"ROOT", "POS":{"IN": pos_nouns}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# Fake news!
pattern_ao  = \
   [#*NONEGWC*3,
    {"DEP":{"IN":attribute_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":"ROOT", "POS":{"IN": pos_nouns}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# Source is unreliable. Title seems false.
pattern_sva = \
   [{"DEP":{"IN":subject_deps}, "OP":"+",
     "LOWER":{"IN":voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.false + voc.attr.unreliable + voc.attr.misleading}}]

# Source is not reliable.
pattern_svna = \
   [{"DEP":{"IN":subject_deps}, "OP":"+",
     "LOWER":{"IN":voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *WC*5,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *WC*2,
    {"DEP":{"IN": ["neg","det"]}, 
     "LEMMA":{"IN":voc.neg}},
    *WC*2,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.real + voc.attr.reliable}}]

# Article does not feel real.
pattern_svna2 = \
   [{"DEP":{"IN":subject_deps}, "OP":"+",
     "LOWER":{"IN":voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *WC*5,
    {"DEP":{"IN": ["neg","det"]}, 
     "LEMMA":{"IN":voc.neg}},
    *WC*2,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.feel + voc.pred.feel}},
    *WC*2,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.real + voc.attr.reliable}}]

# I think this is propaganda.
pattern_svsvo = \
   [{"DEP":{"IN":subject_deps},
     "LOWER":{"IN":voc.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.think}},
    *NONEGWC*3,
    {"DEP":"nsubj", "OP":"+",
     "LEMMA":{"IN":voc.subj.this + voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ccomp", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# I beleive this is fake news.
pattern_svsvao = \
   [{"DEP":{"IN":subject_deps},
     "LOWER":{"IN":voc.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.think}},
    *NONEGWC*3,
    {"DEP":"nsubj", "OP":"+",
     "LOWER":{"IN":voc.subj.this + voc.subj.article + voc.subj.title + voc.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ccomp", 
     "LEMMA":{"IN":voc.pred.be + voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":attribute_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

    
# Looks like disinformation.
pattern_vo = \
   [{"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# Sounds like fake news.
pattern_vao = \
   [{"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":attribute_deps}, "OP":"+",
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]


# You are propagating disinformation.
pattern_yvo = \
   [{"DEP":{"IN":subject_deps}, 
     "LOWER":{"IN":voc.subj.secpers}}, 
    *WC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.spread}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# You are spreading falsehoods.
pattern_yvao = \
   [{"DEP":{"IN":subject_deps}, 
     "LOWER":{"IN":voc.subj.secpers}}, 
    *WC,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.spread }},
    *NONEGWC*5,
    {"DEP":{"IN":attribute_deps}, 
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# Stop the disinformation! 
pattern_stvo = \
   [{"DEP":"ROOT", "POS":"VERB",
     "LEMMA":{"IN":voc.pred.stop+voc.pred.report}},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# Stop spreading falsehoods.
pattern_stvao = \
   [{"DEP":"ROOT", "POS":"VERB",
     "LEMMA":{"IN":voc.pred.stop+voc.pred.report}},
    *NONEGWC*5,
    {"DEP":{"IN":attribute_deps}, 
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":object_deps}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# Propaganda reported
pattern_ost = \
    [{"DEP":{"IN":subject_deps}, 
     "LOWER":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}},
     *NONEGWC*2,
     {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.report}}]
 
# Fake news reported
pattern_aost = \
   [{"DEP":{"IN":attribute_deps}, 
     "LOWER":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":subject_deps}, 
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}},
    *NONEGWC*2,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.report}}]



## III. Build matcher
##

del(matcher)

matcher = Matcher(nlp.vocab)
#This/article is/looks like FN/D.
matcher.add("svo",None,pattern_svo)
matcher.add("svao",None,pattern_svao)
#FN/D!
matcher.add("o",None,pattern_o)
matcher.add("ao",None,pattern_ao)
# Seems like FN/D
matcher.add("vo",None,pattern_vo)
matcher.add("vao",None,pattern_vao)
# I think this/article is FN/D.
matcher.add("svsvo",None,pattern_svsvo)
matcher.add("svsvao",None,pattern_svsvao)
# Article seems fake.
matcher.add("sva",None,pattern_sva)
# Article is not true/real.
matcher.add("svna", None, pattern_svna)
matcher.add("svna2", None, pattern_svna2)
#You spread FN/D
matcher.add("yvo",None,pattern_yvo)
matcher.add("yvao",None,pattern_yvao)
#Stop spreading FN/DISINFO.            
matcher.add("stvo",None,pattern_stvo)
matcher.add("stvao",None,pattern_stvao)
#FN reported.
matcher.add("ost",None,pattern_ost)
matcher.add("aost",None,pattern_aost)



## IV. Testing
##

### Test on one phrase - short 
text = "Reported this coronavirus disinformation"
print_matches(text)
extract_pos(text)

### Test on one phrase  - long
text = "This is fake news. You spread falsehood. Fake news! Disinformation. Article feels like it's misleading."
doc = nlp(text)
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(str(string_id) + ' | ' + str(span))


## V. Encode real data
##

# Import data

HASREGEX = pd.read_csv(rootfold+'/output/HASREGEXSH.csv')

txtsample = HASREGEX.comm_body.sample(1000)
txtsample

def chunk_df(df, n): 
    return (df[i:i+n] for i in range(0,df.shape[0],n))

def print_subm_comm(chunk):
    nmatches = 0
    previous_row = 'none'
    for row in chunk.itertuples():
        current_row = row.subm_id
        text = row.comm_body
        if len(text) < 1000:
            print('\n')
            if previous_row != current_row:
                print('\n\n\n')
                print('r/'+row.subreddit + ' / ' + row.subm_created[:10])
                print('--------------------------')
                print(row.subm_domain)
                print(row.subm_title)
                print('--------------------------')
                previous_row = current_row
            print(highlight_keyw(text))
            doc = nlp(text)
            for sent in doc.sents:
                matches = matcher(sent.as_doc())
                if len(matches)>0:
                    #print('--------------------------------------------------')
                    print_matches(sent, matches)
                    nmatches+=len(matches)
    print('Number of matches: %d/%d' % (nmatches, chunk.shape[0]))


# Color highlight on real data

iterdf = chunk_df(HASREGEX, 50)
for i in range(100): next(iterdf)
chunk = next(iterdf); print_subm_comm(chunk)


# Encode real data

lst_matches = []
for row in HASREGEX.itertuples():
    doc = nlp(row.comm_body)
    for sent_id, sent in enumerate(doc.sents):
        sent = sent.as_doc()
        current_iter = [(row.subreddit, row.comm_id, sent_id, str(sent), nlp.vocab.strings[match_id], str(sent[start:end]) ) \
                        for match_id, start, end in matcher(sent)]
        lst_matches += current_iter
        if len(lst_matches) %10 ==1 and len(current_iter)>0:
            print(current_iter)
    if row.Index % 1000 ==1:
        print('%d rows / %d matches | %s' % (row.Index, len(lst_matches), str(datetime.datetime.now())[:20]) )       

df_matches = pd.DataFrame(lst_matches, columns = ['comm_id','sent_id','type','matched'])

df_matches.matched.str.lower().value_counts().head(30)

### Save dataset 

df_matches.to_csv(rootfold+'/output/comm_matches.csv', index=False)
