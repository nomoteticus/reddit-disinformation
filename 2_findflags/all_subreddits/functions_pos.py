#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:34:21 2020

@author: j0hndoe
"""

import spacy
from spacy.tokens import Span
nlp = spacy.load('en_core_web_sm')

### Define keywords

notsure = set(['example',
               'stop','do not'
'you'
'spread', 'distribute','propagate','post',
                  'stop', 'produce','make', 'come',
])

vague_subjects = set(['it','this'])
clear_subjects = set(['article','submission','sub','post',
                      'title','headline','header',
                      'source','website','site','url','link'])
subjects = vague_subjects.union(clear_subjects)
predicates = set(['be',
                  'look','sound','feel','seem','smell','stink'])
subjects_dependent = set(['i','me'])
predicates_dependent = set(['guess','call','think','beleive','say','go','know'])
prepositions = set(['like','of'])
fullobjects = set(['disinformation','misinformation','malinformation',
                   'misleading','clickbait','innacuracies',
                   'falsehood','fabrication',
                   'untrustworthy','unreliable','unverified',
                   'bullshit','bs',
                   'propaganda'])
partialobjects = set(['news','information','info']).union(clear_subjects)
partialattributes = set(['fake','false','bogus','fabricated',\
                         'untrustworthy','unreliable','unverified',
                         'misleading','editorialized','clickbait','sensationalized',
                         'manipulated','manipulative','inaccurate',
                         'bullshit','bs',
                         'propaganda'])
negatives = set(['not','no','none'])
quotes = set(["'",'"','”'])
sarcasms = set(["\s","/s","/sarcasm","\sarcasm","sarcasm","lol"])


all_keywords =\
{
     'subjects': subjects,
     'predicates': predicates,
     'fullobjects': fullobjects,
     'partialobjects':partialobjects,
     'attributes':partialattributes
}
    
#with open('../input/flag_keywords/subjects.txt') as file: for s in subjects: file.write(s+'\n')
    
###
fake_regex_wrds = 'fake|false\\w*|bogus|'+\
                  'mislead\\w*|disinf\\w*|misinf\\w*|malinf\\w*|'+\
                  'bullshit|\\bbs\\b|'+\
                  'news|info\\w*|title|article|source|submission'

### Functions


def extract_pos(text, df=False):
    """extracts essential POS from sentence"""
    pos_list = [(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, list(t.ancestors),list(t.children)) for t in nlp(text)]
    if df:
        return(pd.DataFrame(pos_list, columns=['text','lemma','pos','tag','dep','ancestors','children']))   
    else:
        return(pos_list)

def children_in(tok, wordlist, childrenofchildren = True):
    """checks if children of token are in wordlist"""
    value = False
    for a in tok.children:
        if a.lower_ in wordlist and a.pos_!='ccomp':
            value = a.lower_
            break
        if childrenofchildren:
            for b in a.children:
                if b.lower_ in wordlist and b.pos_!='ccomp':
                    value = b.lower_
                    break                
    return(value)


def test_root(root):
    if root.lemma_ in predicates_dependent:
        if subjects_dependent.intersection([rc.lower_ for rc in root.children if rc.dep_=='nsubj']):
            for rc in root.children:
                if rc.dep_=='ccomp':
                    root = rc
    return(root)


def is_negative(tok):
    return tok.dep_ == "neg" or (tok.dep_=='det' and tok.lower_ in negatives)

def is_proper_object(c,o,ironic,neg):    
    if c.lower_ in fullobjects:
        o = c.text
    ### Composite Adj+Noun ('fake news')
    elif (c.lemma_ in partialobjects) and children_in(c, partialattributes):
        o = children_in(c, partialattributes) + ' ' + c.text      
    elif children_in(c, fullobjects):
        o = children_in(c, fullobjects)
    elif children_in(c, fullobjects):
        print('do something')
    if children_in(c, quotes, False):
        ironic = True
    if children_in(c, negatives, False):
        neg = True
    return (o, ironic, neg)

def fake_sent_pos(sent):
    """checks if sentence follows a specific structure and uses specific keywords"""
    v,s,o,neg,ironic = [False]*5
    final_phrase = None
    root = sent.root
    nsubjects = 0
    ### If root is a first person (I think, I guess)
    if root.lemma_ in predicates_dependent:
        if subjects_dependent.intersection([rc.lower_ for rc in root.children if rc.dep_=='nsubj']):
            for rc in root.children:
                if rc.dep_=='ccomp':
                    root = rc    
    ### If root is a verb from the list (is, looks like)
    if root.lemma_ in predicates:            
        v = root.text
        for c in root.children:
            ### Find subject (this, article)
            if c.dep_ in ["nsubj","nsubjpass"]:
                nsubjects += 1
                if c.lemma_ in subjects or c.lower_ in subjects:
                    s = c.text
            ### Find object 
            elif c.dep_ in ["dobj","pobj","attr","acomp","nsubjpass","oprd","compound"]:
                ### Noun ('disinformation')
                o, ironic, neg = is_proper_object(c, o,ironic, neg)
            ### looks,sounds LIKE
            elif c.dep_ == 'prep' and c.lower_ in prepositions:
                for g in c.children:
                    if g.dep_ in ["dobj","pobj","attr","acomp","nsubjpass","oprd","compound"]:
                        o, ironic, neg = is_proper_object(g, o,ironic, neg)
                        o = '%s %s' % (c, o) if o else o
            ### 
            if is_negative(c):
                neg = True
            elif c.lower_ in sarcasms:
                ironic = True
        ### Build phrase
        if v and o and not neg and not ironic:
            if s:
                final_phrase = '%s %s %s' % (s, v, o)
            elif nsubjects==0 and root.lemma_ !="be":
                final_phrase = '%s %s' % (v, o)
    ### If there is no predicate
    elif root.pos_ in ['NOUN','PROPN']:
        ### Check if object is in list
        o, ironic, neg = is_proper_object(root, o,ironic, neg)
        if o and not neg and not ironic:
            final_phrase = o
    return(final_phrase) 

Span.set_extension("fake_news_flag", getter=fake_sent_pos, force=True)

def get_sent(text):
    return nlp(text).sents

#txt = 'if you believe a word from those lying propaganda mouth pieces then youre more ignorant than an anti-vaxxer. '
#for sent in get_sent(txt):
#    print(sent)
#    print(sent._.fake_news_flag)
