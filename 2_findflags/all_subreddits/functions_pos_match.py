# -*- coding: utf-8 -*-

import pandas as pd
import re
import os
import spacy
from spacy.tokens import Span, Doc
from spacy.matcher import Matcher
nlp = spacy.load('en_core_web_sm')

rootfold = re.match('^.*reddit-disinformation', os.path.dirname(os.path.realpath(__file__))).group(0)

def extract_pos(text, df=False):
    """extracts essential POS from sentence"""
    pos_list = [(t.text, t.lemma_, t.pos_, t.tag_, t.dep_, list(t.ancestors),list(t.children)) for t in nlp(text)]
    if df:
        return(pd.DataFrame(pos_list, columns=['text','lemma','pos','tag','dep','ancestors','children']))   
    else:
        return(pos_list)


### I. Build vocabulary
##

# questionable, sketchy, not reliable, click-baity, debunk(ed)
# doubt it's real

class pos:
    nouns = ['NOUN','PROPN']
    adjs = ['ADJ']

class deps:
    root = ["ROOT"]
    predicates = ["ROOT","xcomp"]
    subjects = ["nsubj","nsubjpass"]
    objects = ["dobj","pobj","attr","acomp","oprd","conj","compound","nsubjpass"]
    attributes =  ['amod'] + objects
    pos = ['NOUN','PROPN']
    neg = ["neg","det"]
#pos.extract_pos(' '.join(attributes))

class voc:
    class subj:        
        article = ['article','submission','sub','post']
        title = ['title','headline','header']
        source = ['source','website','site','url','link']
        ###
        fpers = ['i','me','we']
        secpers = ['you']
        ###
        this = ['it','this','that','here']        
    class pred:
        be = ['be']
        feel = ['look','sound','feel','seem','smell','stink']
        think = ['guess','think','beleive','say','know','feel','suspect']
        beleive = ['beleive','fall','buy']
        call = ['call']
        stop = ['stop','quit','refrain','do']
        report = ['flag','report','remove']
        spread = ['spread','propagate','spew','distribute','promote','post','submit']
    class obj:
        news = ['news','information','info'] 
        disinfo = ['disinformation','misinformation','malinformation']
        clickbait = ['clickbait','innacuracy','innacuracies']
        falsehood = ['falsehood','fabrication']
        bs = ['bullshit','bs']
        propaganda = ['propaganda']
    class attr:
        false = ['fake','false','bogus','fabricated', 'manipulated','manipulative','inaccurate']
        misleading = ['misleading','mislead','editorialized','editorialize','clickbait','sensationalized','sensationalize','sensationalist']
        unreliable = ['untrustworthy','unreliable','unverified']
        bs = ['bullshit','bs']
        propaganda = ['propaganda']
        real = ['real','true','correct']
        reliable = ['reliable','verified','credible']
    class neg:
        neg = ['not','no']
   
v = voc()
    
# DISINFORMATION, FAKE+FALSEHOOD, MISLEADING+CLICKBAIT, UNRELIABLE, BULLSHIT, PROPAGANDA #
  
## Define wild cards and/or negation
##
WC = [{"OP":"?"}]
NEGATION = [{"LEMMA":{"IN":['not','no']}}]
NONEGWC  = [{"LEMMA":{"NOT_IN":v.neg.neg},"OP":"?"}]


### II. Build phrase patterns
##

# This is disinformation. Article seems like propaganda.
# Article has to be propaganda
pattern_svo  = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.this + v.subj.article + v.subj.title + v.subj.source}}, 
    *NONEGWC*5,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.bs + v.obj.falsehood}}]

# This is a misleading title. Article souds like fake news.
# This has to be fake news.
pattern_svao  = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.this + v.subj.article + v.subj.title + v.subj.source}}, 
    *NONEGWC*5,
    {"DEP":{"IN": deps.predicates}, 
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.news + v.subj.article + v.subj.source + v.subj.title}}]

# Disinformation! Bullshit!
pattern_o  = \
   [#*NONEGWC*3,
    {"DEP":"ROOT", "POS":{"IN": pos.nouns + pos.adjs},
     "LEMMA":{"IN":v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.bs + v.obj.falsehood + \
                   v.attr.false + v.attr.misleading}}]

# Fake news!
pattern_ao  = \
   [#*NONEGWC*3,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs}},
    *NONEGWC*5,
    {"DEP":"ROOT", "POS":{"IN": pos.nouns}, "OP":"+",
     "LEMMA":{"IN":v.obj.news + v.subj.article + v.subj.source + v.subj.title}}]

# Source is unreliable. Title seems false.
pattern_sva = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.article + v.subj.title + v.subj.source + v.subj.this}}, 
    *NONEGWC*5,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.unreliable + v.attr.misleading}}]


# Source is not reliable.
pattern_svna = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.article + v.subj.title + v.subj.source}}, 
    *WC*5,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *WC*2,
    {"DEP":{"IN":deps.neg}, 
     "LEMMA":{"IN":v.neg.neg}},
    *WC*2,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.attr.real + v.attr.reliable}}]

# Article does not feel real.
# Source may not be reliable.
pattern_svna2 = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.article + v.subj.title + v.subj.source}}, 
    *WC*5,
    {"DEP":{"IN":deps.neg}, 
     "LEMMA":{"IN":v.neg.neg}},
    *WC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.feel + v.pred.be}},
    *WC*3,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.attr.real + v.attr.reliable}}]

# Not a reliable source.
pattern_naao = \
   [{"DEP":{"IN":deps.neg}, 
     "LEMMA":{"IN":v.neg.neg}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes},
     "LEMMA":{"IN":v.attr.real + v.attr.reliable}},
    *WC*5,
    {"DEP":{"IN":deps.objects + ["ROOT"]}, 
     "LOWER":{"IN":v.subj.article + v.subj.title + v.subj.source}}]

# I think this is propaganda.
pattern_ivsvo = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.think}},
    *NONEGWC*3,
    {"DEP":"nsubj", "OP":"+",
     "LEMMA":{"IN":v.subj.this + v.subj.article + v.subj.title + v.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ccomp", 
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.falsehood}}]

# I beleive this is fake news.
pattern_ivsvao = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.think}},
    *NONEGWC*3,
    {"DEP":"nsubj", "OP":"+",
     "LOWER":{"IN":v.subj.this + v.subj.article + v.subj.title + v.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ccomp", 
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.news + v.subj.article + v.subj.source + v.subj.title}}]

# I think title is misleading
pattern_ivsva = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.think}},
    *NONEGWC*3,
    {"DEP":"nsubj", "OP":"+",
     "LOWER":{"IN":v.subj.article + v.subj.title + v.subj.source}}, 
    *NONEGWC*5,
    {"DEP":"ccomp", 
     "LEMMA":{"IN":v.pred.be + v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs}}]


# I call bullshit
pattern_ivo = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.call}},
    *NONEGWC*3,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.bs + v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.falsehood}}]

# I call fake news
pattern_ivao = \
   [{"DEP":{"IN":deps.subjects},
     "LOWER":{"IN":v.subj.fpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.call}},
    *NONEGWC*3,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda }},
    *NONEGWC*3,
        {"DEP":{"IN":deps.objects}, "OP":"+",
         "LEMMA":{"IN":v.obj.news + v.subj.article + v.subj.source + v.subj.title}}]
    
# Looks like disinformation.
pattern_vo = \
   [{"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.bs + v.obj.falsehood}}]

# Sounds like fake news.
pattern_vao = \
   [{"DEP":"ROOT", 
     "LEMMA":{"IN":v.pred.feel}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":v.obj.news + v.subj.article + v.subj.source + v.subj.title}}]

# You are propagating disinformation.
pattern_yvo = \
   [{"DEP":{"IN":deps.subjects}, 
     "LOWER":{"IN":voc.subj.secpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.spread}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# You are spreading falsehoods.
pattern_yvao = \
   [{"DEP":{"IN":deps.subjects}, 
     "LOWER":{"IN":voc.subj.secpers}}, 
    *NONEGWC*3,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.spread }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, 
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# Stop spreading the disinformation! 
# Don't fall for / spread propaganda
# Reported for spreading propaganda
pattern_stvo = \
   [{"DEP":{"IN":["ROOT","aux"]}, "POS":{"IN":["VERB","AUX","PROPN"]},
     "LOWER":{"IN":voc.pred.stop+voc.pred.report}},
    *WC*5,
    {"LEMMA":{"IN":voc.pred.spread + voc.pred.beleive}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}}]

# Stop spreading false news!
pattern_stvao = \
   [{"DEP":{"IN":["ROOT","aux"]}, "POS":{"IN":["VERB","AUX","PROPN"]},
     "LOWER":{"IN":voc.pred.stop+voc.pred.report}},
    *WC*5,
    {"LEMMA":{"IN":voc.pred.spread + voc.pred.beleive}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, 
     "LEMMA":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.objects}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}}]

# This is how disinformation spreads.
pattern_svsv  = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.this}}, 
    *NONEGWC*3,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.be}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.subjects}, "OP":"+",
     "LEMMA":{"IN":v.obj.disinfo + v.obj.clickbait + v.obj.propaganda + v.obj.bs + v.obj.falsehood}},
    *NONEGWC*3,
    {"DEP":"ccomp",
     "LEMMA":{"IN":v.pred.spread + v.pred.feel}}    
    ]

# This is what fake news feels like.
pattern_svsav  = \
   [{"DEP":{"IN":deps.subjects}, "OP":"+",
     "LOWER":{"IN":v.subj.this}}, 
    *NONEGWC*3,
    {"DEP":{"IN": deps.predicates},
     "LEMMA":{"IN":v.pred.be}},
    *NONEGWC*5,
    {"DEP":{"IN":deps.attributes}, "OP":"+",
     "LEMMA":{"IN":v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.subjects}, "OP":"+",
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title }},
    *NONEGWC*3,
    {"DEP":"ccomp",
     "LEMMA":{"IN":v.pred.spread + v.pred.feel}}    
    ]


### Not implemented

# Propaganda reported
pattern_ost = \
    [{"DEP":{"IN":deps.subjects}, 
     "LOWER":{"IN":voc.obj.disinfo + voc.obj.clickbait + voc.obj.propaganda + voc.obj.bs + voc.obj.falsehood}},
     *NONEGWC*2,
     {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.report}}]
 
# Fake news reported
pattern_aost = \
   [{"DEP":{"IN":deps.attributes}, 
     "LOWER":{"IN":voc.attr.false + voc.attr.misleading + voc.attr.unreliable + voc.attr.propaganda }},
    *NONEGWC*5,
    {"DEP":{"IN":deps.subjects}, 
     "LEMMA":{"IN":voc.obj.news + voc.subj.article + voc.subj.source + voc.subj.title}},
    *NONEGWC*2,
    {"DEP":"ROOT", 
     "LEMMA":{"IN":voc.pred.report}}]


              
subjects = v.subj.this + v.subj.article + v.subj.title + v.subj.source + v.subj.fpers + v.subj.secpers
predicates = v.pred.be + v.pred.feel + v.pred.report + v.pred.spread + v.pred.stop + v.pred.think
objects_full = v.obj.disinfo + v.obj.clickbait + v.obj.falsehood + v.obj.propaganda + v.obj.bs
objects_all = objects_full + v.obj.news
attributes = v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs 


### III. Build matcher
##

#del(matcher)

structurematcher = Matcher(nlp.vocab)
#This/article is/looks like FN/D.
structurematcher.add("svo",None,pattern_svo)
structurematcher.add("svao",None,pattern_svao)
#FN/D!
structurematcher.add("o",None,pattern_o)
structurematcher.add("ao",None,pattern_ao)
# Seems like FN/D
structurematcher.add("vo",None,pattern_vo)
structurematcher.add("vao",None,pattern_vao)
# I think this/article is FN/D.
structurematcher.add("ivsvo",None, pattern_ivsvo)
structurematcher.add("ivsvao",None, pattern_ivsvao)
structurematcher.add("ivsva",None, pattern_ivsva)
# I call BS/FN
structurematcher.add("ivo",None,pattern_ivo)
structurematcher.add("ivao",None,pattern_ivao)
# Article seems fake.
structurematcher.add("sva",None,pattern_sva)
# Article is not true/real.
structurematcher.add("svna", None, pattern_svna)
structurematcher.add("svna2", None, pattern_svna2)
structurematcher.add("naao", None, pattern_naao)
#You spread FN/D
structurematcher.add("yvo",None,pattern_yvo)
structurematcher.add("yvao",None,pattern_yvao)
#Stop spreading FN/DISINFO.            
structurematcher.add("stvo",None,pattern_stvo)
structurematcher.add("stvao",None,pattern_stvao)
#This is how FN/PROP spreads / feels like.
structurematcher.add("svsv",None,pattern_svsv)
structurematcher.add("svsav",None,pattern_svsav)
#FN reported.
#structurematcher.add("ost",None,pattern_ost)
#structurematcher.add("aost",None,pattern_aost)


objects = v.obj.disinfo + v.obj.clickbait + v.obj.falsehood + v.obj.propaganda + v.obj.bs
attributes = v.attr.false + v.attr.misleading + v.attr.unreliable + v.attr.propaganda + v.attr.bs 


#nlp = spacy.load('en_core_web_sm', disable = ['parser','tagger','ner'])
#nlp.add_pipe(nlp.create_pipe('sentencizer'))

isflagmatcher = Matcher(nlp.vocab)
isflagmatcher.add("flag",None, pattern_svo, pattern_svao, pattern_o, pattern_ao, pattern_vo, pattern_vao,
                               pattern_ivsvo, pattern_ivsvao, pattern_ivo, pattern_ivao, pattern_ivsva,
                               pattern_sva, pattern_svna, pattern_svna2, pattern_naao,
                               pattern_yvo, pattern_yvao, 
                               pattern_stvo, pattern_stvao, pattern_svsv, pattern_svsav)
#, pattern_ost,pattern_aost)

flagtypematcher = Matcher(nlp.vocab)
flagtypematcher.add("flag",None, pattern_svo, pattern_svao, pattern_o, pattern_ao, pattern_vo, pattern_vao,
                                 pattern_ivsvo, pattern_ivsvao, pattern_ivo, pattern_ivao, pattern_ivsva,
                                 pattern_sva, pattern_svna, pattern_svna2, pattern_naao,
                                 pattern_yvo, pattern_yvao, 
                                 pattern_stvo, pattern_stvao, pattern_svsv, pattern_svsav)
#, pattern_ost,pattern_aost)
flagtypematcher.add("disinformation",None, [{"LOWER":{"IN":v.obj.disinfo}}])
flagtypematcher.add("fakenews",None, 
                    [{"LOWER":{"IN":v.obj.falsehood+v.attr.false}}])
flagtypematcher.add("bs",None, [{"LOWER":{"IN":v.obj.bs}}])
flagtypematcher.add("misleading",None, [{"LOWER":{"IN":v.attr.misleading+v.obj.clickbait}}])
flagtypematcher.add("unreliable",None, [{"LOWER":{"IN":v.attr.unreliable}}])
flagtypematcher.add("propaganda",None, [{"LOWER":{"IN":v.obj.propaganda}}])

def structurematched(span):
    return [nlp.vocab.strings[match_id] for match_id, start, end in structurematcher(span.as_doc())]
def flagtypematched(span):
    return [nlp.vocab.strings[match_id] for match_id, start, end in flagtypematcher(span.as_doc())]
def isflagmatched(span):
    return len(isflagmatcher(span.as_doc()))>0

Span.set_extension("structurematched", getter=structurematched, force=True)
Span.set_extension("flagtypematched", getter=flagtypematched, force=True)
Span.set_extension("isflagmatched", getter=isflagmatched, force=True)

### Specify regular expressions

# Disinformation, fake news, clickbait, unreliable sources
regex_flag = re.compile('|'.join([r'\b'+o for o in objects_all+attributes]))             
fake_regex = re.compile('|'.join([r'\b'+o for o in objects_all+attributes]))             

# Sarcasm and irony
sarcasm_regex = "\\\s\\b|\\/s\\b"  
irony_regex = '|'.join(["[*'\"]" + s for s in voc.attr.false + voc.obj.disinfo + voc.attr.propaganda])
joking_regex = '\\bjk\\b'
sarcasm_and_irony_regex = sarcasm_regex + '|' + irony_regex + '|' + joking_regex 

# Bot
bot_body_regex = "([iI] am a bot)|(bot just does our dirty work)"
bot_author_regex = "bot$|automod"

### set regexes
covid_regex =  ['china_flu'] + [re.sub("[\\*\\(\\)]","",'\\b' +w.rstrip()) \
                for w in open(rootfold+'/input/keywords_covid.txt','r')] 
covid_regex = re.compile('|'.join(covid_regex))
#fake_regex = re.compile('\\b(dis|mis|mal)info|\\bpropagand|\\bconspira|\\bfalse|\\bfake (news|info)')
