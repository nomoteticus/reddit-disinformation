#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 02:34:04 2020

@author: j0hndoe
"""

#https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/
#https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py

import pandas as pd
import os

rootfold = os.getcwd()

DF = pd.read_csv(rootfold+'/4_check/data/classified_regex_labeled.csv').set_index('id')

CODED = DF[['word','subm_title','domain','comm_body','sample','class','POSflag2','consensus']]
CODED = CODED[CODED.consensus.isin(['f','n'])]
CODED['isflag'] = [1 if x == 'f' else 0 for x in CODED.consensus]
CODED_all = CODED[CODED['sample'] == "all"]

import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline


# Create our list of punctuation marks
punctuations = string.punctuation
punctuations = '!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

def print_clas_diagn(y_test, y_pred):
    print('Accuracy: %5.2f\nPrecision: %5.2f\nRecall : %5.2f\nF1: %5.2f\nROC: %5.2f\n' % 
          (metrics.accuracy_score(y_test, y_pred),
           metrics.precision_score(y_test, y_pred),
           metrics.recall_score(y_test, y_pred),
           metrics.f1_score(y_test, y_pred),
           metrics.roc_auc_score(y_test, y_pred)))


bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

from sklearn.model_selection import train_test_split


X = CODED_all['comm_body'] # the features we want to analyze
ylabels = CODED_all['isflag'] # the labels, or answers, we want to test against
X2 = CODED['comm_body']
ylabels2 = CODED['POSflag2'] # the labels, or answers, we want to test against


X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)

pos_train_index = list(y_train.index) + list(CODED[CODED['sample'] == "pos"].index)
X_train2, X_test2 = X2.loc[pos_train_index] , X2.loc[y_test.index]
y_train2, y_test2 = ylabels2.loc[pos_train_index] , ylabels2.loc[y_test.index]


#testindex = CODED[CODED['sample']=="all"].sample(200).index
#X_train , X_test = X.loc[~X.index.isin(testindex)] , X.loc[X.index.isin(testindex)]
#y_train , y_test = ylabels.loc[~ylabels.index.isin(testindex)] , ylabels.loc[ylabels.index.isin(testindex)]



# Logistic Regression 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from datetime import time, datetime
from imblearn.pipeline import Pipeline as imbPipeline


classifier = RandomForestClassifier()

undersample = RandomUnderSampler(sampling_strategy='majority')
oversample = RandomOverSampler(sampling_strategy='minority')

# Create pipeline using Bag of Words
pipe = imbPipeline([("cleaner", predictors()),
                    ('vectorizer', tfidf_vector),
                    #('sampler', oversample),
                    ('classifier', classifier)])

max_features = (50,250,500)
n_estimators = (50, 100, 300)
max_depth = (10, 30, 50)
min_samples_split = (2, 5, 10, 15, 100)
min_samples_leaf = (1, 2, 5, 10)

parameters = dict(vectorizer__max_features = max_features, 
                  classifier__n_estimators = n_estimators, 
                  classifier__max_depth = max_depth)

classifier.get_params().keys()


#### ML with MANUAL labels

if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the classifier
    rand_search = RandomizedSearchCV(pipe, parameters, n_iter = 30,
                                     n_jobs=28, verbose=1, scoring='f1')
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    print(parameters)
    t0 = datetime.now()
    rand_search.fit(X_train,y_train)
    print("done in %5.2fm" % ((datetime.now() - t0).seconds/60))
    print()
    print("Best score: %0.3f" % rand_search.best_score_)
    print("Best parameters set:")
    best_parameters = rand_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

from sklearn import metrics
best_estimator = rand_search.best_estimator_
Y_pred = best_estimator.predict(X_test)
Y_pred_prob = best_estimator.predict_proba(X_test)

prdf_init = pd.DataFrame({'observed': y_test, 'predicted': Y_pred})
pd.crosstab(index = prdf_init['observed'],columns = prdf_init['predicted'])
print_clas_diagn(y_test, Y_pred)



#### ML with POS labels

if __name__ == "__main__":
    # find the best parameters for both the feature extraction and the classifier
    rand_search_pos = RandomizedSearchCV(pipe, parameters, n_iter = 30,
                                         n_jobs=28, verbose=1, scoring='accuracy')
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipe.steps])
    print("parameters:")
    print(parameters)
    t0 = datetime.now()
    rand_search_pos.fit(X_train2, y_train2)
    print("done in %5.2fm" % ((datetime.now() - t0).seconds/60))
    print()
    print("Best score: %0.3f" % rand_search_pos.best_score_)
    print("Best parameters set:")
    best_parameters = rand_search_pos.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

from sklearn import metrics
best_estimator_pos = rand_search_pos.best_estimator_
Y_pred_pos = best_estimator_pos.predict(X_test2)
Y_pred_pos_prob = best_estimator.predict_proba(X_test2)

prdf_init_pos = pd.DataFrame({'observed': y_test, 'predicted': Y_pred_pos})
pd.crosstab(index = prdf_init_pos['observed'],columns = prdf_init_pos['predicted'])
print_clas_diagn(y_test, Y_pred_pos)




