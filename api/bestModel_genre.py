#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:09:17 2019

@author: ashleshashinde
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


import warnings
warnings.simplefilter("ignore")

#Loading data#
lyrics_mod_sample = pd.read_csv("train_cleaned_lyrics.tsv",sep="\t",encoding='utf-8')
#print(lyrics_mod_sample.shape)
X_train, X_valid, y_train, y_valid = train_test_split(lyrics_mod_sample["Cleaned_lyric"], lyrics_mod_sample["genre"], test_size=0.2, random_state=42)
lyrics_mod_sample.dropna(inplace = True)
#print(lyrics_mod_sample.shape)

svc = LinearSVC(
    C=1.0,
    class_weight='balanced',
    dual=True,
    fit_intercept=True,
    intercept_scaling=1,
    loss='squared_hinge',
    max_iter=1000,
    multi_class='ovr',
    penalty='l2',
    random_state=0,
    tol=1e-05, 
    verbose=0
)

tfidf = TfidfVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    stop_words=None,
    ngram_range=(1, 3),
    analyzer='word',
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
)

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svc', svc),
])


pipeline.fit(X_train.values.astype('U'),y_train)
y_valid_pred = pipeline.predict(X_valid.values.astype('U'))

pip_score = accuracy_score(y_valid,y_valid_pred)
print("Linear SVC with TF-IDF score", pip_score*100)

joblib.dump(pipeline, 'pipeline_genre.pkl')
