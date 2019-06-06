# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:17:07 2019

@author: 18123
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

#Reading the csv files
train_lyrics = pd.read_csv("train_lyrics_1000.csv")
valid_lyrics = pd.read_csv("valid_lyrics_200.csv")


#Checking for imbalance of Labels
plot_imbal_labels = train_lyrics.groupby(["mood"]).size()
plot_imbal_labels = plot_imbal_labels / plot_imbal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_imbal_labels.keys(), plot_imbal_labels.values).set_title("Happy vs Sad")
ax.set_ylabel('Number of samples')


plot_imbal_labels = train_lyrics.groupby(["genre"]).size()
plot_imbal_labels = plot_imbal_labels / plot_imbal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_imbal_labels.keys(), plot_imbal_labels.values).set_title("Genres")
ax.set_ylabel('Number of samples')



#Preprocessing of the data
def preprocess(given_review):
    review = given_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return (' '.join(review))


train_lyrics['Cleaned_Lyric'] = train_lyrics["lyrics"].apply(lambda x :preprocess(x))
train_lyrics.to_csv("train_cleaned_lyrics.tsv", sep='\t', index = False)

valid_lyrics['Cleaned_Lyric'] = valid_lyrics["lyrics"].apply(lambda x :preprocess(x))
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10)
X_train_bow = cv.fit_transform(train_lyrics['Cleaned_Lyric']).toarray()
X_test_bow = cv.fit_transform(valid_lyrics['Cleaned_Lyric']).toarray()


y_train = train_lyrics["mood"]
y_test = valid_lyrics["mood"]

pickle.dump(cv, open("/Users/ashleshashinde/Desktop/flask-by-example/api/naivebCV.pkl","wb"))



#Naive Bayes 

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
classifier_mulnb = MultinomialNB()
classifier_mulnb.fit(X_train_bow, y_train )
y_test_pred_nulnb = classifier_mulnb.predict(X_test_bow)
mulnb_score = accuracy_score(y_test,y_test_pred_nulnb)
print("Multinomial Naive Bayes score", mulnb_score)


#Predicting one sample
x_test_one_sample = cv.transform([valid_lyrics["Cleaned_Lyric"][0]])
y_test_pred_onesample = classifier_mulnb.predict(x_test_one_sample)


pickle.dump(classifier_mulnb, open("/Users/ashleshashinde/Desktop/flask-by-example/api/naivebmodel.pkl","wb"))

#import model
#import numpy as np
#from numpy import nan as Nan
#retrain_df = model.new_df
#
#
#train_copy = train_lyrics.copy()
#
#train_copy = train_copy.reindex(sorted(train_copy.columns), axis=1)
#retrain_df = retrain_df.reindex(sorted(retrain_df.columns), axis=1)
#
#retrain_df["Key"]=retrain_df.index
#retrain_df=retrain_df.reset_index()
#retrain_df=retrain_df.drop(columns=['index'])
#
#train_copy['Key'] = train_copy[['title', 'artist']].apply(lambda x: ' '.join(x), axis=1)
#train_copy=train_copy.drop(columns=['file'])
#train_copy=train_copy.drop(columns=['year'])
#train_copy=train_copy.drop(columns=['genre'])
#train_copy=train_copy.drop(columns=['lyrics'])
#train_copy = pd.concat([train_copy, retrain_df],sort=True)
##pad_zeroes = train_copy.shape[0]-retrain_df.shape[0]
##pad = pd.Series([Nan,Nan,Nan,Nan], index=['title', 'artist', 'lyrics', 'mood'])
##for i in range(pad_zeroes):
##    retrain_df  = retrain_df.append(pad, ignore_index = True)
#
#
#train_copy = pd.merge(train_copy,retrain_df,on = "Key", how = 'outer')
##train_copy['mood'] = np.where(train_copy['Index'] == retrain_df["Index"], retrain_df["mood"],train_copy['mood'])


######genre######3
from sklearn.model_selection import train_test_split
lyrics_mod_sample = pd.read_csv("train_cleaned_lyrics.tsv",sep="\t",encoding='utf-8')
print(lyrics_mod_sample.shape)
X_train, X_valid, y_train, y_valid = train_test_split(lyrics_mod_sample["Cleaned_lyric"], lyrics_mod_sample["genre"], test_size=0.2, random_state=42)
lyrics_mod_sample.dropna(inplace = True)
print(lyrics_mod_sample.shape)



#Calculating the no-of samples by no.of words per sample

def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)

number_of_words_per_sample = get_num_words_per_sample(lyrics_mod_sample['Cleaned_lyric'])
no_of_samples = lyrics_mod_sample.shape[0]

#we are checking no. of words per samples to no. of sample ratio as as criteria to create our model using n-gram approach or sequence approach. If s_w_ratio>1500, sequence approach otherwise n-gram.
s_w_ratio  = no_of_samples/number_of_words_per_sample
print(s_w_ratio)




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10)

X_train_bow = cv.fit_transform(X_train.values.astype('U')).toarray()
X_valid_bow = cv.fit_transform(X_valid.values.astype('U')).toarray()

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
classifier_mulnb = MultinomialNB()
print("Training...............")
classifier_mulnb.fit(X_train_bow, y_train)
y_valid_pred_mulnb = classifier_mulnb.predict(X_valid_bow)
mulnb_score = accuracy_score(y_valid,y_valid_pred_mulnb)
print("Multinomial Naive Bayes score", mulnb_score)


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
svm_clf =LinearSVC(
        C=1.0,
        class_weight='balanced',
        dual=False,
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

print("Training SVM ...............")
svm_clf.fit(X_train_bow, y_train)
y_valid_pred_svm = svm_clf.predict(X_valid_bow)
svm_score = accuracy_score(y_valid,y_valid_pred_svm)
print("Multinomial Naive Bayes score", svm_score)



from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
print("Training RF ...............")
rf_clf.fit(X_train_bow, y_train)
y_valid_pred_rf = rf_clf.predict(X_valid_bow)
rf_score = accuracy_score(y_valid,y_valid_pred_rf)
print("RF score", rf_score)



#---------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

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
    ngram_range=(1, 2),
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
print("SVM with TF-IDF score", pip_score)

import pickle
from joblib import dump, load

dump(pipeline,'/Users/ashleshashinde/Desktop/flask-by-example/api/ pipe')


################################################################3

import model
import numpy as np
from numpy import nan as Nan
retrain_df = model.new_df
retrain_df["Key"]=retrain_df.index
retrain_df=retrain_df.reset_index()
retrain_df=retrain_df.drop(columns=['index'])
retrain_df['Cleaned_Lyric'] = retrain_df["lyrics"].apply(lambda x :preprocess(x))
train_lyrics['Key'] = train_lyrics[['title', 'artist']].apply(lambda x: ' '.join(x), axis=1)
for index, row in retrain_df.iterrows():
    key=row['Key']
    found = train_lyrics[train_lyrics['Key'].str.contains(row["Key"])]
    if found.empty:
#        train_copy = pd.concat([train_copy, row],sort=True)
        train_lyrics=train_lyrics.append(row)
    else:
        train_lyrics.at[found.index,'mood']=row['mood']