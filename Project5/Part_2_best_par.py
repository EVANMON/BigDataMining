import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import pytz
import re
import nltk
import csv
import pandas as pd
from sklearn.feature_extraction import text
from nltk.corpus import  stopwords
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from nltk import  pos_tag
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
import itertools
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from math import sqrt

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def cv_rmse(X, y, kf, clf):
    test_rmse = []
    for train, test in kf.split(X):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_rmse.append(np.mean((y_pred - y_test)**2))
    return sqrt(np.mean(test_rmse))

def select_best_dim(dcp, X_raw, y, clf, dims):
    print(len(y))
    rmse_list = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for d in dims:
        print('** When decomposition dimension is ' + str(d) + ' **')
        if 'LSI' in dcp:
            svd = TruncatedSVD(n_components=d, random_state=42)
            X = svd.fit_transform(X_raw)
            rmse_list.append(cv_rmse(X, y, kf, clf))
        elif 'NMF' in dcp:
            nmf = NMF(n_components=d, random_state=42)
            X = nmf.fit_transform(X_raw)
            rmse_list.append(cv_rmse(X, y, kf, clf))
    idx = np.argmin(rmse_list)
    print('------------- For '+dcp+' --------------')
    print('The best degree is %i' % dims[idx])
    print('The best RMSE is ' + str(rmse_list[idx]))
    plt.plot(dims, rmse_list, label=dcp+' rmse')
    plt.title('RMSE vs Dim. Plot')
    plt.xlabel('Dimension')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.savefig('pics/RMSE_' + str(dcp)  + '.png')
    plt.close()



X_tfidf = load_sparse_csr('TFIDF.npz')
with open('true_y.csv', 'r') as f:
    reader = csv.reader(f)
    y = list(reader)

#X_tfidf = np.asarray(X_tfidf)
y = np.asarray(y)[0]
y = list(map(int, y))
y = np.array(y)
dims = [50, 100, 200, 300]
## Hard SVM
print('*** Hard SVM ***')
clf = svm.SVC(C=1000, random_state=42, kernel='linear')
select_best_dim('LSI_Hard_SVM', X_tfidf, y, clf, dims)
select_best_dim('NMF_Hard_SVM', X_tfidf, y, clf, dims)

## Soft SVM
print('*** Hard SVM ***')
clf = svm.SVC(C=0.001, random_state=42, kernel='linear')
select_best_dim('LSI_Soft_SVM', X_tfidf, y, clf, dims)
select_best_dim('NMF_Soft_SVM', X_tfidf, y, clf, dims)

## Logistic Regression
print('*** Logistic Regression ***')
clf = LogisticRegression(C=10000000, random_state=42)
select_best_dim('LSI_LR', X_tfidf, y, clf, dims)
select_best_dim('NMF_LR', X_tfidf, y, clf, dims)

## Naive Bayes
print('*** Naive Bayes ***')
clf = MultinomialNB()
select_best_dim('NMF_NB', X_tfidf, y, clf, dims)
