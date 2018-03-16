import json
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
import itertools
from scipy.sparse import csr_matrix
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
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def plot_confusion_matrix(cm, classes, s,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
        file_name = 'pics/conf_'+s+'_norm.png'
    else:
        print('Confusion matrix, without normalization')
        file_name = 'pics/conf_'+s+'_unnorm.png'

    print(cm)

    thresh = cm.max() / 2.
    if normalize:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            # plt.text(j, i, cm[i, j],
            plt.text(j, i, "%.2f"%cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
            # plt.text(j, i, "%.2f"%cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
    plt.close()


def plot_roc(pre, tar, s):
    fpr, tpr, _ = roc_curve(tar, pre)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='AUC = %0.2f' %roc_auc)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig('pics/ROC_'+ s + '.png')
    plt.close()


def fit_predict_and_plot_roc(clf, train_data, train_label, test_data, test_label, s):
    clf.fit(train_data, train_label)
    # pipeline1.predict(twenty_test.data)

    pre = clf.predict(test_data)

    accuracy = accuracy_score(test_label, pre)
    precision = precision_score(test_label, pre)
    recall = recall_score(test_label, pre)

    print('Accuracy of', s, 'is', accuracy)
    print('Precision of', s, 'is', precision)
    print('Recall of', s, 'is', recall)

    plot_roc(pre, test_label, s)
    print(clf.classes_)
    # Compute confusion matrix
    #class_names = ['MA', 'WA']
    class_names = ['MA' if lb==0 else 'WA' for lb in clf.classes_]
    cnf_matrix = confusion_matrix(test_label, pre)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=class_names, s=s, title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    #plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, s=s, normalize=True, title='Normalized confusion matrix')
    #plt.show()

X_tfidf = load_sparse_csr('TFIDF.npz')
with open('true_y.csv', 'r') as f:
    reader = csv.reader(f)
    y = list(reader)
y = np.asarray(y)[0]
y = list(map(int, y))
y = np.array(y)

nmf = NMF(n_components=300, random_state=42)
X = nmf.fit_transform(X_tfidf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = svm.SVC(C=1000, random_state=42, kernel='linear')
fit_predict_and_plot_roc(clf, X_train, y_train, X_test, y_test, 'Hard_SVM')

nmf = NMF(n_components=50, random_state=42)
X = nmf.fit_transform(X_tfidf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = svm.SVC(C=0.001, random_state=42, kernel='linear')
fit_predict_and_plot_roc(clf, X_train, y_train, X_test, y_test, 'Soft_SVM')

nmf = NMF(n_components=300, random_state=42)
X = nmf.fit_transform(X_tfidf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = LogisticRegression(C=10000000)
fit_predict_and_plot_roc(clf, X_train, y_train, X_test, y_test, 'Logistic')

nmf = NMF(n_components=100, random_state=42)
X = nmf.fit_transform(X_tfidf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()
fit_predict_and_plot_roc(clf, X_train, y_train, X_test, y_test, 'Naive_Bayes')
