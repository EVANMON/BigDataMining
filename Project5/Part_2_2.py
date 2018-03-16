import json
import numpy as np
import matplotlib as plt
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

colnames = ['text', 'locat']
data = pd.read_csv('filtered_tweets.csv', names=colnames)
X = data.text.tolist()
y = data.locat.tolist()
X = [str(x) for x in X]
X = list(map(lambda s: s.strip(), X))
# stop words
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

# stemmer
stemmer = nltk.stem.SnowballStemmer('english')

# lemmatizer
wnl = nltk.wordnet.WordNetLemmatizer()
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'
def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
            for word, tag in pos_tag(list_word)]

analyzer = CountVectorizer().build_analyzer()
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())
def stem_all(doc):
    return (word for word in lemmatize_sent((stemmer.stem(w) for w in analyzer(doc))) if word not in combined_stopwords and not word.isdigit())

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

count_vect = CountVectorizer(min_df=2, analyzer=stem_rmv_punc)
X_counts = count_vect.fit_transform(X)
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

save_sparse_csr('TFIDF', X_tfidf)
with open('true_y.csv', 'w') as f:
    writer = csv.writer(f, dialect='excel')
    writer.writerow(y)

'''

#========= LSI ============
svd = TruncatedSVD(n_components=100, random_state=42)
X_lsi = svd.fit_transform(X_tfidf)

with open('LSI_matrix.csv', 'w') as f:
    writer = csv.writer(f, dialect='excel')
    [writer.writerow(r) for r in X_lsi]

#========= NMF =============
nmf = NMF(n_components=50, init='random', random_state=42)
X_nmf = nmf.fit_transform(X_tfidf)
with open('NMF_matrix.csv', 'w') as f:
    writer = csv.writer(f, dialect='excel')
    [writer.writerow(r) for r in X_nmf]
'''

