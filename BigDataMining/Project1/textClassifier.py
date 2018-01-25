import os
import tarfile

try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen

from sklearn.datasets import fetch_20newsgroups
'''
URL = ("http://people.csail.mit.edu/jrennie/"
       "20Newsgroups/20news-bydate.tar.gz")

ARCHIVE_NAME = URL.rsplit('/', 1)[1]
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"


if not os.path.exists(TRAIN_FOLDER) or not os.path.exists(TEST_FOLDER):

    if not os.path.exists(ARCHIVE_NAME):
        print("Downloading dataset from %s (14 MB)" % URL)
        opener = urlopen(URL)
        open(ARCHIVE_NAME, 'wb').write(opener.read())

    print("Decompressing %s" % ARCHIVE_NAME)
    tarfile.open(ARCHIVE_NAME, "r:gz").extractall(path='.')
    os.remove(ARCHIVE_NAME)
'''
categories = ['comp.graphics', 'comp.sys.mac.hardware']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

print type(twenty_train)
print twenty_train.keys()