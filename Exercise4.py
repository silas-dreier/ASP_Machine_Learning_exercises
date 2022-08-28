#!/usr/bin/env python3
# coding: utf-8
# Author:   Silas Dreier <silas.dreier@ifw-kiel.de>
"""Exercise 4"""

import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from string import digits, punctuation
import pickle
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt


CONSTANT1 = Path("./Asp_Machine_Leaning_exercises")

# 1: Completed. Arrived at https://regexone.com/problem/complete?

# 2
# (a)
file_list = sorted(Path('./data/speeches').glob('R0*'))
corpus = []
for file_path in file_list:
    try:
       with open(file_path, encoding="utf8") as f_input:
            corpus.append(f_input.read())
    except UnicodeDecodeError:
        print(file_path)

# (b)
# Stemming, Tokenizing, Stopwords
_stemmer = nltk.snowball.SnowballStemmer("english")
_stopwords = nltk.corpus.stopwords.words("english")
def tokenize_and_stem(text):
    d = {p: "" for p in digits + punctuation}
    text = text.translate(str.maketrans(d))
    return [_stemmer.stem(t) for t in nltk.word_tokenize(text)]
# Vectorizing
tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words=_stopwords, ngram_range=(1,3) )
tfidf_matrix = tfidf.fit_transform(corpus)
# Creating Sparse Matrix
df_count = pd.DataFrame(tfidf_matrix.todense().T,
                        index=tfidf.get_feature_names_out())
# Pickle Data
with open("./output/speech_matrix.pk", 'wb') as handle:
    pickle.dump(df_count, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save Output as csv
df_count.to_csv("./output/terms.csv")

# 3 Speeches II
# a)
pickle_in = open("./output/speech_matrix.pk", "rb")
loaded_matrix = pickle.load(pickle_in)

# b)
array = np.array(loaded_matrix)

y = pdist(array, metric='cosine')
Z = linkage(y, method='complete')
dn = dendrogram(Z, no_labels=True)


# Now we plot the dendrogram for the linkage_array containing the distances
# between clusters
dendrogram(linkage_array)
# Mark the cuts in the tree that signify two or three clusters
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")


# 4 Job Ads
filepath = "./data/Stellenanzeigen.txt"

def parse(filepath):
    data = []
    with open(filepath, 'r') as file:
        line = file.readline()
        while line:
            reg_match = _RegExLib(line)

            if reg_match.newspaper:
                newspaper = reg_match.newspaper.group(1)
            if reg_match.date:
                date = reg_match.date.group(1)
            if reg_march.ad:
                ad = reg_match.ad.group(1)
            dict_of_data = {
                'Newspaper': newspaper,
                'Date' : date,
                'Job Ad': ad
            }
            data.append(dict_of_data)

        line = file.readline()
data = pd.DataFrame(data)



class _RegExLib:
    """Set up regular expressions"""
    _reg_newspaper = re.compile('^(Neue Züricher Zeitung),\d*.\w*\d{4}$|^(Tages-Anzeiger),\d*.\w*\d{4}$')
    _reg_date = re.compile('^(Neue Züricher Zeitung),(\d*.\w*\d{4})$|^(Tages-Anzeiger),(\d*.\w*\d{4})$')
    _reg_ad = re.compile('^.*$')