# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from datasets import retrieve_dataset
import pandas as pd
from scipy.sparse import hstack
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def _load_22newsgroups(ngram_len, analyzer, dsname):
    cats = ['alt.atheism', 'sci.space']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cats, shuffle=True,
                                          random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(ngram_len, ngram_len), analyzer=analyzer)
    X = vectorizer.fit_transform(newsgroups_train.data)
    y = newsgroups_train.target

    return X, y, dsname, LogisticRegression(random_state=42)


def load_bing():
    vectorizer = TfidfVectorizer(ngram_range=(3, 3), analyzer="word")
    Z = pd.read_csv("BingAdult-TRAIN_14MB_97k-rows.txt", sep="\t", header=None)
    X1 = vectorizer.fit_transform(Z.values[:, 2])
    X2 = vectorizer.fit_transform(Z.values[:, 3])

    X = hstack([X1, X2])
    y = Z.values[:, 1]

    return X, y.astype(int), "bing adult", LogisticRegression(random_state=42)


def load_22newsgroups_trigram():
    return _load_22newsgroups(3, "char", "22 newsgroups char trigrams")


def load_22newsgroups_word():
    return _load_22newsgroups(1, "word", "22 newsgroups word grams")


def load_msx():
    Z = retrieve_dataset('msx_transformed_2226.npz')

    return Z[:, :-2], Z[:, -2].toarray().flatten(), "msx", LinearRegression()
