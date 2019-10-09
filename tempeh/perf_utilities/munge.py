# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))  # noqa
from constants import FeatureType  # noqa


def nn(data_c, data_n, algorithm='auto'):
    """ Finds the nearest neighbor of a point

    :param data_c: Contains the neighboring continuous data points
    :type data_c: 2d numpy array of the other continuous data points
    :param data_n: Contains the neighboring nominal data points
    :type data_n: 2d numpy array of the other nominal data points
    :param algorithm: nn algorithm to use
    :type algorithm: str
    """

    # One-hot enconding
    one_hot = pd.get_dummies(pd.DataFrame(data_n))
    data = np.hstack((data_c, one_hot.values))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm=algorithm).fit(data)
    return nbrs.kneighbors(data)[1]


def munge(examples, multiplier, prob, loc_var, data_t, seed=0):
    """ Generates a dataset from the original one

    :param examples: Training examples
    :type examples: 2d numpy array
    :param multiplier: size multiplier
    :type multiplier: int k
    :param prob: probability of swapping values
    :type prob: flt (0 to 1)
    :param loc_var: local variance parameter
    :type loc_var: flt
    :param data_t: Identifies whether or not the attribute is continuous or nominal
    :type data_t: Numpy array of strs
    """

    np.random.seed(seed)
    new_dataset = None
    continuous = [True if x == FeatureType.CONTINUOUS else False for x in data_t]
    nominal = np.logical_not(continuous)
    data_c = examples[:, continuous].astype(float)
    # Scales data linearly from 0 to 1
    norm_data_c = normalize(data_c - np.min(data_c, axis=0), axis=0, norm='max')
    data_n = examples[:, nominal]
    indicies = nn(norm_data_c, data_n)
    for i in range(multiplier):
        T_prime = np.copy(examples)
        # Runs through all the examples in the dataset
        for j in range(examples.shape[0]):
            index = indicies[j, 1] if indicies[j, 0] == j else indicies[j, 0]
            pt1 = T_prime[j, :]
            pt2 = T_prime[index, :]
            # Runs through all features for an example and its nn
            for k in range(len(data_t)):
                # Swaps the two fields with probability prob
                if np.random.ranf() < prob:
                    if data_t[k] == FeatureType.CONTINUOUS:
                        std = abs(float(pt1[k]) - float(pt2[k])) / loc_var
                        temp = float(pt1[k])
                        pt1[k] = np.random.normal(float(pt2[k]), std)
                        pt2[k] = np.random.normal(temp, std)
                    else:
                        temp = pt1[k]
                        pt1[k] = pt2[k]
                        pt2[k] = temp
        # Combines the dataset to the final one
        if new_dataset is None:
            new_dataset = np.copy(T_prime)
        else:
            new_dataset = np.vstack((new_dataset, T_prime))
    return new_dataset

# Sample test
# data_t = ['c','n','c','n']

# data = np.array([[10, 'r', 20, 'g'],
#                  [ 5, 'r', 30, 'b'],
#                  [11, 'w',  5, 'g'],
#                  [10, 'w',  6, 'g']])

# print(munge(data, 1, 0, 1, data_t))
