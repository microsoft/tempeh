# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a dataset wrapper for the performance testing framework
to test on different model/dataset permutations.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tempeh.perf_utilities.munge import munge  # noqa
from tempeh.constants import ClassVars


class BasePerformanceDatasetWrapper(object):
    """Base dataset wrapper."""

    task = None
    data_type = None
    size = None

    def __init__(self, dataset, targets, nrows=None, data_t=None):
        """Initialize base dataset wrapper.

        :param dataset: A matrix of feature vector examples
        :type dataset: numpy.array or pandas.DataFrame or pandas.Series
        :param targets: An array of target values
        :type targets: numpy.array or pandas.Series
        :param nrows: Num of rows (optional)
        :type nrows: int
        :param data_t: array specifying continuous or nominal attributes
        :type data_t: array
        """
        self._features = None
        self._target_names = None
        self._data_t = data_t

        self._dataset = dataset
        self._targets = targets

        dataset_is_df = isinstance(dataset, pd.DataFrame)
        dataset_is_series = isinstance(dataset, pd.Series)
        if dataset_is_df or dataset_is_series:
            self._features = dataset.columns.values.tolist()
            self._dataset = dataset.values

        targets_is_df = isinstance(targets, pd.DataFrame)
        targets_is_series = isinstance(targets, pd.Series)
        if targets_is_df or targets_is_series:
            self._targets = targets.values

        self._nrows = nrows

        self._sample()
        self._training_split()

    def _sample(self, prob=.8, local_var=1):
        """Samples up or down depending on self._nrows."""
        if self._nrows is not None and self._nrows != self._dataset.shape[0]:
            # Set random seed to insure consistency across runs
            np.random.seed(219)

            # Creates random indices for sampling
            # We need to replace if self._nrows > self._dataset.shape[0]
            size = abs(self._nrows - self._dataset.shape[0])
            index = np.random.choice(self._dataset.shape[0], size=size,
                                     replace=self._nrows > self._dataset.shape[0])

            if self._nrows > self._dataset.shape[0]:
                T = np.hstack((self._dataset, np.array([self._targets]).T))
                # Combining the target column to actual dataset
                # Values for probability and local variance taken from github
                new_data = munge(T, self._nrows // T.shape[0],
                                 prob, local_var, self._data_t)
                # Produces a new data set with size equal to an integer multiple of the original
                self._dataset = new_data[:, :-1]
                self._targets = new_data[:, -1:]
            else:
                self._dataset = np.delete(self._dataset, index, 0)
                self._targets = np.delete(self._targets, index, 0)
        else:
            self._nrows = self._dataset.shape[0]

    def _training_split(self):
        """Creates a training split."""
        self._X_train, self._X_test, self._y_train, self._y_test = \
            train_test_split(self._dataset, self._targets, test_size=0.33, random_state=123)

    def get_X(self, format=np.ndarray):
        """ Returns the features of both the training and the test data.

        :param format: either numpy.ndarray or pandas.DataFrame
        :type format: type
        """
        if format == np.ndarray:
            return self._X_train, self._X_test
        elif format == pd.DataFrame:
            X_train = pd.DataFrame(self._X_train, columns=self._features)
            X_test = pd.DataFrame(self._X_test, columns=self._features)
            return X_train, X_test
        else:
            raise ValueError("Only numpy.ndarray and pandas.DataFrame are currently supported.")

    def get_y(self, format=np.ndarray):
        """ Returns the targets of both the training and the test data.

        :param format: either numpy.ndarray or pandas.Series
        :type format: type
        """
        if format == np.ndarray:
            return self._y_train, self._y_test
        elif format == pd.Series:
            kwargs = {}
            if hasattr(self, ClassVars.TARGET_COL):
                kwargs["name"] = self._target_col
            y_train = pd.Series(self._y_train.squeeze(), **kwargs)
            y_test = pd.Series(self._y_test.squeeze(), **kwargs)
            return y_train, y_test
        else:
            raise ValueError("Only numpy.ndarray and pandas.Series are currently supported.")

    def get_sensitive_features(self, names, format=np.ndarray):
        """ Returns the sensitive features of both the training and the test data. If the
        sensitive features don't exist under the specified name a ValueError is returned.

        :param name: a string describing the sensitive feature, e.g. gender, race, or age
        :type name: str
        :param format: either numpy.ndarray or pandas.Series
        :type format: type
        """
        if type(names) == str:
            # keeping the version with single string for backward compatibility
            names = [names]

        if format not in [np.ndarray, pd.Series, pd.DataFrame]:
            raise ValueError("Only numpy.ndarray, pandas.DataFrame and pandas.Series are "
                             "currently supported.")

        if len(names) > 1 and format not in [pd.DataFrame, np.ndarray]:
            raise Exception("Multiple sensitive features were requested, so the only supported "
                            "formats are pandas.DataFrame and numpy.ndarray.")

        sensitive_features_train_names = {}
        sensitive_features_test_names = {}
        sensitive_features_train = {}
        sensitive_features_test = {}
        for name in names:
            sensitive_features_train_names[name] = "_{}_train".format(name)
            sensitive_features_test_names[name] = "_{}_test".format(name)
            if not hasattr(self, sensitive_features_train_names[name]) or \
                    not hasattr(self, sensitive_features_test_names[name]):
                raise ValueError("This dataset does not have sensitive features with the name {}."
                                 .format(name))

            sensitive_features_train[name] = getattr(self, sensitive_features_train_names[name]).squeeze()  # noqa: E501
            sensitive_features_test[name] = getattr(self, sensitive_features_test_names[name]).squeeze()  # noqa: E501

        if format == pd.Series:
            # we can assume that there's only one feature in this case
            return pd.Series(sensitive_features_train, name=name), \
                pd.Series(sensitive_features_test, name=name)

        # for multiple features we need to merge the individual columns together before returning.
        return _merge_columns_from_dict_into_matrix(sensitive_features_train,
                                                    output_format=format), \
            _merge_columns_from_dict_into_matrix(sensitive_features_test,
                                                 output_format=format)


def _merge_columns_from_dict_into_matrix(column_dict, output_format=np.ndarray):
    array = np.concatenate(column_dict.values(), axis=1)

    if output_format == np.ndarray:
        return array

    if output_format != pd.DataFrame:
        raise Exception("Merging columns into a matrix is only supported with output_format "
                        "numpy.ndarray and pandas.DataFrame.")

    return pd.DataFrame(array, columns=column_dict.keys())
