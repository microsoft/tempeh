# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the SEAPHE datasets.
http://www.seaphe.org/databases.php
"""
import os
import pandas as pd
import numpy as np
import tempfile
import requests
import zipfile

from .base_wrapper import BasePerformanceDatasetWrapper
from tempeh.constants import FeatureType, Tasks, DataTypes, ClassVars, SEAPHEDatasets  # noqa


def load_lawschool_data(target):
    """ Downloads SEAPHE lawschool data from the SEAPHE webpage.
    For more information refer to http://www.seaphe.org/databases.php

    :param target: the name of the target variable, either pass_bar or zfygpa
    :type target: str
    :return: pandas.DataFrame with columns
    """
    if target not in ['pass_bar', 'zfygpa']:
        raise ValueError("Only pass_bar and zfygpa are supported targets.")

    with tempfile.TemporaryDirectory() as temp_dir:
        response = requests.get("http://www.seaphe.org/databases/LSAC/LSAC_SAS.zip")
        temp_file_name = os.path.join(temp_dir, "LSAC_SAS.zip")
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(response.content)
        with zipfile.ZipFile(temp_file_name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        data = pd.read_sas(os.path.join(temp_dir, "lsac.sas7bdat"))

        # data contains 'sex', 'gender', and 'male' which are all identical except for the type;
        # map string representation of feature "sex" to 0 for Female and 1 for Male
        data = data.assign(gender=(data["gender"] == b"male") * 1)

        # filter out all records except the ones with the most common two races
        data = data.assign(white=(data["race1"] == b"white") * 1)
        data = data.assign(black=(data["race1"] == b"black") * 1)
        data = data[(data['white'] == 1) | (data['black'] == 1)]

        # encode dropout as 0/1
        data = data.assign(dropout=(data["Dropout"] == b"YES") * 1)

        if target == 'pass_bar':
            # drop NaN records for pass_bar
            data = data[(data['pass_bar'] == 1) | (data['pass_bar'] == 0)]
        elif target == 'zfygpa':
            # drop NaN records for zfygpa
            data = data[np.isfinite(data['zfygpa'])]
        
        # drop NaN records for features
        data = data[np.isfinite(data["lsat"]) & np.isfinite(data['ugpa'])]

        # Select relevant columns for machine learning.
        # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
        # TODO: consider using 'fam_inc', 'age', 'parttime', 'dropout'
        data = data[['white', 'black', 'gender', 'lsat', 'ugpa', target]]

    return data


def recover_categorical_encoding_for_compas_race(data, starting_column=0):
    return np.array(list(map(lambda tuple: "".join(list(tuple)), zip(
        [
            "white" if is_column_true else "" for is_column_true in data[:, starting_column]
        ],
        [
            "black" if is_column_true else "" for is_column_true in data[:, starting_column + 1]
        ]))))


def recover_categorical_encoding_for_compas_gender(data, starting_column=2):
    return np.array(
        [
            "male" if is_column_true else "female" for is_column_true in data[:, starting_column]
        ])


class SEAPHEPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """SEAPHE Datasets"""

    dataset_map = {
        SEAPHEDatasets.LAWSCHOOL_PASSBAR: (lambda: load_lawschool_data('pass_bar'),
                                           "pass_bar",
                                           [FeatureType.NOMINAL] * 3 +
                                           [FeatureType.CONTINUOUS] * 2 +
                                           [FeatureType.NOMINAL]),
        SEAPHEDatasets.LAWSCHOOL_GPA: (lambda: load_lawschool_data('zfygpa'),
                                       "zfygpa",
                                       [FeatureType.NOMINAL] * 3 +
                                       [FeatureType.CONTINUOUS] * 3),
    }

    metadata_map = {
        SEAPHEDatasets.LAWSCHOOL_PASSBAR: (Tasks.BINARY, DataTypes.TABULAR, (20460, 6)),
        SEAPHEDatasets.LAWSCHOOL_GPA: (Tasks.REGRESSION, DataTypes.TABULAR, (22342, 6))
    }

    load_function = None
    feature_type = None
    target_col = None

    def __init__(self, drop_race=True, drop_gender=True):
        """Initializes the SEAPHE dataset """

        bunch = type(self).load_function()
        target = bunch[self._target_col]
        if self._target_col == "pass_bar":
            target = target.astype(int)
        bunch.drop(self._target_col, axis=1, inplace=True)
        bunch = bunch.astype(float)

        super().__init__(bunch, target, nrows=self._size[0], data_t=self._feature_type)

        self._features = list(bunch)

        if drop_race:
            self._race_train = recover_categorical_encoding_for_compas_race(self._X_train)
            self._race_test = recover_categorical_encoding_for_compas_race(self._X_test)

            # race is in columns 0-1
            self._X_train = np.delete(self._X_train, np.s_[0:2], axis=1)
            self._X_test = np.delete(self._X_test, np.s_[0:2], axis=1)
            del[self._features[0:2]]
        
        if drop_gender:
            starting_column = 0 if drop_race else 2
            self._gender_train = recover_categorical_encoding_for_compas_gender(self._X_train, starting_column)
            self._gender_test = recover_categorical_encoding_for_compas_gender(self._X_test, starting_column)
            # gender is in column 2 (only binary gender data available),
            # unless race has been dropped already, then column 0
            self._X_train = np.delete(self._X_train, np.s_[starting_column], axis=1)
            self._X_test = np.delete(self._X_test, np.s_[starting_column], axis=1)
            del[self._features[0 if drop_race else 2]]

        self._target_names = np.unique(target)

    @classmethod
    def generate_dataset_class(cls, name, nrows=None):
        """Generate a dataset class.

        :param name: the name of the dataset
        :type name: str
        :param nrows: number of rows to resize the dataset to
        :type nrows: int
        :rtype: cls
        """
        load_function, target_col, feature_type = cls.dataset_map[name]
        task, data_type, size = cls.metadata_map[name]

        if nrows is not None:
            size = (nrows, size[1])

        class_name = name.title() + "PerformanceDatasetWrapper"
        return type(class_name, (cls, ), {ClassVars.LOAD_FUNCTION: load_function,
                                          ClassVars.FEATURE_TYPE: feature_type,
                                          ClassVars.TASK: task,
                                          ClassVars.DATA_TYPE: data_type,
                                          ClassVars.SIZE: size,
                                          ClassVars.TARGET_COL: target_col})
