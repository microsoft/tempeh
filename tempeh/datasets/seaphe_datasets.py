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


def lawschool_data_loader():
    """ Downloads SEAPHE lawschool data from the SEAPHE webpage.
    For more information refer to http://www.seaphe.org/databases.php

    :return: pandas.DataFrame with columns 
    """
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

        # drop NaN records for pass_bar and features
        data = data[(data['pass_bar'] == 1) | (data['pass_bar'] == 0)]
        data = data[(np.isnan(data["lsat"]) != True) & (np.isnan(data['ugpa']) != True)]

        # Select relevant columns for machine learning.
        # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
        # TODO: add another dataset where we expose 'zfygpa' for regression
        # TODO: add another version with 'gender'
        # TODO: consider using 'fam_inc', 'age', 'parttime', 'dropout'
        data = data[['white', 'black', 'lsat', 'ugpa', 'pass_bar']]

    return data


def recover_categorical_encoding_for_compas_race(data):
    return list(map(lambda tuple: "".join(list(tuple)), zip(
        [
            "white" if is_column_true else "" for is_column_true in data[:, 0]
        ],
        [
            "black" if is_column_true else "" for is_column_true in data[:, 1]
        ])))


class SEAPHEPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """SEAPHE Datasets"""

    dataset_map = {
        SEAPHEDatasets.LAWSCHOOL_PASSBAR: (lawschool_data_loader, "pass_bar",
                                           [FeatureType.NOMINAL] * 2 +
                                           [FeatureType.CONTINUOUS] * 2 +
                                           [FeatureType.NOMINAL])
    }

    metadata_map = {
        SEAPHEDatasets.LAWSCHOOL_PASSBAR: (Tasks.BINARY, DataTypes.TABULAR, (23103, 5))
    }

    load_function = None
    feature_type = None
    target_col = None

    def __init__(self, drop_race=True):
        """Initializes the SEAPHE dataset """

        bunch = type(self).load_function()
        target = bunch[self.target_col].astype(int)
        bunch.drop(self.target_col, axis=1, inplace=True)
        bunch = bunch.astype(float)

        super().__init__(bunch, target, nrows=self.size[0], data_t=self.feature_type)

        self.features = list(bunch)

        if drop_race:
            self.race_train = recover_categorical_encoding_for_compas_race(self.X_train)
            self.race_test = recover_categorical_encoding_for_compas_race(self.X_test)

            # race is in columns 0-1 because the super class constructor removes the target
            self.X_train = np.delete(self.X_train, np.s_[0:2], axis=1)
            self.X_test = np.delete(self.X_test, np.s_[0:2], axis=1)
            del[self.features[0:2]]

        self.target_names = np.unique(target)

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
