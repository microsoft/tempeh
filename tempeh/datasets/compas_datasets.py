# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the COMPAS dataset."""
import pandas as pd
import numpy as np

from .base_wrapper import BasePerformanceDatasetWrapper
from tempeh.constants import FeatureType, Tasks, DataTypes, ClassVars, CompasDatasets  # noqa


def compas_data_loader():
    """ Downloads COMPAS data from the propublica GitHub repository.

    :return: pandas.DataFrame with columns 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
       'juv_other_count', 'priors_count', 'two_year_recid', 'age_cat_25 - 45',
       'age_cat_Greater than 45', 'age_cat_Less than 25', 'race_African-American',
       'race_Caucasian', 'c_charge_degree_F', 'c_charge_degree_M'
    """
    data = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")  # noqa: E501
    # filter similar to
    # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]
    # filter out all records except the ones with the most common two races
    data = data[(data['race'] == 'African-American') | (data['race'] == 'Caucasian')]
    # Select relevant columns for machine learning.
    # We explicitly leave in age_cat to allow linear classifiers to be non-linear in age
    data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]]
    # map string representation of feature "sex" to 0 for Female and 1 for Male
    data = data.assign(sex=(data["sex"] == "Male") * 1)
    data = pd.get_dummies(data)
    return data


def recover_categorical_encoding_for_compas_race(data):
    return list(map(lambda tuple: "".join(list(tuple)), zip(
        [
            "African-American" if is_column_true else "" for is_column_true in data[:, 9]
        ],
        [
            "Caucasian" if is_column_true else "" for is_column_true in data[:, 10]
        ])))


class CompasPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """COMPAS Datasets"""

    dataset_map = {
        CompasDatasets.COMPAS: (compas_data_loader, "two_year_recid",
                                [FeatureType.NOMINAL] + [FeatureType.CONTINUOUS] * 5 +
                                [FeatureType.NOMINAL] * 8)
    }

    metadata_map = {
        CompasDatasets.COMPAS: (Tasks.BINARY, DataTypes.TABULAR, (6172, 14))
    }

    load_function = None
    feature_type = None
    target_col = None

    def __init__(self, drop_race=True, drop_sex=True):
        """Initializes the COMPAS dataset """

        bunch = type(self).load_function()
        target = bunch[self.target_col].astype(int)
        bunch.drop(self.target_col, axis=1, inplace=True)
        bunch = bunch.astype(float)

        super().__init__(bunch, target, nrows=self.size[0], data_t=self.feature_type)

        self.features = list(bunch)

        if drop_race:
            self.race_train = recover_categorical_encoding_for_compas_race(self.X_train)
            self.race_test = recover_categorical_encoding_for_compas_race(self.X_test)

            # race is in columns 9-10 because the super class constructor removes the target
            self.X_train = np.delete(self.X_train, np.s_[9:11], axis=1)
            self.X_test = np.delete(self.X_test, np.s_[9:11], axis=1)
            del[self.features[9:11]]

        if drop_sex:
            self.sex_train = self.X_train[:, 0]
            self.sex_test = self.X_test[:, 0]

            self.X_train = np.delete(self.X_train, 0, axis=1)
            self.X_test = np.delete(self.X_test, 0, axis=1)
            del[self.features[0]]

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
