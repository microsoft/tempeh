# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the COMPAS dataset."""
import pandas as pd
import numpy as np

from .base_wrapper import BasePerformanceDatasetWrapper
from tempeh.constants import FeatureType, Tasks, DataTypes, ClassVars, CompasDatasets  # noqa


def compas_data_loader():
    data = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
    # filter similar to https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
    data = data[(data['days_b_screening_arrest'] <= 30) &
                (data['days_b_screening_arrest'] >= -30) &
                (data['is_recid'] != -1) &
                (data['c_charge_degree'] != "O") &
                (data['score_text'] != "N/A")]
    # select relevant columns for machine learning
    data = data[["sex", "age", "race", "juv_fel_count", "decile_score", "juv_misd_count",
                 "juv_other_count", "priors_count", "c_charge_degree", "is_recid",
                 "r_charge_degree", "is_violent_recid", "vr_charge_degree", "decile_score.1",
                 "v_decile_score", "priors_count.1", "two_year_recid"]]
    # map string representation of feature "sex" to 0 for Female and 1 for Male
    data = data.assign(sex=(data["sex"] == "Male") * 1)
    data = pd.get_dummies(data)
    return data


def recover_categorical_encoding_for_compas_race(data):
    return list(map(lambda tuple: "".join(list(tuple)), zip(
        [
            "African-American" if is_column_12_true else "" for is_column_12_true in data[:, 12]
        ],
        [
            "Asian" if is_column_13_true else "" for is_column_13_true in data[:, 13]
        ],
        [
            "Caucasian" if is_column_14_true else "" for is_column_14_true in data[:, 14]
        ],
        [
            "Hispanic" if is_column_15_true else "" for is_column_15_true in data[:, 15]
        ],
        [
            "Native American" if is_column_16_true else "" for is_column_16_true in data[:, 16]
        ],
        [
            "Other" if is_column_17_true else "" for is_column_17_true in data[:, 17]
        ])))


class CompasPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """COMPAS Datasets"""

    dataset_map = {
        CompasDatasets.COMPAS: (compas_data_loader, "two_year_recid",
                                [FeatureType.NOMINAL] + [FeatureType.CONTINUOUS] * 6 +
                                [FeatureType.NOMINAL] * 2 + [FeatureType.CONTINUOUS] * 3 +
                                [FeatureType.NOMINAL] * 28)
    }

    metadata_map = {
        CompasDatasets.COMPAS: (Tasks.BINARY, DataTypes.TABULAR, (6172, 39))
    }

    load_function = None
    feature_type = None
    target_col = None

    def __init__(self, drop_race=True, drop_sex=False):
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

            # race is in columns 13-19 because the super class constructor removes the target
            self.X_train = np.delete(self.X_train, np.s_[12:18], axis=1)
            self.X_test = np.delete(self.X_test, np.s_[12:18], axis=1)
            del[self.features[12:18]]

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
