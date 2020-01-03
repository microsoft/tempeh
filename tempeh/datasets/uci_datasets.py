# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the UCI datasets."""
import numpy as np

from .base_wrapper import BasePerformanceDatasetWrapper
from .uci_dataset_cleaner import bank_data_parser, bank_data_additional_parser, car_eval_parser, \
    adult_data_parser, communities_parser
from tempeh.constants import FeatureType, Tasks, DataTypes, UCIDatasets, ClassVars  # noqa


class UCIPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """UCI Datasets"""

    dataset_map = {
        UCIDatasets.BANK: (bank_data_parser, "y",
                           [FeatureType.CONTINUOUS] * 10 + [FeatureType.NOMINAL] * 39),
        UCIDatasets.BANK_ADD: (bank_data_additional_parser, "y",
                               [FeatureType.CONTINUOUS] * 10 + [FeatureType.NOMINAL] * 54),
        UCIDatasets.CAR: (car_eval_parser, "CAR", [FeatureType.NOMINAL] * 22),
        UCIDatasets.ADULT: (adult_data_parser, "y", [FeatureType.CONTINUOUS] +
                            [FeatureType.NOMINAL] * 7 + [FeatureType.CONTINUOUS] * 3 +
                            [FeatureType.NOMINAL]),
        UCIDatasets.COMMUNITIES: (communities_parser, 'y', [FeatureType.CONTINUOUS] * 102)
    }

    metadata_map = {
        UCIDatasets.BANK: (Tasks.BINARY, DataTypes.TABULAR, (45211, 48)),
        UCIDatasets.BANK_ADD: (Tasks.BINARY, DataTypes.TABULAR, (41188, 63)),
        UCIDatasets.CAR: (Tasks.MULTICLASS, DataTypes.TABULAR, (1728, 21)),
        UCIDatasets.ADULT: (Tasks.BINARY, DataTypes.TABULAR, (32561, 13)),
        UCIDatasets.COMMUNITIES: (Tasks.REGRESSION, DataTypes.TABULAR, (1994, 102)),
    }

    load_function = None
    feature_type = None
    target_col = None

    def __init__(self):
        """Initializes the UCI dataset """

        bunch = type(self).load_function()

        if np.isnan(bunch[self._target_col]).sum() > 0:
            raise Exception("NaN in array")
        if self.task != Tasks.REGRESSION:
            target = bunch[self._target_col].astype(int)
        else:
            target = bunch[self._target_col]
        bunch.drop(self._target_col, axis=1, inplace=True)
        bunch = bunch.astype(float)

        super().__init__(bunch, target, nrows=self._size[0], data_t=self._feature_type)
        self._features = list(bunch)
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

        class_name = "".join((x.title() for x in name.split("-"))) + "PerformanceDatasetWrapper"
        return type(class_name, (cls, ), {ClassVars.LOAD_FUNCTION: load_function,
                                          ClassVars.FEATURE_TYPE: feature_type,
                                          ClassVars.TASK: task,
                                          ClassVars.DATA_TYPE: data_type,
                                          ClassVars.SIZE: size, ClassVars.TARGET_COL: target_col})
