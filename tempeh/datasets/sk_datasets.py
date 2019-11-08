# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the scikit datasets."""

from sklearn import datasets
from .base_wrapper import BasePerformanceDatasetWrapper

from tempeh.constants import FeatureType, Tasks, DataTypes, SKLearnDatasets, ClassVars  # noqa


class SKLearnPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """sklearn Datasets"""

    dataset_map = {
        SKLearnDatasets.BOSTON: (datasets.load_boston, [FeatureType.CONTINUOUS] * 3 + [FeatureType.NOMINAL] + [FeatureType.CONTINUOUS] * 10),  # noqa: E501
        SKLearnDatasets.IRIS: (datasets.load_iris, [FeatureType.CONTINUOUS] * 4 + [FeatureType.NOMINAL]),  # noqa: E501
        SKLearnDatasets.DIABETES: (datasets.load_diabetes, [FeatureType.CONTINUOUS] * 11),
        SKLearnDatasets.DIGITS: (datasets.load_digits, [FeatureType.CONTINUOUS] * 64 + [FeatureType.NOMINAL]),  # noqa: E501
        SKLearnDatasets.WINE: (datasets.load_wine, [FeatureType.CONTINUOUS] * 13 + [FeatureType.NOMINAL]),  # noqa: E501
        SKLearnDatasets.CANCER: (datasets.load_breast_cancer, [FeatureType.CONTINUOUS] * 30 + [FeatureType.NOMINAL])  # noqa: E501
    }
    metadata_map = {
        SKLearnDatasets.BOSTON: (Tasks.REGRESSION, DataTypes.TABULAR, (506, 13)),
        SKLearnDatasets.IRIS: (Tasks.MULTICLASS, DataTypes.TABULAR, (150, 4)),
        SKLearnDatasets.DIABETES: (Tasks.REGRESSION, DataTypes.TABULAR, (442, 10)),
        SKLearnDatasets.DIGITS: (Tasks.MULTICLASS, DataTypes.IMAGE, (1797, 64)),
        SKLearnDatasets.WINE: (Tasks.MULTICLASS, DataTypes.TABULAR, (178, 13)),
        SKLearnDatasets.CANCER: (Tasks.BINARY, DataTypes.TABULAR, (569, 30))
    }

    load_function = None
    feature_type = None

    def __init__(self):
        """Initializes an sklearn dataset """

        bunch = type(self).load_function()

        super().__init__(bunch.data, bunch.target, nrows=self._size[0], data_t=self._feature_type)

        if "feature_names" in bunch:
            self._features = bunch.feature_names
        if "target_names" in bunch:
            self._target_names = bunch.target_names

    @classmethod
    def generate_dataset_class(cls, name, nrows=None):
        """Generate a dataset class.

        :param name: the name of the dataset
        :type name: str
        :param nrows: number of rows to resize the dataset to
        :type nrows: int
        :rtype: cls
        """
        load_function, feature_type = cls.dataset_map[name]
        task, data_type, size = cls.metadata_map[name]

        if nrows is not None:
            size = (nrows, size[1])

        class_name = name.title() + "PerformanceDatasetWrapper"
        return type(class_name, (cls, ), {ClassVars.LOAD_FUNCTION: load_function,
                                          ClassVars.FEATURE_TYPE: feature_type,
                                          ClassVars.TASK: task,
                                          ClassVars.DATA_TYPE: data_type,
                                          ClassVars.SIZE: size})
