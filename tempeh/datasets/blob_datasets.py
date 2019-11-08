# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines a class for the Blob datasets."""
import numpy as np

from .base_wrapper import BasePerformanceDatasetWrapper

from .blob_dataset_retriever import create_msx_small, create_msx_big

from tempeh.constants import FeatureType, Tasks, DataTypes, ClassVars, BlobDatasets  # noqa


class BlobPerformanceDatasetWrapper(BasePerformanceDatasetWrapper):
    """Blob Datasets"""
    dataset_map = {BlobDatasets.MSX_SMALL: (create_msx_small, 58083 * [FeatureType.NOMINAL]),
                   BlobDatasets.MSX_BIG: (create_msx_big, 82871 * [FeatureType.NOMINAL])}

    metadata_map = {
        BlobDatasets.MSX_SMALL: (Tasks.MULTICLASS, DataTypes.TABULAR, (2226, 58083)),
        BlobDatasets.MSX_BIG: (Tasks.MULTICLASS, DataTypes.TABULAR, (5550, 82871))
    }

    load_function = None
    feature_type = None

    def __init__(self):
        """Initializes the blob datasets."""
        bunch, target = type(self).load_function()

        super().__init__(bunch, target, nrows=self._size[0], data_t=self._feature_type)
        self._target_names = np.unique(target)
        self._features = list(range(bunch.shape[1]))

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
                                          ClassVars.TASK: task, ClassVars.DATA_TYPE: data_type,
                                          ClassVars.SIZE: size})
