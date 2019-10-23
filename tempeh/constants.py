# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Defines constants for use in the performance testing framework."""


class DataTypes:
    """Data types"""

    TABULAR = 'tabular'
    IMAGE = 'image'
    TEXT = 'text'


class Algorithms:
    """Algorithms"""

    SVM = 'svm'
    LOGISTIC = 'logistic'
    DEEP = 'deep'
    TREE = 'tree'
    RIDGE = 'ridge'


class Tasks:
    """Machine learning tasks"""

    BINARY = 'binary'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'


class FeatureType:
    """Feature types"""

    CONTINUOUS = 'c'
    NOMINAL = 'n'


class Metrics:
    """Metrics"""

    PEAK_MEMORY = 'peak memory'
    MEMORY_USAGE = 'memory usage'
    EXECUTION_TIME = 'execution time'
    FEATURES = 'features'
    ROWS = 'rows'
    SHAP_NDCG = 'shap ndcg'
    TRUE_NDCG = 'true ndcg'
    DATASET = 'dataset'
    EXPLAINER = 'explainer'
    MODEL = 'model'
    AGREEMENT = 'agreement explainer'
    CELLS = 'cells'


class Timing:
    """Timing constants."""

    TIMEOUT = 256


class ModelParams:
    """Parameters used in the model"""

    RANDOM_STATE = 'random_state'


class ClassVars:
    """Dynamically-set class variables."""

    LOAD_FUNCTION = 'load_function'
    TASK = 'task'
    FEATURE_TYPE = 'feature_type'
    DATA_TYPE = 'data_type'
    SIZE = 'size'
    TARGET_COL = 'target_col'
    EXPLAINER_FUNC = 'explainer_func'
    LIMIT = 'limit'
    DATA_TYPES = 'data_types'
    ALGORITHMS = 'algorithms'
    TASKS = 'tasks'
    EXPLAIN_PARAMS = 'explain_params'


class SKLearnDatasets:
    BOSTON = 'boston'
    IRIS = 'iris'
    DIABETES = 'diabetes'
    DIGITS = 'digits'
    WINE = 'wine'
    CANCER = 'cancer'


class UCIDatasets:
    CAR = 'car-eval'
    BANK = 'bank'
    BANK_ADD = 'bank-additional'
    ADULT = 'adult'


class BlobDatasets:
    MSX_SMALL = 'msx_small'
    MSX_BIG = 'msx_big'


class CompasDatasets:
    COMPAS = "compas"


class DatasetSizes:
    MEDIUM = 750000
    BIG = 1500000
    GIANT = 5000000


class LightGBMParams:
    BOOSTING_TYPE = 'boosting_type'
    BAGGING_FRACTION = 'bagging_fraction'
    DEFAULT_BAGGING_FRACTION = 0.33
    BAGGING_FREQ = 'bagging_freq'
    DEFAULT_BAGGING_FREQ = 5
    RF = 'rf'
    DART = 'dart'
    GOSS = 'goss'


class DatasetConstants(object):
    """Dataset related constants."""
    CATEGORICAL = 'categorical'
    CLASSES = 'classes'
    FEATURES = 'features'
    NUMERIC = 'numeric'
    X_TEST = 'x_test'
    X_TRAIN = 'x_train'
    Y_TEST = 'y_test'
    Y_TRAIN = 'y_train'
