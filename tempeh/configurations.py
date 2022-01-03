# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Holds all the datasets and models that are automatically enqueued."""

from tempeh.datasets import SKLearnPerformanceDatasetWrapper, UCIPerformanceDatasetWrapper, \
    BlobPerformanceDatasetWrapper, CompasPerformanceDatasetWrapper, \
    SEAPHEPerformanceDatasetWrapper
from tempeh.models import RBMSVMModelWrapper, LinearSVMModelWrapper, LogisticModelWrapper, \
    RidgeModelWrapper, DecisionTreeClassifierWrapper, RandomForestClassifierWrapper, \
    RandomForestRegressorWrapper, PytorchMulticlassClassifierWrapper, \
    PytorchBinaryClassifierWrapper, PytorchRegressionWrapper, XGBoostClassifierWrapper, \
    XGBoostRegressorWrapper, LightGBMClassifierWrapper, LightGBMRegressorWrapper, \
    KerasMulticlassClassifierWrapper, KerasBinaryClassifierWrapper, KerasRegressionWrapper
from tempeh.constants import DatasetSizes

# datasets dictionary
datasets = {
    'boston_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('boston'),
    'iris_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('iris'),
    'diabetes_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('diabetes'),
    'digits_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('digits'),
    'wine_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('wine'),
    'cancer_sklearn': SKLearnPerformanceDatasetWrapper.generate_dataset_class('cancer'),
    'bank_uci': UCIPerformanceDatasetWrapper.generate_dataset_class('bank'),
    'bank_add_uci': UCIPerformanceDatasetWrapper.generate_dataset_class('bank-additional'),
    'car_uci': UCIPerformanceDatasetWrapper.generate_dataset_class('car-eval'),
    'adult_uci': UCIPerformanceDatasetWrapper.generate_dataset_class('adult'),
    'communities_uci': UCIPerformanceDatasetWrapper.generate_dataset_class('communities'),
    'msx_small_blob': BlobPerformanceDatasetWrapper.generate_dataset_class('msx_small'),
    'msx_big_blob': BlobPerformanceDatasetWrapper.generate_dataset_class('msx_big'),
    'medium_cancer': SKLearnPerformanceDatasetWrapper.generate_dataset_class('cancer', nrows=DatasetSizes.MEDIUM),
    'medium_iris': SKLearnPerformanceDatasetWrapper.generate_dataset_class('iris', nrows=DatasetSizes.MEDIUM),
    'medium_wine': SKLearnPerformanceDatasetWrapper.generate_dataset_class('wine', nrows=DatasetSizes.MEDIUM),
    'medium_boston': SKLearnPerformanceDatasetWrapper.generate_dataset_class('boston', nrows=DatasetSizes.MEDIUM),
    'big_cancer': SKLearnPerformanceDatasetWrapper.generate_dataset_class('cancer', nrows=DatasetSizes.BIG),
    'big_iris': SKLearnPerformanceDatasetWrapper.generate_dataset_class('iris', nrows=DatasetSizes.BIG),
    'big_wine': SKLearnPerformanceDatasetWrapper.generate_dataset_class('wine', nrows=DatasetSizes.BIG),
    'big_boston': SKLearnPerformanceDatasetWrapper.generate_dataset_class('boston', nrows=DatasetSizes.BIG),
    'giant_cancer': SKLearnPerformanceDatasetWrapper.generate_dataset_class('cancer', nrows=DatasetSizes.GIANT),
    'giant_iris': SKLearnPerformanceDatasetWrapper.generate_dataset_class('iris', nrows=DatasetSizes.GIANT),
    'giant_wine': SKLearnPerformanceDatasetWrapper.generate_dataset_class('wine', nrows=DatasetSizes.GIANT),
    'giant_boston': SKLearnPerformanceDatasetWrapper.generate_dataset_class('boston', nrows=DatasetSizes.GIANT),
    'compas': CompasPerformanceDatasetWrapper.generate_dataset_class('compas'),
    'lawschool_passbar': SEAPHEPerformanceDatasetWrapper.generate_dataset_class('lawschool_passbar'),
    'lawschool_gpa': SEAPHEPerformanceDatasetWrapper.generate_dataset_class('lawschool_gpa')
}

# models dictionary
models = {
    'rbm_svm': RBMSVMModelWrapper,
    'linear_svm': LinearSVMModelWrapper,
    'logistic': LogisticModelWrapper,
    'ridge': RidgeModelWrapper,
    'decision_tree_classifier': DecisionTreeClassifierWrapper,
    'random_forest_classifier': RandomForestClassifierWrapper,
    'random_forest_regressor': RandomForestRegressorWrapper,
    'pytorch_multiclass_classifier': PytorchMulticlassClassifierWrapper,
    'pytorch_binary_classifier': PytorchBinaryClassifierWrapper,
    'pytorch_regressor': PytorchRegressionWrapper,
    'xgboost_classifier': XGBoostClassifierWrapper,
    'xgboost_regressor': XGBoostRegressorWrapper,
    'lightgbm_classifier': LightGBMClassifierWrapper,
    'lightgbm_regressor': LightGBMRegressorWrapper,
    'keras_multiclass_classifier': KerasMulticlassClassifierWrapper,
    'keras_binary_classifier': KerasBinaryClassifierWrapper,
    'keras_regressor': KerasRegressionWrapper,
}
