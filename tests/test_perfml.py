# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from tempeh.perm_configs import datasets, models


def test_true():
    assert True


@pytest.mark.parametrize('Dataset', datasets.values())
@pytest.mark.parametrize('Model', models.values())
def test_fit_predict(Dataset, Model):
    dataset = Dataset()
    model = Model()
    if model.compatible_with_dataset(dataset):
        model.fit(dataset.X_train, dataset.y_train)
        model.predict(dataset.X_test)
