# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from .conftest import get_selected_datasets, get_selected_models


@pytest.mark.parametrize('Dataset', get_selected_datasets())
@pytest.mark.parametrize('Model', get_selected_models())
def test_fit_predict(Dataset, Model):
    dataset = Dataset()
    model = Model()
    if model.compatible_with_dataset(dataset):
        model.fit(dataset.X_train, dataset.y_train)
        model.predict(dataset.X_test)
