[![Build Status](https://img.shields.io/azure-devops/build/responsibleai/tempeh/19/master?failed_label=bad&passed_label=good&label=GatedCheckin%3ADev)](https://dev.azure.com/responsibleai/tempeh/_build/latest?definitionId=19&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![pypi badge](https://img.shields.io/badge/pypi-0.1.6-blue)


# tempeh

tempeh is a framework to

**TE**st

**M**achine learning

**PE**rformance

ex**H**austively

which includes tracking memory usage and run time. This is particularly useful as a pluggable tool for your repository's performance tests. Typically, people want to run them periodically over various datasets and/or with a number of models to catch regressions with respect to run time or memory consumption. This should be as easy as

```python
import pytest
from time import time
from tempeh.configurations import datasets, models

@pytest.mark.parametrize('Dataset', datasets.values())
@pytest.mark.parametrize('Model', models.values())
def test_fit_predict_regression(Dataset, Model):
    dataset = Dataset()
    model = Model()
    max_execution_time = get_max_execution_time(dataset, model)
    if model.compatible_with_dataset(dataset):
        start_time = time()
        model.fit(dataset.X_train, dataset.y_train)
        model.predict(dataset.X_test)
        duration = time() - start_time

        assert duration < max_execution_time
```

## Installation

tempeh depends on various packages to provide models, including `tensorflow`, `torch`, `xgboost`, `lightgbm`. To install a release version of `tempeh` just run

```python
pip install tempeh
```

<details>
<summary>
<strong>
<em>
Common issues
</em>
</strong>
</summary>

- If you're using a 32-bit Python version you might need to switch to a 64-bit Python version first to successfully install tensorflow.
- If the installation of `torch` fails try using the recommendation from the [pytorch website](https://pytorch.org/get-started/locally/) for stable versions without CUDA for your python version on your operating system.

</details>

## Structure

### Datasets

Datasets (located in the `datasets/` directory) encapsulate different datasets used for testing.

#### To add a new one

+ Create a python file in the `datasets/` directory with naming convention `[name]_datasets.py`
+ Subclass `BasePerformanceDatasetWrapper`. The naming convention is `[dataset_name]PerformanceDatasetWrapper`
+ In `__init__` load the dataset and call `super().__init__(data, targets, size)`
+ Add the class to `__init__.py`
+ Make sure the class contains class variables `task`, `data_type`, `size`
+ Add an entry to the `datasets` dictionary in `configurations.py`.

### Models

Models (`models/` directory) wrap different machine learning models.

#### To add a new one

+ Create a python file in the `models/` directory with naming convention `[name]_model.py`
+ Subclass `BaseModelWrapper` and name the class `[name]ModelWrapper`
+ In `__init__` train the model; we expect format `__init__(self, ...)`
+ Models must contain `tasks` and `algorithm`
+ Add an entry to the `models` dictionary in `configurations.py`.


## Maintainers

In alphabetical order:

- [Eduardo de Leon](https://github.com/eedeleon)
- [Ilya Matiach](https://github.com/imatiach-msft)
- [Roman Lutz](https://github.com/romanlutz)


# Contributing

To contribute please check our [Contributing Guide](CONTRIBUTING.md).

# Issues

## Regular (non-Security) Issues
Please submit a report through [Github issues](https://github.com/microsoft/tempeh/issues). A maintainer will respond within a reasonable period of time to handle the issue as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


## Reporting Security Issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
