# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Development Process

Development happens against the `master` branch following the [GitHub flow model](https://guides.github.com/introduction/flow/). Contributors create feature branches off of `master`, and their pull requests should target the `master` branch. Maintainers will review pull requests within two business days.

Pull requests against `master` trigger automated pipelines that are run through Azure DevOps. Additional test suites are run periodically. When adding new code paths or features tests are a requirement to complete a pull request. They should be added in the `tests` directory.

### Investigating automated test failures

For every pull request to `master` with automated tests you can check the logs of the tests to find the root cause of failures. Our tests currently run through Azure Pipelines with steps for setup, test suites, and teardown. The `Checks` view of a pull request contains a link to the [Azure Pipelines Page](dev.azure.com/responsibleai/tempeh/_build/results). All the steps are represented in the Azure Pipelines page, and you can see logs by clicking on a specific step. If you encounter problems with this workflow please reach out through the `Issues`.


# Publishing to Pypi

Make sure to run
```
git clean -xdf
```
before running the commands below or you might run into issues with reuploading an existing version.

```
python setup.py sdist bdist_wheel
python -m twine upload  dist/* --verbose
```
