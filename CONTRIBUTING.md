# Contributing to TT-BUDA-DEMOS

Thank you for your interest in this project.
We want to make contributing to this project as easy and transparent as possible.

If you are interested in making a contribution, then please familiarize
yourself with our technical [contribution standards](#contribution-standards) as set forth in this guide.

Next, please request appropriate write permissions by [opening an
issue](https://github.com/tenstorrent/tt-buda/issues/new/choose) for
GitHub permissions.

All contributions require:

- an issue
  - Your issue should be filed under an appropriate project. Please file a
    feature support request or bug report under Issues to get help with finding
    an appropriate project to get a maintainer's attention.
- a pull request (PR).
- Your PR must be approved by appropriate reviewers.





## Setting up environment

Install all dependencies from [requriements-dev.txt](requirements-dev.txt) and install pre-commit hooks in a Python environment with PyBuda installed.

```bash
cd model_demos
pip install -r requirements-dev.txt
pre-commit install
```

## Developing model_demos

## Adding models

Contribute to the model demos by include Python script files under the respective model type directories in `model_demos`. If it's a new model architecture, please create a directory for that model. The script should be self-contained and include pre/post-processing steps.

```bash
model_demos/
├── cv_demos/
│ ├── resnet/
│ │ └── pytorch_resnet.py
│ ├── new_model_arch/
│ │ └── pytorch_new_model.py
```

If external dependencies are required, please add the dependencies to the [requriements.txt](requirements.txt) file.

### Cleaning the dev environment

`make clean` and `make clean_tt` clears out model and build artifacts. Please make sure no artifacts are being pushed.

### Running pre-commit hooks

You must run hooks before you commit something.

To manually run the style formatting, run:

```bash
make style
```

### Adding tests

For new model demos, please include a test case under `tests/` with the naming format of `test_[framework]_[model].py`.

Also include a pytest marker for each model family and update the markers list in [pyproject.toml](pyproject.toml).

For example,

```python
tests/test_pytorch_bert.py

@pytest.mark.bert
def test_bert_masked_lm_pytorch(clear_pybuda):
    run_bert_masked_lm_pytorch()
```

### Updating Models Table

For new model demos, please include an entry in the [Models Table](README.md/#models-table) along with the supported hardware.


### Model Weights:

In general, we avoid including files that contain data, weights, etc within the files. Instead, these should be incorporated into the model script at runtime. For a practical illustration of this approach, please refer to this [example](https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/resnet/onnx_resnet.py#L68).

## Contribution standards

### Code reviews

We actively welcome your pull requests.
- A PR must be opened for any code change with the following criteria:
  - Be approved, by a maintaining team member and any codeowners whose modules
    are relevant for the PR.
  - Run pre-commit hooks.
  - Pass any acceptance criteria mandated in the original issue.
  - Pass automated github actions worksflow test
  - Pass any testing criteria mandated by codeowners whose modules are relevant
    for the PR.

For more information on the GitHub Actions and Pull Request Workflow, please see the [GitHub Actions and Pull Request Workflow section](#github-actions-and-pull-request-workflow) within the document.

### GitHub Actions and Pull Request Workflow

#### Automated Linting with GitHub Actions

Linting, styling, and cleaning checks are automatically performed on pull requests using GitHub Actions. This ensures that contributed code meets standard Python coding standards before it's merged into the main branch.

1. **Pull Request Process**: When you open a pull request, GitHub Actions will automatically trigger linting, styling, and cleaning checks on the changes made within the `model_demos` directory.

2. **Approval Requirement**: In order to merge a pull request, it must pass the GitHub Actions workflow test. This ensures that all contributions adhere to our coding standards and maintain consistency throughout our `tt-buda-demos` repository.

3. **Interpreting Results**: If linting fails on your pull request, review the output to identify and fix any issues. You'll need to address these issues before the pull request can be approved and merged. In case of repeated failures or failures within the GitHub Actions workflow files, please reach out to one of the repository maintainers from the [Maintainers.md](MAINTAINERS.md).

#### Automated Commit by GitHub Actions

The GitHub Actions workflow also automatically makes a commit with the message ```*** AUTOMATED COMMIT | Applied Code Formatting and Cleanup ✨ 🍰 ✨***``` authored by ```[anirudTT]``` when it performs code formatting and cleanup. If you open a pull request and subsequently push more changes, we suggest **rebasing or pulling again** from the pull request branch before pushing your changes again to avoid conflicts.


