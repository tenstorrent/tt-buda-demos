# TT-Buda Demos

## Table of Contents

- [Introduction](#introduction)
- [First 5 Things To-Do](first_5_steps/)
- [Model Demos](model_demos/)
- [Documentation](https://docs.tenstorrent.com/)
- [FAQ & Troubleshooting](FAQ.md)
- [Communication](#communication)
- [Contributing](#contributing)

## Introduction

The TT-Buda software stack can compile AI/ML models from several different frameworks such as PyTorch and Tensorflow, and execute them in many different ways on Tenstorrent hardware.

**Note on terminology:**

TT-Buda is the official Tenstorrent AI/ML compiler stack and PyBuda is the Python interface for TT-Buda. PyBuda allows users to access and utilize TT-Buda's features directly from Python. This includes directly importing model architectures and weights from PyTorch, TensorFlow, ONNX, and TFLite.

## First 5 Things To-Do

For a simple, 5-step, starting guide on learning the basics of TT-Buda please visit [first_5_steps](first_5_steps/).

In that directory, you will find the following user guides:

- `1_install_ttbuda.md` -> Installation guide for TT-Buda
- `2_running_nlp_models.ipynb` -> Run your first NLP model with TT-Buda
- `3_running_cv_models.ipynb` -> Run your first CNN model with TT-Buda
- `4_batched_inputs.ipynb` -> Learn how to run with batched inputs and how to benchmark models on TT-Buda
- `5_serving_tt_models.ipynb` -> Use FastAPI to host a model running on Tenstorrent hardware to build custom APIs

## Model Demos

For additional example code for running some popular AI/ML models, please visit [model_demos](model_demos/).

## Documentation

Please refer to the [official documentation website](https://docs.tenstorrent.com/).

## FAQ & Troubleshooting

We keep a running FAQ & troubleshoot guide to help you quickly resolve some of the most common issues at [FAQ & Troubleshooting](FAQ.md).

## Communication

If you would like to formally propose a new feature, report a bug, or have issues with permissions, please file through [GitHub issues](../../issues).

Please access the [Discord community](https://discord.gg/xUHw4tMcRV) forum for updates, tips, live troubleshooting support, and more!

## Contributing

We are excited to move our development to the public, open-source domain. However, we are not adequately staffed to review contributions in an expedient and manageable time frame at this time. In the meantime, please review the [contributor's guide](CONTRIBUTING.md) for more information about contribution standards.
