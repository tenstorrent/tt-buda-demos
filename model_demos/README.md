# Model Demos

Short demos for a broad range of NLP and CV models.

## Setup Instructions

### Install requirements

First, create either a Python virtual environment with PyBuda installed or execute from a Docker container with PyBuda installed.

Installation instructions can be found at [Install TT-Buda](../first_5_steps/1_install_tt_buda.md).

Next, install the model requirements:

```bash
pip install -r requirements.txt
```

## Quick Start

With an activate Python environment and all dependencies installed, run:

```bash
export PYTHONPATH=.
python cv_demos/resnet/pytorch_resnet.py
```

## Models Support Table

| **Model**                                                 | **e75** | **e150** | **n150** | **Supported Release** |
| --------------------------------------------------------- | :-----: | :------: | :------: | :-------------------: |
| [ALBERT](nlp_demos/albert/)                               |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Autoencoder (convolutional)](cv_demos/conv_autoencoder/) |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Autoencoder (linear)](cv_demos/linear_autoencoder/)      |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [BeiT](cv_demos/beit/)                                    |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [BERT](nlp_demos/bert/)                                   |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [CLIP](cv_demos/clip/)                                    |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [CodeGen](nlp_demos/codegen/)                             |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [DeiT](cv_demos/deit/)                                    |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [DenseNet](cv_demos/densenet/)                            |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [DistilBERT](nlp_demos/distilbert/)                       |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [DPR](nlp_demos/dpr/)                                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [EfficientNet-Lite](cv_demos/efficientnet_lite/)          |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [Falcon-7B](nlp_demos/falcon/)                            |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [FLAN-T5](nlp_demos/flant5/)                              |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Fuyu-8B](nlp_demos/fuyu8b/)                              |   ✘    |    ✘    |    ✘    |        v0.12.3        |
| [GhostNet](cv_demos/ghostnet/)                            |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [GoogLeNet](cv_demos/googlenet/)                          |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [GPT-2](nlp_demos/gpt2/)                                  |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [GPT Neo](nlp_demos/gptneo/)                              |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Hand Landmark](cv_demos/landmark/)                       |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [HardNet](cv_demos/hardnet/)                              |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [HRNet](cv_demos/hrnet/)                                  |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Inception-v4](cv_demos/inceptionv4/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [MLP-Mixer](cv_demos/mlpmixer/)                           |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [MobileNetSSD](cv_demos/mobilenet_ssd/)                   |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [MobileNetV1](cv_demos/mobilenet_v1/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [MobileNetV2](cv_demos/mobilenet_v2/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [MobileNetV3](cv_demos/mobilenet_v3/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [OpenPose](cv_demos/openpose/)                            |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [OPT](nlp_demos/opt/)                                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Pose Landmark](cv_demos/landmark/)                       |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [Perceiver IO](cv_demos/perceiverio/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [ResNet](cv_demos/resnet/)                                |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [ResNeXt](cv_demos/resnext/)                              |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [RetinaNet](cv_demos/retinanet/)                          |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [RoBERTa](nlp_demos/roberta/)                             |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [SqueezeBERT](nlp_demos/squeezebert/)                     |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Stable Diffusion](cv_demos/stable_diffusion/)            |   ✘    |    ✘    |    ✔️    |        v0.12.3        |
| [T5](nlp_demos/t5/)                                       |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [U-Net](cv_demos/unet/)                                   |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [VGG](cv_demos/vgg/)                                      |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [ViT](cv_demos/vit/)                                      |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [ViLT](cv_demos/vilt/)                                    |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [VoVNet](cv_demos/vovnet/)                                |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [WideResNet](cv_demos/wideresnet/)                        |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Whisper](audio_demos/whisper/)                           |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [Xception](cv_demos/xception/)                            |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [XGLM](nlp_demos/xglm/)                                   |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [YOLOv3](cv_demos/yolo_v3/)                               |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |
| [YOLOv5](cv_demos/yolo_v5/)                               |   ✔️    |    ✔️    |    ✔️    |        v0.12.3        |

### Legend

- ✔️: Supported on the device
- ✘: Not supported on the device

## Note:

Please note that releases identified as alpha (e.g., v0.10.9-alpha) are preliminary and not considered stable. They may not offer the full functionality found in stable versions. Furthermore, alpha releases are compatible only with models specifically released under the same version, as detailed in the above table. For ensured full functionality, we strongly advise opting for a stable release.
