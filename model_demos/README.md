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

| **Model**                                                 | **e75** | **e150** | **n150** | **n300 (single-chip)** | **Supported Release** |
| --------------------------------------------------------- | :-----: | :------: | :------: | :--------------------: | :-------------------: |
| [ALBERT](nlp_demos/albert/)                               |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Autoencoder (convolutional)](cv_demos/conv_autoencoder/) |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Autoencoder (linear)](cv_demos/linear_autoencoder/)      |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [BeiT](cv_demos/beit/)                                    |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [BERT](nlp_demos/bert/)                                   |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [CLIP](cv_demos/clip/)                                    |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [CodeGen](nlp_demos/codegen/)                             |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [DeiT](cv_demos/deit/)                                    |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [DenseNet](cv_demos/densenet/)                            |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [DistilBERT](nlp_demos/distilbert/)                       |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [DLA](cv_demos/dla/)                                      |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [DPR](nlp_demos/dpr/)                                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [EfficientNet-Lite](cv_demos/efficientnet_lite/)          |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.12.3        |
| [Falcon-7B](nlp_demos/falcon/)                            |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.18.2        |
| [FLAN-T5](nlp_demos/flant5/)                              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| FPN                                                       |   ✘     |    ✘     |    ✘     |             ✘          |        TBD            |
| [Fuyu-8B](nlp_demos/fuyu8b/)                              |   ✘     |    ✘     |    ✘     |             ✘          |        TBD            |
| [GhostNet](cv_demos/ghostnet/)                            |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [GoogLeNet](cv_demos/googlenet/)                          |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [GPT-2](nlp_demos/gpt2/)                                  |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [GPT Neo](nlp_demos/gptneo/)                              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Hand Landmark](cv_demos/landmark/)                       |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.12.3        |
| [HardNet](cv_demos/hardnet/)                              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [HRNet](cv_demos/hrnet/)                                  |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Inception-v4](cv_demos/inceptionv4/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [MLP-Mixer](cv_demos/mlpmixer/)                           |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [MobileNetSSD](cv_demos/mobilenet_ssd/)                   |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.12.3        |
| [MobileNetV1](cv_demos/mobilenet_v1/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [MobileNetV2](cv_demos/mobilenet_v2/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [MobileNetV3](cv_demos/mobilenet_v3/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Monodle](cv_demos/monodle/)                              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [OpenPose](cv_demos/openpose/)                            |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [OPT](nlp_demos/opt/)                                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Pose Landmark](cv_demos/landmark/)                       |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.12.3        |
| [Perceiver IO](cv_demos/perceiverio/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [ResNet](cv_demos/resnet/)                                |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [ResNeXt](cv_demos/resnext/)                              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [RetinaNet](cv_demos/retinanet/)                          |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [RoBERTa](nlp_demos/roberta/)                             |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [SegFormer](cv_demos/segformer/)                          |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [SqueezeBERT](nlp_demos/squeezebert/)                     |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [SSD300 ResNet50](cv_demos/ssd300_resnet50/)              |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Stable Diffusion](cv_demos/stable_diffusion/)            |   ✘     |    ✘     |    ✔️     |             ✔️          |        v0.18.2        |
| [T5](nlp_demos/t5/)                                       |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [U-Net](cv_demos/unet/)                                   |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [VGG](cv_demos/vgg/)                                      |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [ViLT](cv_demos/vilt/)                                    |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [ViT](cv_demos/vit/)                                      |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [VoVNet](cv_demos/vovnet/)                                |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [WideResNet](cv_demos/wideresnet/)                        |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Whisper](audio_demos/whisper/)                           |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [Xception](cv_demos/xception/)                            |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [XGLM](nlp_demos/xglm/)                                   |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [YOLOv3](cv_demos/yolo_v3/)                               |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [YOLOv5](cv_demos/yolo_v5/)                               |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |
| [YOLOv6](cv_demos/yolo_v6/)                               |   ✔️     |    ✔️     |    ✔️     |             ✔️          |        v0.18.2        |

### Legend

- ✔️: Supported on the device
- ✘: Not supported on the device

## Note

Please note that releases identified as alpha (e.g., v0.15.0-alpha) are preliminary and not considered stable. They may not offer the full functionality found in stable versions. Furthermore, alpha releases are compatible only with models specifically released under the same version, as detailed in the above table. For ensured full functionality, we strongly advise opting for a stable release.
