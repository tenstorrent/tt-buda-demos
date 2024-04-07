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

| **Model** | **Supported Hardware** <br> GS - Grayskull <br> WH - Wormhole | **Supported Release** |
|--------------------------------------------------------------|:------------:|:-------:|
|   [ALBERT](nlp_demos/albert/)                                |     GS, WH   | v0.10.5 |
|   [Autoencoder (convolutional)](cv_demos/conv_autoencoder/)  |     GS, WH   | v0.10.5 |
|   [Autoencoder (linear)](cv_demos/linear_autoencoder/)       |     GS, WH   | v0.10.5 |
|   [BeiT](cv_demos/beit/)                                     |     GS, WH   | v0.10.5 |
|   [BERT](nlp_demos/bert/)                                    |     GS, WH   | v0.10.5 |
|   [CLIP](cv_demos/clip/)                                     |     GS, WH   | v0.10.5 |
|   [CodeGen](nlp_demos/codegen/)                              |     GS, WH   | v0.10.5 |
|   [DeiT](cv_demos/deit/)                                     |     GS, WH   | v0.10.5 |
|   [DenseNet](cv_demos/densenet/)                             |     GS, WH   | v0.10.5 |
|   [DistilBERT](nlp_demos/distilbert/)                        |     GS, WH   | v0.10.5 |
|   [DPR](nlp_demos/dpr/)                                      |     GS, WH   | v0.10.5 |
|   [EfficientNet-Lite](cv_demos/efficientnet_lite/)           |         WH   | v0.10.5 |
|   [Falcon-7B](nlp_demos/falcon/)                             |         WH   | v0.10.5 |
|   [FLAN-T5](nlp_demos/flant5/)                               |     GS, WH   | v0.10.5 |
|   [FPN](cv_demos/fpn/)                                       |     GS, WH   |         |
|   [Fuyu-8B](nlp_demos/fuyu8b/)                               |              |         |
|   [GhostNet](cv_demos/ghostnet/)                             |     GS, WH   | v0.10.5 |
|   [GoogLeNet](cv_demos/googlenet/)                           |     GS, WH   | v0.10.5 |
|   [GPT-2](nlp_demos/gpt2/)                                   |     GS, WH   | v0.10.5 |
|   [GPT Neo](nlp_demos/gptneo/)                               |     GS, WH   | v0.10.5 |
|   [Hand Landmark](cv_demos/landmark/)                        |         WH   | v0.10.5 |
|   [HardNet](cv_demos/hardnet/)                               |         WH   |         |
|   [HRNet](cv_demos/hrnet/)                                   |     GS, WH   | v0.10.5 |
|   [Inception-v4](cv_demos/inceptionv4/)                      |     GS, WH   | v0.10.5 |
|   [MLP-Mixer](cv_demos/mlpmixer/)                            |     GS, WH   | v0.10.5 |
|   [MobileNetSSD](cv_demos/mobilenet_ssd/)                    |         WH   | v0.10.5 |
|   [MobileNetV1](cv_demos/mobilenet_v1/)                      |     GS, WH   | v0.10.5 |
|   [MobileNetV2](cv_demos/mobilenet_v2/)                      |     GS, WH   | v0.10.5 |
|   [MobileNetV3](cv_demos/mobilenet_v3/)                      |     GS, WH   | v0.10.5 |
|   [OpenPose](cv_demos/openpose/)                             |     GS, WH   | v0.10.5 |
|   [OPT](nlp_demos/opt/)                                      |     GS, WH   | v0.10.5 |
|   [Pose Landmark](cv_demos/landmark/)                        |         WH   | v0.10.5 |
|   [Perceiver IO](cv_demos/perceiverio/)                      |     GS, WH   | v0.10.9-alpha |
|   [ResNet](cv_demos/resnet/)                                 |     GS, WH   | v0.10.5 |
|   [ResNeXt](cv_demos/resnext/)                               |     GS, WH   | v0.10.5 |
|   [RetinaNet](cv_demos/retinanet/)                           |     GS, WH   | v0.10.5 |
|   [RoBERTa](nlp_demos/roberta/)                              |     GS, WH   | v0.10.5 |
|   [SqueezeBERT](nlp_demos/squeezebert/)                      |     GS, WH   | v0.10.5 |
|   [Stable Diffusion](cv_demos/stable_diffusion/)             |         WH   | v0.10.5 |
|   [T5](nlp_demos/t5/)                                        |     GS, WH   | v0.10.5 |
|   [U-Net](cv_demos/unet/)                                    |     GS, WH   | v0.10.5 |
|   [VGG](cv_demos/vgg/)                                       |     GS, WH   | v0.10.5 |
|   [ViT](cv_demos/vit/)                                       |     GS, WH   | v0.10.5 |
|   [ViLT](cv_demos/vilt/)                                     |     GS, WH   | v0.10.5 |
|   [VoVNet](cv_demos/vovnet/)                                 |     GS, WH   | v0.10.5 |
|   [WideResNet](cv_demos/wideresnet/)                         |     GS, WH   | v0.10.5 |
|   [Whisper](audio_demos/whisper/)                            |     GS, WH   | v0.10.5 |
|   [Xception](cv_demos/xception/)                             |     GS, WH   | v0.10.5 |
|   [XGLM](nlp_demos/xglm/)                                    |     GS, WH   | v0.10.5 |
|   [YOLOv3](cv_demos/yolo_v3/)                                |     GS, WH   | v0.10.5 |
|   [YOLOv5](cv_demos/yolo_v5/)                                |     GS, WH   | v0.10.5 |


## Note:

Please note that alpha relases are not stable releases and may not support or have all functionality as the stable releaes. If full functionality is needed we suggest picking a stable relases.
