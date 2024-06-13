# ssd300_resnet50 demo script

import pybuda
import numpy as np
import torch
import os
import skimage
import requests
from pybuda._C.backend_api import BackendDevice


# preprocessing scripts referred from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/examples/SSD300_inference.py


def load_image(image_path):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    mean, std = 128, 128
    img = skimage.img_as_float(skimage.io.imread(image_path))
    if len(img.shape) == 2:
        img = np.array([img, img, img]).swapaxes(0, 2)
    return img


def rescale(img, input_height, input_width):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if aspect < 1:
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if aspect == 1:
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    return imgScaled


def crop_center(img, cropx, cropy):
    """Code from Loading_Pretrained_Models.ipynb - a Caffe2 tutorial"""
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def normalize(img, mean=128, std=128):
    img = (img * 256 - mean) / std
    return img


def prepare_input(img_uri):
    img = load_image(img_uri)
    img = rescale(img, 300, 300)
    img = crop_center(img, 300, 300)
    img = normalize(img)
    return img


def run_pytorch_ssd300_resnet50():

    # STEP 1 : Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.balancer_op_override("max_pool2d_14.dc.sparse_matmul.5.dc.sparse_matmul.1.lc2", "t_stream_shape", (2, 1))

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Grayskull:
            os.environ["PYBUDA_RIBBON2"] = "1"
            compiler_cfg.place_on_new_epoch("reshape_769.dc.matmul.13")
            compiler_cfg.place_on_new_epoch("reshape_769.dc.matmul.7")

        if available_devices[0] == BackendDevice.Wormhole_B0:
            compiler_cfg.place_on_new_epoch("conv2d_766.dc.matmul.11")
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "12288"

    # STEP 2 : prepare model
    model = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd", pretrained=False)
    url = "https://api.ngc.nvidia.com/v2/models/nvidia/ssd_pyt_ckpt_amp/versions/19.09.0/files/nvidia_ssdpyt_fp16_190826.pt"
    checkpoint_path = "nvidia_ssdpyt_fp16_190826.pt"

    response = requests.get(url)
    with open(checkpoint_path, "wb") as f:
        f.write(response.content)

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tt_model = pybuda.PyTorchModule("pt_ssd300_resnet50", model)

    # STEP 3 : prepare input
    img = "http://images.cocodataset.org/val2017/000000397133.jpg"
    HWC = prepare_input(img)
    CHW = np.swapaxes(np.swapaxes(HWC, 0, 2), 1, 2)
    batch = np.expand_dims(CHW, axis=0)
    input_batch = torch.from_numpy(batch).float()

    # STEP 4 : Inference
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
    output = output_q.get()

    for i in range(len(output)):
        output[i] = output[i].value()

    output = [o.detach().clone() for o in output]

    # postprocessing scripts referred from https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/

    # STEP 5 : Postprocess
    utils = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_ssd_processing_utils")
    classes_to_labels = utils.get_coco_object_dictionary()
    results_per_input = utils.decode_results(output)
    best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

    for image_idx in range(len(best_results_per_input)):
        bboxes, classes, confidences = best_results_per_input[image_idx]
        for idx in range(len(bboxes)):
            left, bot, right, top = bboxes[idx]
            x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
            box_info = f"Co-ordinates: [{left:.2f}, {bot:.2f}, {right:.2f}, {top:.2f}]"
            class_info = f"Class: {classes_to_labels[classes[idx] - 1]}"
            confidence_info = f"Confidence: {confidences[idx]*100:.2f}%"
            print(f"{box_info}, {class_info}, {confidence_info}")

    # STEP 6 : remove the downloaded weights
    os.remove(checkpoint_path)


if __name__ == "__main__":
    run_pytorch_ssd300_resnet50()
