# yolo_v6 demo script

import math
import os

import cv2
import numpy as np
import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from yolov6 import YOLOV6
from yolov6.core.inferer import Inferer
from yolov6.utils.events import load_yaml
from yolov6.utils.nms import non_max_suppression

# preprocessing & postprocessing steps referred form https://github.com/meituan/YOLOv6/blob/main/inference.ipynb


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
        new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)


def check_img_size(img_size, s=32, floor=0):
    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f"WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}")
    return new_size if isinstance(img_size, list) else [new_size] * 2


def process_image(path, img_size, stride, half):
    """Process image before image inference."""

    img_src = np.asarray(Image.open(requests.get(path, stream=True).raw))
    image = letterbox(img_src, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))  # HWC to CHW
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    return image, img_src


def run_yolov6_pytorch(variant, batch_size=1):

    # STEP 1 : Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if variant in ["yolov6m", "yolov6l"]:
        os.environ["PYBUDA_FORK_JOIN_BUF_QUEUES"] = "1"
        os.environ["PYBUDA_FORK_JOIN_EXPAND_OUTPUT_BUFFERS"] = "1"
        os.environ["PYBUDA_FORK_JOIN_SKIP_EXPANDING_BUFFERS"] = "1"
        os.environ["PYBUDA_MAX_FORK_JOIN_BUF"] = "1"

        # Device specific configurations
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull and variant == "yolov6m":
                compiler_cfg.balancer_op_override(
                    "conv2d_258.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (1, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_258.dc.reshape.12.dc.sparse_matmul.3.lc2",
                    "t_stream_shape",
                    (2, 1),
                )

            if available_devices[0] == BackendDevice.Wormhole_B0 and variant == "yolov6l":
                os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

            if available_devices[0] == BackendDevice.Grayskull and variant == "yolov6l":
                compiler_cfg.balancer_op_override(
                    "conv2d_484.dc.reshape.0.dc.sparse_matmul.4.lc2", "grid_shape", (1, 1)
                )
                compiler_cfg.balancer_op_override(
                    "conv2d_484.dc.reshape.12.dc.sparse_matmul.3.lc2",
                    "t_stream_shape",
                    (2, 1),
                )

    # STEP 2 :prepare model
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{variant}.pt"
    weights = f"{variant}.pt"

    try:
        response = requests.get(url)
        with open(weights, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url} to {weights}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

    model = YOLOV6(weights)
    model = model.model
    model.eval()

    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # STEP 3 : prepare input
    url = "http://images.cocodataset.org/val2017/000000397133.jpg"
    stride = 32
    input_size = 640
    img_size = check_img_size(input_size, s=stride)
    img, img_src = process_image(url, img_size, stride, half=False)
    input_batch = img.unsqueeze(0)
    batch_input = torch.cat([input_batch] * batch_size, dim=0)

    # STEP 4 : Inference
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # STEP 5 : Postprocess
    det = non_max_suppression(output[0].value())

    # Send a GET request to fetch the YAML file
    response = requests.get("https://github.com/meituan/YOLOv6/raw/main/data/coco.yaml")

    with open("coco.yaml", "wb") as file:
        # Write the content to the file
        file.write(response.content)

    class_names = load_yaml("coco.yaml")["names"]

    if len(det):
        for sample in range(batch_size):
            print("Sample ID: ", sample)
            det[sample][:, :4] = Inferer.rescale(input_batch.shape[2:], det[sample][:, :4], img_src.shape).round()

            for *xyxy, conf, cls in reversed(det[sample]):
                class_num = int(cls)  # Convert class index to integer
                conf_value = conf.item()  # Get the confidence value
                coordinates = [int(x.item()) for x in xyxy]  # Convert tensor to list of integers

                # Get the class label
                label = class_names[class_num]

                # Detections
                print(f"Coordinates: {coordinates}, Class: {label}, Confidence: {conf_value:.2f}")
            print("\n")

    # STEP 6 : remove downloaded weights and YAML file
    os.remove(weights)
    os.remove("coco.yaml")


if __name__ == "__main__":
    run_yolov6_pytorch(variant="yolov6m")
