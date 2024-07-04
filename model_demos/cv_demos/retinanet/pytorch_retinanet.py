import os
import shutil
import sys
import zipfile

import pybuda
import requests
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision import transforms

from cv_demos.retinanet.model.model_implementation import Model
from cv_demos.retinanet.post_process.post_process import detection_postprocess

torch.multiprocessing.set_sharing_strategy("file_system")


def img_preprocess():

    url = "https://i.ytimg.com/vi/q71MCWAEfL8/maxresdefault.jpg"
    pil_img = Image.open(requests.get(url, stream=True).raw)
    new_size = (640, 480)
    pil_img = pil_img.resize(new_size, resample=Image.BICUBIC)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img


def run_retinanet_pytorch(variant, batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_DECOMPOSE_SIGMOID"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "73728"

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0:

            if variant == "retinanet_rn18fpn":
                compiler_cfg.place_on_new_epoch("conv2d_357.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_300.dc.matmul.11", "grid_shape", (1, 1))

            elif variant == "retinanet_rn34fpn":
                compiler_cfg.place_on_new_epoch("conv2d_589.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_532.dc.matmul.11", "grid_shape", (1, 1))

            elif variant == "retinanet_rn50fpn":
                compiler_cfg.place_on_new_epoch("conv2d_826.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_769.dc.matmul.11", "grid_shape", (1, 1))

            elif variant == "retinanet_rn101fpn":
                compiler_cfg.place_on_new_epoch("conv2d_1557.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_1500.dc.matmul.11", "grid_shape", (1, 1))

            elif variant == "retinanet_rn152fpn":
                compiler_cfg.place_on_new_epoch("conv2d_2288.dc.matmul.11")
                compiler_cfg.balancer_op_override("conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1))
                compiler_cfg.balancer_op_override("conv2d_2231.dc.matmul.11", "grid_shape", (1, 1))

        elif available_devices[0] == BackendDevice.Grayskull:

            if variant == "retinanet_rn18fpn":
                compiler_cfg.balancer_op_override("conv2d_322.dc.matmul.11", "t_stream_shape", (1, 1))

            elif variant == "retinanet_rn34fpn":
                compiler_cfg.balancer_op_override("conv2d_554.dc.matmul.11", "t_stream_shape", (1, 1))

            elif variant == "retinanet_rn50fpn":
                compiler_cfg.balancer_op_override("conv2d_791.dc.matmul.11", "t_stream_shape", (1, 1))

            elif variant == "retinanet_rn101fpn":
                compiler_cfg.balancer_op_override("conv2d_1522.dc.matmul.11", "t_stream_shape", (1, 1))

            elif variant == "retinanet_rn152fpn":
                compiler_cfg.balancer_op_override("conv2d_2253.dc.matmul.11", "t_stream_shape", (1, 1))

    # Prepare model

    url = f"https://github.com/NVIDIA/retinanet-examples/releases/download/19.04/{variant}.zip"
    local_zip_path = f"{variant}.zip"

    response = requests.get(url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)

    # Unzip the file
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    # Find the path of the .pth file
    extracted_path = f"{variant}"
    checkpoint_path = ""
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            if file.endswith(".pth"):
                checkpoint_path = os.path.join(root, file)

    model = Model.load(checkpoint_path)
    model.eval()
    tt_model = pybuda.PyTorchModule(f"pt_{variant}", model)

    # Prepare input
    input_batch = [img_preprocess()] * batch_size
    batch_input = torch.cat(input_batch, dim=0)

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([batch_input]))
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    for i in range(len(output)):
        output[i] = output[i].value()

    # URL to download the classes text file
    url = "https://github.com/matlab-deep-learning/Object-Detection-Using-Pretrained-YOLO-v2/blob/main/+helper/coco-classes.txt?raw=true"
    response = requests.get(url)

    # Split the text content into lines
    lines = response.text.split("\n")
    # Remove empty lines and strip leading/trailing spaces
    lines = [line.strip() for line in lines if line.strip()]
    # Create a dictionary from the lines with indices as keys
    class_names = {i: line for i, line in enumerate(lines)}

    cls_heads = output[:5]
    box_heads = output[5:]

    for sample_id in range(batch_size):

        scores, boxes, labels = detection_postprocess(
            input_batch[sample_id],
            [tensor[sample_id].unsqueeze(0) for tensor in cls_heads],
            [tensor[sample_id].unsqueeze(0) for tensor in box_heads],
        )

        print(f"Sample {sample_id} Detections :\n")
        for i in range(scores.size(1)):
            score = scores[0, i].item()
            if score == 0:
                break
            label = int(labels[0, i].item())
            box = boxes[0, i, :].tolist()
            class_name = class_names[label]

            print(
                f"Class: {class_name}, Score: {score:.4f}, Box Coordinates: [{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f}]"
            )

    # Delete the extracted folder and the zip file
    shutil.rmtree(extracted_path)
    os.remove(local_zip_path)


if __name__ == "__main__":
    run_retinanet_pytorch(variant="retinanet_rn18fpn")
