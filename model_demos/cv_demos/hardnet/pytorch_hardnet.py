import os
import urllib

import pybuda
import torch
from PIL import Image
from pybuda._C.backend_api import BackendDevice
from torchvision import transforms


def run_hardnet_pytorch(variant):

    # STEP 1: Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.balancer_policy = "Ribbon"
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Device specific configurations
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if available_devices[0] == BackendDevice.Wormhole_B0 and variant == "hardnet85":
            os.environ["PYBUDA_FORCE_CONV_MULTI_OP_FRACTURE"] = "1"

    # load only the model architecture without pre-trained weights.
    model = torch.hub.load("PingoLH/Pytorch-HarDNet", variant, pretrained=False)

    if variant == "hardnet68":
        checkpoint_url = "https://ping-chao.com/hardnet/hardnet68-5d684880.pth"

    if variant == "hardnet85":
        checkpoint_url = "https://ping-chao.com/hardnet/hardnet85-a28faa00.pth"

    if variant == "hardnet68ds":
        torch.multiprocessing.set_sharing_strategy("file_system")
        checkpoint_url = "https://ping-chao.com/hardnet/hardnet68ds-632474d2.pth"

    if variant == "hardnet39ds":
        checkpoint_url = "https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth"

    # path to save the weights
    checkpoint_path = f"./{variant}.pth"

    # Download model weights from specified URLs in checkpoint_path
    urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

    # Load weights from the checkpoint file and maps tensors to CPU, ensuring compatibility even without a GPU.
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # Inject weights into model
    model.load_state_dict(state_dict)
    model.eval()

    # STEP 2: Create PyBuda module from PyTorch model
    model_name = f"pt_{variant}"
    tt_model = pybuda.PyTorchModule(model_name, model)

    # STEP 3: Prepare input
    url, filename = (
        "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
        "dog.jpg",
    )
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    input_image = Image.open(filename)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # run inference on Tenstorrent device
    output_q = pybuda.run_inference(tt_model, inputs=([input_batch]))
    output = output_q.get()[0].value()

    # Data postprocessing
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # Get imagenet class mappings
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    image_classes = urllib.request.urlopen(url)
    categories = [s.decode("utf-8").strip() for s in image_classes.readlines()]

    # Print top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())

    # Remove the weights file
    os.remove(checkpoint_path)

    # Remove the downloaded image
    os.remove(filename)


if __name__ == "__main__":
    run_hardnet_pytorch()
