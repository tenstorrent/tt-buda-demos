# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pybuda
import requests
import torch
from PIL import Image
from transformers import ViltConfig, ViltForQuestionAnswering, ViltProcessor

from cv_demos.vilt.vilt_model import ViLtEmbeddingWrapper, ViltModelWrapper


def run_vilt_for_question_answering_pytorch(variant="dandelin/vilt-b32-finetuned-vqa", batch_size=1):

    # Set PyBuda configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.balancer_policy = "Ribbon"

    os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
    os.environ["PYBUDA_RIBBON2"] = "1"

    # Sample Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    sample_image = Image.open(requests.get(url, stream=True).raw)
    batch_image = [sample_image] * batch_size

    # Sample text
    batch_text = ["How many cats are there?"] * batch_size

    model_ckpt = variant

    # Set model configurations
    config = ViltConfig.from_pretrained(model_ckpt)  # matmul_2008
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config = ViltConfig(**config_dict)

    # Load model and processor from HuggingFace
    processor = ViltProcessor.from_pretrained(model_ckpt)
    model = ViltForQuestionAnswering.from_pretrained(model_ckpt, config=config)
    model.eval()

    # Sample inputs
    encoding = processor(batch_image, batch_text, return_tensors="pt")

    # Wrapper
    text_vision_embedding_model = ViLtEmbeddingWrapper(model)
    viltquestionanswering_model = ViltModelWrapper(model, task="qa")

    embedding_output, attention_mask = text_vision_embedding_model(**encoding)

    tt0 = pybuda.TTDevice("tt0", module=pybuda.PyTorchModule("pt_viltquestionanswering", viltquestionanswering_model))

    tt0.push_to_inputs(embedding_output.detach().cpu(), attention_mask.detach().cpu().to(torch.float32))

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(_sequential=True)

    # Model output (i.e Predicted answer: 2)
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    for sample_id in range(batch_size):
        predicted_value = output[0].value()[sample_id].argmax(-1).item()
        print(
            f"Sample ID: {sample_id} | Question: {batch_text[sample_id]} | Predicted Sentiment: {model.config.id2label[predicted_value]}"
        )


if __name__ == "__main__":
    run_vilt_for_question_answering_pytorch()
