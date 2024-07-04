# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# GPT Neo Demo Script - Sequence Classification

import os

import pybuda
import torch
from pybuda._C.backend_api import BackendDevice
from transformers import AutoTokenizer, GPTNeoForSequenceClassification


def run_gptneo_sequence_classification(variant="EleutherAI/gpt-neo-125M", batch_size=1):

    # Load tokenizer and model from HuggingFace
    # Variants: # EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B
    model_ckpt = variant

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    if "1.3B" in model_ckpt or "2.7B" in model_ckpt:
        os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

    if variant == "EleutherAI/gpt-neo-2.7B":
        available_devices = pybuda.detect_available_devices()
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoForSequenceClassification.from_pretrained(model_ckpt, torchscript=True)

    # Load data sample
    review = ["the movie was great!"] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_gptneo_seq_classification", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Data postprocessing
    for sample_id in range(batch_size):
        predicted_value = output[0].value()[sample_id].argmax(-1).item()

        # Answer - "positive"
        print(
            f"Sample ID: {sample_id} | Review: {review[sample_id]} | Predicted Sentiment: {model.config.id2label[predicted_value]}"
        )


if __name__ == "__main__":
    run_gptneo_sequence_classification()
