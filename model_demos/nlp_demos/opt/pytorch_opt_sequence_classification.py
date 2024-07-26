# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# OPT Demo Script - Sequence Classification

import os

import pybuda
import torch
from transformers import AutoTokenizer, OPTForSequenceClassification


def run_opt_sequence_classification(variant="facebook/opt-350m", batch_size=1):

    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("adv_index")
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_ckpt = variant
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = OPTForSequenceClassification.from_pretrained(model_ckpt, torchscript=True)

    if model_ckpt == "facebook/opt-1.3b" or model_ckpt == "facebook/opt-350m":
        compiler_cfg.enable_auto_fusing = False
        if model_ckpt == "facebook/opt-1.3b":
            os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

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
        pybuda.PyTorchModule("pt_opt_sequence_classification", model),
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
    run_opt_sequence_classification()
