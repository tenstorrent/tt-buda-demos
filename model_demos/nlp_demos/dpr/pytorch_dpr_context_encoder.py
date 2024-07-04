# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DPR Demo Script - Context Encoder

import os

import pybuda
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer


def run_dpr_context_encoder_pytorch(variant="facebook/dpr-ctx_encoder-multiset-base", batch_size=1):

    # Load Bert tokenizer and model from HuggingFace
    # Variants: facebook/dpr-ctx_encoder-single-nq-base, facebook/dpr-ctx_encoder-multiset-base
    model_ckpt = variant
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_ckpt)
    model = DPRContextEncoder.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = ["Hello, is my dog cute?"] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_dpr_context_encoder", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Postprocessing
    embeddings = output[0].value()

    # Print Outputs
    for sample_id in range(batch_size):
        print(f"Sample ID: {sample_id} | Context: {sample_text[sample_id]} | Embeddings: {embeddings[sample_id]}")


if __name__ == "__main__":
    run_dpr_context_encoder_pytorch()
