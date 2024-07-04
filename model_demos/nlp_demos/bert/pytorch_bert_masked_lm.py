# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BERT Demo Script - Masked LM

import os

import pybuda
import torch
from transformers import BertForMaskedLM, BertTokenizer


def run_bert_masked_lm_pytorch(batch_size=1):

    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForMaskedLM.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = ["The capital of France is [MASK]."] * batch_size

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
        pybuda.PyTorchModule("pt_bert_masked_lm", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    for sample_id in range(batch_size):
        # Data postprocessing
        mask_token_index = (input_tokens["input_ids"] == tokenizer.mask_token_id)[sample_id].nonzero(as_tuple=True)[0]
        predicted_token_id = output[0].value()[sample_id, mask_token_index].argmax(axis=-1)
        answer = tokenizer.decode(predicted_token_id)

        # Answer - "paris"
        print(f"Sample ID: {sample_id}")
        print(f"Context: {sample_text[sample_id]}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    run_bert_masked_lm_pytorch()
