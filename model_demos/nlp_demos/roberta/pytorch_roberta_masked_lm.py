# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# RoBERTa demo script - Masked language modeling

import os

import pybuda
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def run_roberta_mlm_pytorch(batch_size=1):

    # Load RoBERTa tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base")

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Input processing
    sample_text = ["Hello I'm a <mask> model."] * batch_size
    input_tokens = tokenizer(
        sample_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_roberta", model),
        inputs=[input_tokens],
    )
    output = output_q.get()  # inference will return a queue object, get last returned object

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    # Output processing
    output_pb = output[0].value()
    scores = output_pb.softmax(dim=-1)
    for idx, score in enumerate(scores):
        mask_token_index = (input_tokens["input_ids"] == tokenizer.mask_token_id)[idx].nonzero(as_tuple=True)[0]
        predicted_token_rankings = output_pb[idx, mask_token_index].argsort(axis=-1, descending=True)[0]

        # Report output
        top_k = 5
        print(f"Sample ID: {idx}")
        print(f"Masked text: {sample_text[idx]}")
        print(f"Top {top_k} predictions:")
        for i in range(top_k):
            prediction = tokenizer.decode(predicted_token_rankings[i])
            pred_score = score[mask_token_index, predicted_token_rankings[i]]
            print(f"{i+1}: {prediction} (score = {round(float(pred_score), 3)})")


if __name__ == "__main__":
    run_roberta_mlm_pytorch()
