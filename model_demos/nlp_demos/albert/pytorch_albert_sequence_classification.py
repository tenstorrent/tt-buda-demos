# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ALBERT Demo Script - SST-2 Text Classification

import os

import pybuda
import torch
from transformers import AlbertForSequenceClassification, AlbertTokenizer


def run_albert_sequence_classification_pytorch(batch_size=1):

    # Set PyBUDA configuration parameters
    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )

    # Load ALBERT tokenizer and model from HuggingFace
    model_ckpt = "textattack/albert-base-v2-imdb"
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForSequenceClassification.from_pretrained(model_ckpt)

    # Load data sample
    review = ["the movie was great!"] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        review,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_albert_sequence_classification", model),
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
    run_albert_sequence_classification_pytorch()
