# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# RoBERTa Demo Script - Text Classification

import csv
import os
import urllib.request

import pybuda
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def run_roberta_sentiment_pytorch(batch_size=2):

    # Load Bart tokenizer and model from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load label mapping
    labels = []
    mapping_link = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
    with urllib.request.urlopen(mapping_link) as f:
        html = f.read().decode("utf-8").split("\n")
        csvreader = csv.reader(html, delimiter="\t")
    labels = [row[1] for row in csvreader if len(row) > 1]

    # Example from multi-nli validation set
    text = ["""Great road trip views! @ Shartlesville, Pennsylvania"""] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(pybuda.PyTorchModule("pt_roberta", model), inputs=[input_tokens])
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_tensor = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        buda_tensor = pybuda.Tensor.create_from_torch(concat_tensor)
        output = [buda_tensor]

    for sample_id in range(batch_size):
        # Data postprocessing
        logits = output[0].value()[sample_id, :]
        scores = logits.softmax(dim=0)
        ranking = scores.argsort(descending=True)

        # Report results
        print(f"Sample ID: {sample_id}")
        print(f"Text: {text[sample_id]}")
        for i in range(scores.shape[0]):
            label = labels[ranking[i]]
            score = scores[ranking[i]]
            print(f"{i+1}) {label} {round(float(score), 3)}")


if __name__ == "__main__":
    run_roberta_sentiment_pytorch()
