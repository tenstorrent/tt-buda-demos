# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# DistilBERT Demo Script - NER

import os

import pybuda
import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizer


def run_distilbert_token_classification_pytorch(batch_size=1):

    # Load DistilBERT tokenizer and model from HuggingFace
    model_ckpt = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)
    model = DistilBertForTokenClassification.from_pretrained(model_ckpt)

    compiler_cfg = pybuda.config._get_global_compiler_config()  # load global compiler config object
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Load data sample
    sample_text = ["HuggingFace is a company based in Paris and New York"] * batch_size

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
        pybuda.PyTorchModule("pt_distilbert_token_classification", model),
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
        predicted_token_class_ids = output[0].value()[sample_id].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (input_tokens["attention_mask"][sample_id] == 1)
        )
        predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]

        # Answer - ['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']
        print(f"Sample ID: {sample_id}")
        print(f"Context: {sample_text[sample_id]}")
        print(f"Answer: {predicted_tokens_classes}")


if __name__ == "__main__":
    run_distilbert_token_classification_pytorch()
