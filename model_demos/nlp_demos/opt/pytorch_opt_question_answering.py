# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# OPT Demo Script - Question Answering

import os

import pybuda
import torch
from transformers import AutoTokenizer, OPTForQuestionAnswering


def run_opt_question_answering(variant="facebook/opt-350m", batch_size=1):

    # set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b

    # Load tokenizer and model from HuggingFace
    # Variants: "facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"
    # NOTE: These model variants are pre-trined only. They need to be fine-tuned
    # on a downstream task. Code is for demonstration purposes only.
    model_ckpt = variant

    if model_ckpt == "facebook/opt-1.3b":
        compiler_cfg.default_df_override = pybuda.DataFormat.Float16
        os.environ["PYBUDA_FORCE_SEQUENTIAL"] = "1"

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = OPTForQuestionAnswering.from_pretrained(model_ckpt, torchscript=True)

    # Load data sample
    question = ["Who was Jim Henson?"] * batch_size
    context = ["Jim Henson was a nice puppet"] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_opt_question_answering", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

    # Combine outputs for data parallel runs
    if os.environ.get("PYBUDA_N300_DATA_PARALLEL", "0") == "1":
        concat_answer_start = torch.cat((output[0].to_pytorch(), output[1].to_pytorch()), dim=0)
        concat_answer_end = torch.cat((output[2].to_pytorch(), output[3].to_pytorch()), dim=0)
        buda_answer_start = pybuda.Tensor.create_from_torch(concat_answer_start)
        buda_answer_end = pybuda.Tensor.create_from_torch(concat_answer_end)
        output = [buda_answer_start, buda_answer_end]

    # Data postprocessing
    for sample_id in range(batch_size):
        answer_start = output[0].value()[sample_id].argmax(-1).item()
        answer_end = output[1].value()[sample_id].argmax(-1).item()
        answer = tokenizer.decode(input_tokens["input_ids"][sample_id, answer_start : answer_end + 1])

        # Answer - "Denver Broncos"
        print(f"Sample ID: {sample_id}")
        print(f"Context: {context[sample_id]}")
        print(f"Question: {question[sample_id]}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    run_opt_question_answering()
