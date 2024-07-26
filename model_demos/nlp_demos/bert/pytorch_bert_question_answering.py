# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# BERT Demo Script - SQuADv1.1 QA

import os

import pybuda
import torch
from transformers import BertForQuestionAnswering, BertTokenizer


def run_bert_question_answering_pytorch(batch_size=1):

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b
    compiler_cfg.default_dram_parameters = False
    compiler_cfg.balancer_policy = "Ribbon"
    os.environ["PYBUDA_RIBBON2"] = "1"
    os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{81*1024}"

    # Load Bert tokenizer and model from HuggingFace
    model_ckpt = "bert-large-cased-whole-word-masking-finetuned-squad"
    tokenizer = BertTokenizer.from_pretrained(model_ckpt)
    model = BertForQuestionAnswering.from_pretrained(model_ckpt)

    # Load data sample from SQuADv1.1
    context = (
        [
            """Super Bowl 50 was an American football game to determine the champion of the National Football League
    (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the
    National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title.
    The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
    As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed
    initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals
    (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently
    feature the Arabic numerals 50."""
        ]
        * batch_size
    )

    question = ["Which NFL team represented the AFC at Super Bowl 50?"] * batch_size

    # Data preprocessing
    input_tokens = tokenizer(
        question,
        context,
        max_length=384,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # Run inference on Tenstorrent device
    output_q = pybuda.run_inference(
        pybuda.PyTorchModule("pt_bert_question_answering", model),
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
    run_bert_question_answering_pytorch()
