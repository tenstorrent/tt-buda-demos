# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Qwen1.5-0.5B Demo - Text Generation

import os

import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer


def run_qwen1_5_causal_lm(batch_size=1):
    os.environ["TT_BACKEND_TIMEOUT"] = "0"

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"

    # Setup model configuration
    config = Qwen2Config.from_pretrained("Qwen/Qwen1.5-0.5B")
    config.use_cache = False
    config.return_dict = False

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Disable DynamicCache
    # See: https://github.com/tenstorrent/tt-buda/issues/42
    model._supports_cache_class = False

    # Example usage
    prompt = ["My name is Jim Keller and"] * batch_size

    # Initialize pipeline
    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Inference on TT device
    response = text_generator(
        prompt,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_new_tokens=512,
        num_beams=1,
        do_sample=True,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True,
    )

    # Display Responses
    for batch_id in range(batch_size):
        print(f"Batch: {batch_id}")
        print(f"Response: {response[batch_id][0]['generated_text']}")
        print()


if __name__ == "__main__":
    run_qwen1_5_causal_lm()
