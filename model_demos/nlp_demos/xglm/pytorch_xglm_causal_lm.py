# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# XGLM Demo - CausalLM
import os

import pybuda
from pybuda import BackendDevice
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, XGLMConfig, XGLMForCausalLM


def run_xglm_causal_lm(variant="facebook/xglm-564M", batch_size=1):

    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.cpu_fallback_ops.add("take")
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_enumerate_u_kt = False

    # Variants: "facebook/xglm-564M", "facebook/xglm-1.7B"
    model_ckpt = variant
    available_devices = pybuda.detect_available_devices()
    if available_devices:
        if model_ckpt == "facebook/xglm-1.7B":
            compiler_cfg.amp_level = 1
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{16*1024}"
        if (available_devices[0] == BackendDevice.Grayskull and model_ckpt == "facebook/xglm-564M") or (
            available_devices[0] == BackendDevice.Wormhole_B0
        ):
            os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = "65536"
        if available_devices[0] == BackendDevice.Grayskull and model_ckpt == "facebook/xglm-564M":
            compiler_cfg.default_dram_parameters = True

    # set model configurations
    config = XGLMConfig.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = XGLMConfig(**config_dict)

    # Load tokenizer and model from HuggingFace
    model = XGLMForCausalLM.from_pretrained(model_ckpt, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = ["My name is Thomas and my main"] * batch_size

    # Create text generator object
    text_generator = pybuda_pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size)

    # Run inference on Tenstorrent device
    answer = text_generator(
        prefix_text,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        no_repeat_ngram_size=2,
    )

    # Report output
    for sample_id in range(batch_size):
        print(f"Sample ID: {sample_id}")
        print(f"Prefix text: {prefix_text[sample_id]}")
        print(f"Generated text: {answer[sample_id]}")


if __name__ == "__main__":
    run_xglm_causal_lm()
