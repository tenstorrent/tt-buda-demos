# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ALBERT demo script - Masked Language Modeling

import os

import pybuda
import torch
from pybuda._C.backend_api import BackendDevice
from transformers import AlbertForMaskedLM, AlbertTokenizer


def run_albert_masked_lm_pytorch(size="base", variant="v2", batch_size=1):
    available_devices = pybuda.detect_available_devices()

    # Set PyBUDA configuration parameters
    pybuda.config.set_configuration_options(
        default_df_override=pybuda.DataFormat.Float16,
        amp_level=2,
    )

    # Variants: albert-base-v1, albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1
    # albert-base-v2, albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2
    model_ckpt = f"albert-{size}-{variant}"
    if "xxlarge" in model_ckpt:
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.enable_auto_fusing = False
                compiler_cfg.amp_level = 2
                os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
                if variant == "v2":
                    compiler_cfg.enable_enumerate_u_kt = False
            elif available_devices[0] == BackendDevice.Wormhole_B0:
                pybuda.config.set_configuration_options(
                    enable_t_streaming=True,
                    enable_auto_fusing=False,
                    enable_enumerate_u_kt=False,
                    amp_level=1,
                    default_df_override=pybuda.DataFormat.Float16_b,
                )
                os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{105*1024}"
                os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"
    elif "xlarge" in model_ckpt:
        os.environ["TT_BACKEND_OVERLAY_MAX_EXTRA_BLOB_SIZE"] = f"{8*1024}"

        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                compiler_cfg = pybuda.config._get_global_compiler_config()
                os.environ["PYBUDA_NLP_MANUAL_TARGET"] = "2000000"

    elif "large" in model_ckpt:
        if available_devices:
            if available_devices[0] == BackendDevice.Grayskull:
                os.environ["PYBUDA_FORCE_EMULATE_HARVESTED"] = "1"
                compiler_cfg = pybuda.config._get_global_compiler_config()
                compiler_cfg.enable_auto_fusing = False
                os.environ["PYBUDA_TEMP_ELT_UNARY_ESTIMATES_LEGACY"] = "1"

    # Load ALBERT tokenizer and model from HuggingFace
    tokenizer = AlbertTokenizer.from_pretrained(model_ckpt)
    model = AlbertForMaskedLM.from_pretrained(model_ckpt)

    # Load data sample
    sample_text = ["The capital of [MASK] is Paris."] * batch_size

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
        pybuda.PyTorchModule("pt_albert_masked_lm", model),
        inputs=[input_tokens],
    )
    output = output_q.get()

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
    run_albert_masked_lm_pytorch()
