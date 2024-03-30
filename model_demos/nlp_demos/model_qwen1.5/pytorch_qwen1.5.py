import pybuda

from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2Config
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

"""
=== Models ===

Qwen/Qwen1.5-0.5B
Qwen/Qwen1.5-0.5B-Chat
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int4
Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8
"""

model_name = "Qwen/Qwen1.5-0.5B"


def run_qwen_causal_lm(max_length=1024, top_p=0.9, top_k=50, temperature=0.7):
    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda._C.DataFormat.Float16_b

    # Config
    config = Qwen2Config.from_pretrained(model_name)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False

    config = Qwen2Config(**config_dict)

    # Load the model and tokenizer
    model = Qwen2ForCausalLM.from_pretrained(
        model_name, config=config, device_map=device)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name, device_map=device)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Example usage
    prompt = "What is a neural network?"

    # Initialize pipeline
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    # Inference
    output = text_generator(
        prompt,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Display output
    print("OUTPUT:\n", output[0]["generated_text"])


if __name__ == "__main__":
    run_qwen_causal_lm(
        max_length=1024,
        top_p=0.9,
        top_k=50,
        temperature=0.7
    )