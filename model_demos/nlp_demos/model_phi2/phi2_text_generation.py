import os
import pybuda

from transformers import PhiForCausalLM, AutoTokenizer, PhiConfig
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

model_name = "microsoft/phi-2"

def run_qwen_causal_lm(max_length=1024, top_p=0.9, top_k=50, temperature=0.7):
    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"

    # Load the model configuration
    config = PhiConfig.from_pretrained(model_name)
    config.use_cache = True

    # Load the model and tokenizer with the updated config
    model = PhiForCausalLM.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Example usage
    prompt = ["My name is Jimmy and"]

    # Initialize pipeline
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Inference
    output = text_generator(
        prompt,
        do_sample=True,
        num_beams=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Display output
    print("OUTPUT:\n", output[0][0]["generated_text"])


if __name__ == "__main__":
    run_qwen_causal_lm(
        max_length=100,
        top_p=0.7,
        top_k=50,
        temperature=0.7
    )