import pybuda
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline
from transformers import AutoTokenizer, Phi3Config
# from transformers import AutoModelForCausalLM
from nlp_demos.phi3.utils.modeling_phi3 import Phi3ForCausalLM


def run_phi3_causal_lm(variant="microsoft/Phi-3-mini-4k-instruct", batch_size=1):
    # Set PyBUDA configuration parameters
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.enable_enumerate_u_kt = False

    # Variants: "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"
    model_ckpt = variant

    # set model configurations
    config = Phi3Config.from_pretrained(model_ckpt)
    config_dict = config.to_dict()
    config_dict["return_dict"] = False
    config_dict["use_cache"] = False
    config = Phi3Config(**config_dict)

    # Load tokenizer and model from HuggingFace
    # model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config)
    model = Phi3ForCausalLM.from_pretrained(model_ckpt, config=config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.pad_token = tokenizer.eos_token

    # Input sample
    prefix_text = ["Happy birth"] * batch_size

    # Create text generator object
    text_generator = pybuda_pipeline(
        "text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size
    )

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
    run_phi3_causal_lm()
