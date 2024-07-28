import os
import pybuda
import re

from transformers import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2Config
from pybuda.transformers.pipeline import pipeline as pybuda_pipeline

def parse_chat_completion(text: str):
    pattern = r'<\|im_start\|>\s*(\w+)\s*([\s\S]*?)\s*(?:<\|im_end\|>|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    messages = []
    for role, content in matches:
        messages.append({"role": role, "content": content.strip()})
    
    return messages

def run_qwen_chat():
    os.environ["TT_BACKEND_TIMEOUT"] = '0'

    # Set PyBuda configurations
    compiler_cfg = pybuda.config._get_global_compiler_config()
    compiler_cfg.default_df_override = pybuda.DataFormat.Float16_b
    compiler_cfg.enable_auto_fusing = False
    compiler_cfg.balancer_policy = "Ribbon"

    # Setup model configuration
    config = Qwen2Config.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    config.use_cache = True
    config.return_dict = True

    # Load model and tokenizer with config
    model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", config=config)
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
    tokenizer.pad_token, tokenizer.pad_token_id = (tokenizer.eos_token, tokenizer.eos_token_id)

    # Sample chat messages
    batch_messages = [
        [
            {"role": "system", "content": "You are Jim Keller, the CEO of Tenstorrent"},
            {"role": "user", "content": "Introduce yourself please!"}
        ]
    ]
    batch_size = len(batch_messages)

    # Apply chat template to each batch
    chat_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages[:batch_size]
    ]

    # Initialize pipeline
    text_generator = pybuda_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    # Inference on TT device
    responses = text_generator(
        chat_texts,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_new_tokens=512,
        num_beams=1,
        do_sample=True,
        no_repeat_ngram_size=5,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        early_stopping=True
    )

    # Display Responses
    for batch_id in range(batch_size):
        print(f"Batch: {batch_id}")
        raw_text = responses[batch_id][0]['generated_text']
        parsed_messages = parse_chat_completion(raw_text)

        for message in parsed_messages:
            print(f"{message['role'].capitalize()}: {message['content']}")
        print()

if __name__ == "__main__":
    run_qwen_chat()