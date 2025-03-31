"""
this file is used to finish the inference task
"""

import model_inference_module
from model_inference_module import QKV_distribution
import threading
import torch
from model_inference_module import inference_server

def generation_loop(initial_input, max_tokens_length, model, tokenizer, config, server, allocation_list, user_config):
    """
    Generation loop for autocontinuation
    :param initial_input: initial input text
    :param max_tokens_length: maximum number of tokens to generate (read from user_config)
    :returns: the full generated text
    """
    current_input = initial_input
    generated_tokens = 0
    full_output = ""

    while True:
        # Perform single-step reasoning
        next_text = inference_server(
            model=model,
            tokenizer=tokenizer,
            config=config,
            server=server,
            input_text=current_input,
            allocation_list=allocation_list,
            user_config=user_config
        )
        
        # Update Status
        full_output += next_text
        generated_tokens += 1
        current_input += next_text  # Splicing the newly generated text
        
        # Stop condition detection
        stop_conditions = [
            generated_tokens >= max_tokens_length,          # Exceeds maximum length
            next_text in ["</s>", "<|endoftext|>"]   # Common terminators (adjusted to the actual tokenizer)
        ]
        
        if any(stop_conditions):
            break

    return full_output
