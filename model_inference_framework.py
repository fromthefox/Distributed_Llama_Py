"""
this file is used to finish the inference task
"""

import model_inference_module
from model_inference_module import QKV_distribution
import threading
import torch
from model_inference_module import inference_server, dynamic_weights_dis, proportinal_allocation_dis, total_score_dis, cal_new_base_weights

def generation_loop(initial_input, max_tokens_length, model, tokenizer, config, server, allocation_list, user_config, dynamic_part, nodes_info_dict):
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
        new_base_weights = cal_new_base_weights(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
        dynamic_weights_array = dynamic_weights_dis(dynamic_weights=dynamic_part, base_weights=new_base_weights)
        scores_list = total_score_dis(nodes_info_dict, dynamic_weights_array)
        # here 128 is the unsplitted dim of the model.
        allocation_list = proportinal_allocation_dis(scores_list, 128)
        
        res = inference_server(
            model=model,
            tokenizer=tokenizer,
            config=config,
            server=server,
            input_text=current_input,
            allocation_list=allocation_list,
            user_config=user_config
        )
        next_text = res[0]
        computation_time_list = res[1]
        translation_time_list = res[2]

        print(f"Generated text: {next_text}")
        
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
