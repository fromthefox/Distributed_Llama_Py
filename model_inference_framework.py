"""
this file is used to finish the inference task
"""

import model_inference_module
from model_inference_module import QKV_distribution
import threading
import torch

def generation_loop(initial_input, max_tokens, model, tokenizer, config, server, allocation_list, user_config):
    """
    自动续写的生成循环
    :param initial_input: 初始输入文本
    :param max_tokens: 最大生成token数（从user_config读取）
    :returns: 完整生成的文本
    """
    current_input = initial_input
    generated_tokens = 0
    full_output = ""

    while True:
        # 执行单步推理
        next_text = inference_server(
            model=model,
            tokenizer=tokenizer,
            config=config,
            server=server,
            input_text=current_input,
            allocation_list=allocation_list,
            user_config=user_config
        )
        
        # 更新状态
        full_output += next_text
        generated_tokens += 1
        current_input += next_text  # 拼接新生成的文本
        
        # 停止条件检测
        stop_conditions = [
            generated_tokens >= max_tokens,          # 超过最大长度
            next_text in ["</s>", "<|endoftext|>"]   # 常见结束符（根据实际tokenizer调整）
        ]
        
        if any(stop_conditions):
            break

    return full_output
