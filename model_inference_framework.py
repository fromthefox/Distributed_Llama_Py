"""
this file is used to finish the inference task
"""

import model_inference_module
from model_inference_module import QKV_distribution
import threading
import torch
from model_inference_module import inference_server, dynamic_weights_dis, proportinal_allocation_dis, total_score_dis, cal_new_base_weights, cal_new_dynamic_ratio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compute_score_module import dynamic_weights_with_performance, total_score, reallocation_needed
from function_modules import proportinal_allocation
import numpy as np

def generation_loop(initial_input, max_tokens_length, model, tokenizer, config, server, allocation_list, 
                   user_config, dynamic_part, nodes_info_dict, performance_tracker=None, 
                   enable_dynamic_reallocation=False):
    """
    Generation loop for autocontinuation with dynamic reallocation capability
    :param initial_input: initial input text
    :param max_tokens_length: maximum number of tokens to generate (read from user_config)
    :param performance_tracker: PerformanceTracker object for tracking node performance
    :param enable_dynamic_reallocation: bool, whether to enable dynamic reallocation
    :returns: the full generated text
    """
    current_input = initial_input
    generated_tokens = 0
    full_output = ""
    current_allocation_list = allocation_list.copy()  # 创建当前分配的副本
    
    # 动态重分配的参数
    reallocation_threshold = 0.15  # 重分配阈值
    min_tokens_before_reallocation = 1  # 至少生成几个token后才考虑重分配

    while True:
        # Perform single-step reasoning
        
        res = inference_server(
            model=model,
            tokenizer=tokenizer,
            config=config,
            server=server,
            input_text=current_input,
            allocation_list=current_allocation_list,
            user_config=user_config
        )
        next_text = res[0]
        computation_time_list = res[1]
        translation_time_list = res[2]

        print(f"Generated text: {next_text}")
        print(f"Computation time: {computation_time_list}")
        print(f"Translation time: {translation_time_list}")
        
        # 记录性能数据
        if performance_tracker is not None and enable_dynamic_reallocation:
            for node_id, (comp_time, comm_time) in enumerate(zip(computation_time_list, translation_time_list)):
                performance_tracker.record_compute_time(node_id, comp_time)
                performance_tracker.record_comm_time(node_id, comm_time)
        
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

        # 动态重分配逻辑
        if (enable_dynamic_reallocation and performance_tracker is not None and 
            generated_tokens >= min_tokens_before_reallocation):
            
            # 使用性能数据更新权重
            updated_weights, updated_dynamic_part = dynamic_weights_with_performance(
                nodes_info_dict=nodes_info_dict,
                performance_tracker=performance_tracker,
                base_weights=np.array([0.5, 0.4, 0.1]),
                dynamic_ratio=0.7,
                performance_weight=0.3
            )
            
            # 计算新的分数
            new_scores = total_score(nodes_info_dict, updated_weights)
            
            # 判断是否需要重分配
            need_reallocation, new_allocation = reallocation_needed(
                current_allocation_list, 
                new_scores, 
                threshold=reallocation_threshold
            )
            
            if need_reallocation:
                print(f"Token {generated_tokens}: Performing dynamic reallocation")
                print(f"Old allocation: {current_allocation_list}")
                print(f"New allocation: {new_allocation}")
                print(f"Updated weights: {updated_weights}")
                
                # 更新分配列表
                current_allocation_list = new_allocation
                dynamic_part = updated_dynamic_part
                
                # 这里可以添加通知worker节点重新分配的逻辑
                # notify_workers_for_reallocation(server, new_allocation)
            else:
                print(f"Token {generated_tokens}: No reallocation needed")
        
        # 原有的注释代码保持不变作为备用
        # new_base_weights = cal_new_base_weights(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
        # new_dynamic_ratio = cal_new_dynamic_ratio(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
        # dynamic_weights_array = dynamic_weights_dis(dynamic_weights=dynamic_part, base_weights=new_base_weights)
        # scores_list = total_score_dis(nodes_info_dict, dynamic_weights_array)
        # # here 128 is the unsplitted dim of the model.
        # allocation_list = proportinal_allocation_dis(scores_list, 128)

    return full_output

