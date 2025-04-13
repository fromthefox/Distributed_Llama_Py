31, 2301, 2301], 'memory': [8, 8, 350, 350], 'bandwidth': [160, 160, 80, 80]}
computation_time_list = [0.8460162929996926, 0.8853536270001321, 0.5917502362281084, 0.5708553295116872]
translation_time_list = [175.27169220706753, 236.51146707308698, 365.3462711638422, 357.4909621703555]
dynamic_part = [0.33333333, 0.33333333, 0.33333333]
new_base_weights = cal_new_base_weights(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
new_dynamic_ratio = cal_new_dynamic_ratio(computation_time_list=computation_time_list, translation_time_list=translation_time_list)
dynamic_weights_array = dynamic_weights_dis(dynamic_weights=dynamic_part, base_weights=new_base_weights, dynamic_ratio=new_dynamic_ratio)
scores_list = total_score_dis(nodes_info_dict, dynamic_weights_array)
allocation_list = proportinal_allocation_dis(scores_list, 128)
print(f"new_dynamic_ratio: {new_dynamic_ratio}")
print(f"Dynamic Part: {dynamic_part}")
print(f"Dynamic Weights: {dynamic_weights_array}")
print(f"Final Scores: {scores_list}")
print(f"Allocation List: {allocation_list}")